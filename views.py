from fastapi import HTTPException
from fastapi.responses import JSONResponse
from joblib import load
from schemas import PatientData
from utils import transform_age, transform_creatinine, load_model, get_stage_info, get_feature_explanation, recommandation, get_llm
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Charge les scalers sauvegardés
std_scaler = load("scalers/std_scaler.save")
rob_scaler = load("scalers/rob_scaler.save")


last_prediction_results = {}

async def prediction(patient_data: PatientData):
    """
    Reçoit les données patient, exécute la prédiction CKD et génère une explication.
    """

    try:
        model = load_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Erreur lors du chargement du modèle")

        # Application des transformations
        transformed_age = transform_age(patient_data.Age)
        transformed_creatinine = transform_creatinine(patient_data.Creatinine)

        # Création DataFrame
        input_data = pd.DataFrame({
            "Sexe": [patient_data.Sexe],
            "Age": [transformed_age],
            "Créatinine (mg/L)": [transformed_creatinine],
            # "PathologiesVirales": [patient_data.PathologiesVirales],
            # "HTAFamiliale": [patient_data.HTAFamiliale],
            # "Glaucome": [patient_data.Glaucome]
        })
        print(input_data)

        # Application des scalers
        input_data["Age"] = std_scaler.transform(input_data[["Age"]])
        input_data["Créatinine (mg/L)"] = rob_scaler.transform(input_data[["Créatinine (mg/L)"]])

        # Prédiction
        prediction = model.predict(input_data)
        stage_num = int(prediction[0])
        stage_info = get_stage_info(stage_num)
        
        print(prediction)

        explanation = (
            f"Votre patient est classé au {stage_info['text']} ({stage_info['name']}) de la maladie rénale chronique. "
            f"Les facteurs clés pris en compte dans cette prédiction sont:\n"
            f"- Âge: {patient_data.Age} ans\n"
            f"- Créatinine: {patient_data.Creatinine} mg/L\n"
            f"- Sexe: {'Homme' if patient_data.Sexe == 1 else 'Femme'}\n"
            f"- Présence de pathologies virales: {'Oui' if patient_data.PathologiesVirales == 1 else 'Non'}\n"
            f"- Antécédents familiaux d'HTA: {'Oui' if patient_data.HTAFamiliale == 1 else 'Non'}\n"
            f"- Glaucome: {'Oui' if patient_data.Glaucome == 1 else 'Non'}"
        )
        
        feature_explanation = get_feature_explanation({
            "Sexe": patient_data.Sexe,
            "Age" : patient_data.Age,
            "Creatine": patient_data.Creatinine
        }, stage_num)
        
        recommendation = recommandation(feature_explanation)
        
        print(recommendation)

        result = {
            "stage": stage_info['text'],
            "stage_name": stage_info['name'],
            "recommendation": recommendation["recommandation"],#stage_info['recommendation'],
            "explanation": recommendation["explication"], #explanation,
            "patient_data": {
                "Sexe": "Homme" if patient_data.Sexe == 1 else "Femme",
                "Age": patient_data.Age,
                "Creatinine": patient_data.Creatinine,
                # "PathologiesVirales": patient_data.PathologiesVirales,
                # "HTAFamiliale": patient_data.HTAFamiliale,
                # "Glaucome": patient_data.Glaucome
            }
        }
        print(result["recommendation"], result["explanation"])

        global last_prediction_results
        last_prediction_results["derniere_prediction"] = result

        return JSONResponse(
            status_code=200,
            content={
                "stage": stage_info['text'],
                "stage_name": stage_info['name'],
                "stage_num": stage_num,
                "recommendation": recommendation['recommandation'],
                "explanation": recommendation["explication"]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")



# Création du prompt template avec intégration des derniers résultats
def get_chatbot_agent(last_prediction):
    
    # Ajout des derniers résultats de prédiction s'ils existent
    last_prediction_info = ""
    if last_prediction:
        last_prediction_info = f"""
        Dernière prédiction pour le patient :
        - Stade CKD: {last_prediction.get('stage', 'Non disponible')}
        - Recommandation: {last_prediction.get('recommendation', 'Non disponible')}
        - Explication: {last_prediction.get('explanation', 'Non disponible')}
        """
        
    prompt = f"""
    Tu es un assistant médical spécialisé en néphrologie, conçu pour aider les médecins à interpréter 
    les résultats de prédiction de maladie rénale chronique (CKD).
    
    {last_prediction_info}
    
    Réponds aux questions de façon factuelle, précise et concise. 
    Évite de donner des conseils médicaux définitifs, mais propose des pistes de réflexion basées sur 
    les meilleures pratiques en néphrologie.
    """
    
    ckd_agent = create_react_agent(get_llm(), tools=[], prompt=prompt, checkpointer=memory)
    
    return ckd_agent

async def chatbote(chat_message):
    """
    Endpoint pour interagir avec le chatbot LLM avec intégration des résultats de prédiction.
    """
    try:
        # Récupération des derniers résultats de prédiction (si disponibles)
        last_prediction = last_prediction_results.get("derniere_prediction", None)
        
        # Création de la chaîne de conversation avec les derniers résultats
        agent = get_chatbot_agent(last_prediction)
        
        # Génération de la réponse
        response = agent.invoke({"messages": [{"role": "user", "content": chat_message.message}]}, {"configurable": {"thread_id": str(1)}}, stream_mode="values")
        
        return JSONResponse(
            status_code=200,
            content={
                "response": response["messages"][-1].content,
                "has_prediction_context": last_prediction is not None
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la génération de la réponse: {str(e)}"}
        )