from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Imports LangChain pour Google Chat et prompt personnalisé
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

app = FastAPI(title="API Prédiction CKD", version="1.0")

# Configuration CORS pour autoriser l'accès depuis l'application React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustez selon votre domaine de développement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage temporaire des derniers résultats de prédiction (à remplacer par une DB en production)
last_prediction_results = {}

# Chargement du modèle pkl
MODEL_PATH = "best_rf.pkl"  # Assurez-vous que ce chemin est correct

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

# Initialisation du modèle LangChain avec Google Generative AI
def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API")  # À remplacer par votre clé API
    )
    return llm

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

# Fonction pour traduire le stade numérique en texte et recommandations
def get_stage_info(stage_num):
    stage_map = {
        0: {"name": "CKD 1", "text": "Stade 1", "recommendation": "Surveillance annuelle de la fonction rénale."},
        1: {"name": "CKD 2", "text": "Stade 2", "recommendation": "Contrôle des facteurs de risque et suivi biannuel."},
        2: {"name": "CKD 3a", "text": "Stade 3a", "recommendation": "Consultation néphrologique recommandée, suivi trimestriel."},
        3: {"name": "CKD 3b", "text": "Stade 3b", "recommendation": "Prise en charge néphrologique renforcée."},
        4: {"name": "CKD 4", "text": "Stade 4", "recommendation": "Préparation potentielle aux thérapies de suppléance."},
        5: {"name": "CKD 5", "text": "Stade 5", "recommendation": "Traitement de suppléance à envisager rapidement."}
    }
    return stage_map.get(stage_num, {"name": "Indéterminé", "text": "Indéterminé", "recommendation": "Consultation néphrologique recommandée."})

# Modèle Pydantic pour les données patient
class PatientData(BaseModel):
    Sexe: int  # 0 pour Femme, 1 pour Homme
    Age: int
    Creatinine: float  # Créatinine (mg/L)
    PathologiesVirales: int  # Personnels Médicaux/Pathologies virales (HB, HC, HIV)
    HTAFamiliale: int  # Personnels Familiaux/HTA
    Glaucome: int  # Pathologies/Glaucome

# Endpoint de prédiction
@app.post("/predict")
async def predict(patient_data: PatientData):
    """
    Reçoit les données patient, exécute la prédiction CKD et génère une explication.
    """
    # Chargement du modèle
    model = load_model()
    if not model:
        raise HTTPException(status_code=500, detail="Erreur lors du chargement du modèle")
    
    # Préparation des données pour la prédiction
    input_data = pd.DataFrame({
        'Sexe': [patient_data.Sexe],
        'Personnels Médicaux/Pathologies virales (HB, HC, HIV)': [patient_data.PathologiesVirales],
        'Personnels Familiaux/HTA': [patient_data.HTAFamiliale],
        'Pathologies/Glaucome': [patient_data.Glaucome],
        'Age': [patient_data.Age],
        'Créatinine (mg/L)': [patient_data.Creatinine]
    })
    
    try:
        # Prédiction
        prediction = model.predict(input_data)
        stage_num = int(prediction[0])
        
        # Récupération des informations sur le stade
        stage_info = get_stage_info(stage_num)
        
        # Génération de l'explication
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
        
        # Stockage des résultats pour utilisation dans le chatbot
        result = {
            "stage": stage_info['text'],
            "stage_name": stage_info['name'],
            "recommendation": stage_info['recommendation'],
            "explanation": explanation,
            "patient_data": {
                "Sexe": "Homme" if patient_data.Sexe == 1 else "Femme",
                "Age": patient_data.Age,
                "Creatinine": patient_data.Creatinine,
                "PathologiesVirales": patient_data.PathologiesVirales,
                "HTAFamiliale": patient_data.HTAFamiliale,
                "Glaucome": patient_data.Glaucome
            }
        }
        
        # On stocke les résultats en utilisant un identifiant unique
        global last_prediction_results
        last_prediction_results["derniere_prediction"] = result

        return JSONResponse(
            status_code=200,
            content={
                "stage": stage_info['text'],
                "stage_name": stage_info['name'],
                "stage_num": stage_num,
                "recommendation": stage_info['recommendation'],
                "explanation": explanation
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API AI4CKD. Consultez /docs pour voir les endpoints."}

# Modèle pour les messages du chatbot
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"  # Identifiant de session pour gérer plusieurs conversations

@app.post("/chatbot")
async def chatbot(chat_message: ChatMessage):
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

# Endpoint optionnel pour l'import d'un fichier CSV
@app.post("/import_csv")
async def import_csv(file: UploadFile = File(...)):
    """
    Traite le fichier CSV et retourne les données auto-mappées.
    """
    try:
        content = await file.read()
        # Implémentez ici le traitement du fichier CSV pour extraire les données
        return JSONResponse(
            status_code=200,
            content={"message": "Fichier importé avec succès", "filename": file.filename}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)