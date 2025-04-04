from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json
from dotenv import load_dotenv
load_dotenv()

# Imports LangChain pour Google Chat et prompt personnalisé
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage

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

# Initialisation du modèle LangChain avec Google Generative AI
def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API")  # À remplacer par votre clé API
    )
    return llm

# Création du prompt template avec intégration des derniers résultats
def get_chatbot_chain(last_prediction=None):
    system_template = """
    Tu es un assistant médical spécialisé en néphrologie, conçu pour aider les médecins à interpréter 
    les résultats de prédiction de maladie rénale chronique (CKD).
    
    {last_prediction_info}
    
    Réponds aux questions de façon factuelle, précise et concise. 
    Évite de donner des conseils médicaux définitifs, mais propose des pistes de réflexion basées sur 
    les meilleures pratiques en néphrologie.
    """
    
    # Ajout des derniers résultats de prédiction s'ils existent
    last_prediction_info = ""
    if last_prediction:
        last_prediction_info = f"""
        Dernière prédiction pour le patient :
        - Stade CKD: {last_prediction.get('stage', 'Non disponible')}
        - Recommandation: {last_prediction.get('recommendation', 'Non disponible')}
        - Explication: {last_prediction.get('explanation', 'Non disponible')}
        """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        llm=get_llm(),
        prompt=chat_prompt,
        memory=memory,
        verbose=True
    )
    
    return conversation

# Modèle Pydantic pour les données patient
class PatientData(BaseModel):
    age: int
    sexe: str  # "Homme" ou "Femme"
    poids: float
    taille: float
    creatinine: float
    egfr: float
    proteinurie: Optional[str] = None  # "Faible", "Modérée", "Élevée"
    albuminurie: Optional[float] = None
    uree: Optional[float] = None
    # Ajoutez ici les autres variables (jusqu'à 40-50 champs)
    hypertension: Optional[bool] = None
    diabete: Optional[bool] = None
    cardio: Optional[bool] = None
    antecedents_familiaux: Optional[bool] = None
    ains: Optional[bool] = None
    fumeur: Optional[str] = None  # "Actif", "Ancien", "Jamais"

# Endpoint de prédiction
@app.post("/predict")
async def predict(patient_data: PatientData):
    """
    Reçoit les données patient, exécute la prédiction CKD et génère une explication.
    """
    # Exemple de logique de prédiction (à adapter à votre modèle)
    if patient_data.egfr >= 90:
        stage = "Stade 1"
        recommendation = "Surveillance classique."
    elif 60 <= patient_data.egfr < 90:
        stage = "Stade 2"
        recommendation = "Surveillance et suivi régulier."
    elif 30 <= patient_data.egfr < 60:
        stage = "Stade 3"
        recommendation = "Consultez un néphrologue dans les prochaines semaines."
    elif 15 <= patient_data.egfr < 30:
        stage = "Stade 4"
        recommendation = "Prise en charge spécialisée recommandée."
    else:
        stage = "Stade 5"
        recommendation = "Prise en charge urgente requise."

    # Exemple d'explication générée par le LLM
    explanation = (
        f"Votre patient est classé {stage} en raison d'un egfr de {patient_data.egfr} mL/min/1.73m². "
        "Les facteurs clés sont une créatinine élevée et une protéinurie modérée."
    )
    
    # Stockage des résultats pour utilisation dans le chatbot
    result = {
        "stage": stage,
        "recommendation": recommendation,
        "explanation": explanation,
        "patient_data": patient_data.dict()
    }
    
    # On stocke les résultats en utilisant un identifiant unique (on pourrait utiliser un ID patient)
    # Pour cet exemple, on utilise simplement "derniere_prediction"
    global last_prediction_results
    last_prediction_results["derniere_prediction"] = result

    return JSONResponse(
        status_code=200,
        content={
            "stage": stage,
            "recommendation": recommendation,
            "explanation": explanation
        }
    )

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
        conversation = get_chatbot_chain(last_prediction)
        
        # Génération de la réponse
        response = conversation.predict(input=chat_message.message)
        
        return JSONResponse(
            status_code=200,
            content={
                "response": response,
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