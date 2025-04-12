from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import recommandation, get_feature_explanation, test_model_validation
from schemas import PatientData, ChatMessage
from typing import Optional, List, Dict, Any
import uvicorn
import json
import os
from joblib import load
import numpy as np
from scipy.stats import boxcox, yeojohnson
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from views import prediction, get_chatbot_agent, chatbote
from utils import get_llm

# Imports LangChain pour Google Chat et prompt personnalisé
from langchain_openai import AzureChatOpenAI
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

    
    
@app.post("/predict")
async def predict(patient_data: PatientData):
    """
    Reçoit les données patient, exécute la prédiction CKD et génère une explication.
    """
    
    return await prediction(patient_data)


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API AI4CKD. Consultez /docs pour voir les endpoints."}

# Modèle pour les messages du chatbot

@app.post("/chatbot")
async def chatbot(chat_message: ChatMessage):
    """
    Endpoint pour interagir avec le chatbot LLM avec intégration des résultats de prédiction.
    """
    return await chatbote(chat_message)


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
    print("=== Test du modèle ===")
    test_model_validation()
    uvicorn.run(app, host="0.0.0.0", port=8000)
