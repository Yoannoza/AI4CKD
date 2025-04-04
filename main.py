from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="API Prédiction CKD", version="1.0")

# Configuration CORS pour autoriser l'accès depuis l'application React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustez selon votre domaine de développement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        f"Votre patient est classé {stage} en raison d’un egfr de {patient_data.egfr} mL/min/1.73m². "
        "Les facteurs clés sont une créatinine élevée et une protéinurie modérée."
    )

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


# Endpoint pour le Chatbot IA
class ChatMessage(BaseModel):
    message: str

@app.post("/chatbot")
async def chatbot(message: ChatMessage):
    """
    Endpoint pour interagir avec le chatbot LLM.
    """
    # Exemple de réponse simulée (à remplacer par l'appel à votre LLM réel)
    response = f"Réponse du LLM pour votre question : '{message.message}'"
    return JSONResponse(
        status_code=200,
        content={"response": response}
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
