from pydantic import BaseModel
from typing import Optional

# Modèle Pydantic pour les données patient
class PatientData(BaseModel):
    Sexe: int  # 0 pour Femme, 1 pour Homme
    Age: int
    Creatinine: float  # Créatinine (mg/L)
    PathologiesVirales: int  # Personnels Médicaux/Pathologies virales (HB, HC, HIV)
    HTAFamiliale: int  # Personnels Familiaux/HTA
    Glaucome: int  # Pathologies/Glaucome

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"  # Identifiant de session pour gérer plusieurs conversations
