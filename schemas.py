from pydantic import BaseModel

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