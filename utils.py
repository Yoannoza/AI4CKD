from schemas import PatientData
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from joblib import load
from scipy.stats import boxcox, yeojohnson
import numpy as np
import pandas as pd
import shap, json

load_dotenv()

MODEL_PATH = "models/best_model.pkl"  # Assurez-vous que ce chemin est correct


def get_llm():
    
    llm = AzureChatOpenAI(
        azure_deployment = os.getenv("DEPLOYMENT_NAME"),
        api_version = os.getenv("API_VERSION"), 
    )
    return llm

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = load(file)
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

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



def recommandation(explanation_result: dict):
    """
    Fournit une explication compréhensible et des recommandations à partir des résultats SHAP
    
    Args:
        explanation_result: Dictionnaire contenant les top features et leur contribution
    
    Returns:
        Une explication textuelle générée par LLM
    """
    
    top_features = explanation_result.get("top_features", [])
    features_text = "\n".join(
        [f"- {item['feature']}: contribution {item['contribution']}" for item in top_features]
    )
    
    system_prompt = """Tu es un expert médical spécialisé dans les maladies rénales chroniques. 
                        Ta mission est d’aider les patients à comprendre les résultats d’un modèle de prédiction, 
                        et à leur fournir des explications claires, pédagogiques et rassurantes.

                        Tu dois :
                        - Expliquer les facteurs qui ont influencé la prédiction (à partir d’indicateurs médicaux)
                        - Traduire les termes médicaux de manière simple
                        - Donner des recommandations concrètes, adaptées à la situation, pour aider le patient à améliorer ou surveiller sa santé rénale

                        Sois bienveillant, informatif, et évite le jargon médical inutile.
                    """
    
    user_prompt = f"""
                    Voici les résultats de l'analyse des facteurs ayant influencé la prédiction du stade de la maladie rénale chronique chez un patient.

                    Les trois indicateurs les plus influents sont :

                    {features_text}
                    
                    Réponds avec un dictionnaire JSON qui contient les clés suivantes :

                   1. "explication" : Une explication simple des facteurs influents, avec leur signification.
                   2. "recommandation" : Des recommandations pratiques adaptées pour chaque facteur.
                   
                   Pour les recommandations, ne prends pas chaque variable individuellement, va directement aux recommandation. Comme valeur de la clé, ce sera une chaîne de carctère qui comporte la recommandation
                   L'explication doit clairement spécifié comment chaque caractéristique ont impacté le résultat et évite de montrer des valeurs au patient car ça n'a aucune importance, il n'est pas du domaine.
                    Assure-toi que la réponse soit au format JSON, avec chaque clé correspondante à son explication et recommandation.
                    Assure-toi que la structure du JSON soit correcte et n'inclus que le JSON, sans autre texte.

                    Le but est que le patient comprenne mieux sa situation et sache quoi faire ensuite.
                    
                    """
    
    messages=[
            (
                "system",
                system_prompt
            ),
            (
                "human", user_prompt
            )
            ]
    while True:
        response = get_llm().invoke(messages)
        response = response.content
        try:
            response_dict = json.loads(response.strip())
            break
            # print(response_dict, type(response_dict))  # Affiche le dictionnaire et son type
        except json.JSONDecodeError:
            print("La réponse n'est pas un JSON valide.")
            continue
    return response_dict
    


def get_feature_explanation(input_data: dict, prediction) -> dict:
    """
    Fonction pour expliquer les résultats du modèle
    
    Args:
        input_data: "données entrées par l'utilisateur"
        prediction: "Résultat de la prédiction"
    
    Returns:
        une dictionnaire qui contient les éléments qui ont impactés la prédiction
    """
    print(input_data)
    df_input = pd.DataFrame([input_data])
    explainer = shap.TreeExplainer(load_model())
    shap_values = explainer.shap_values(df_input)
    # On prend la classe prédite
    predicted_class = prediction
    shap_contributions = shap_values[0, :, predicted_class]
    
    print(shap_contributions)

    # Retourne les 3 features les plus influentes
    
    feature_impact = sorted(
        zip(df_input.columns, shap_contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    return {
        "top_features": [
            {"feature": f, "contribution": float(round(c, 4))}
            for f, c in feature_impact
        ]
    }
    

def transform_age(age_value):
    try:
        if age_value <= 0:
            raise ValueError("Age doit être strictement positif pour Box-Cox")
        transformed, _ = boxcox(np.array([age_value, age_value + 0.1]))  # éviter la constance
        return transformed[0]
    except Exception as e:
        print(f"⚠️ Échec Box-Cox sur Age ({age_value}) — retour sans transformation. Erreur : {e}")
        try:
            transformed, _ = yeojohnson(np.array([age_value, age_value + 0.1]))
            return transformed[0]
        except Exception as e2:
            raise ValueError(f"❌ Age invalide ({age_value}) : transformation impossible. Erreur : {e2}")
 
 
def transform_creatinine(creat_value):
    try:
        if creat_value <= 0:
            raise ValueError("Créatinine doit être > 0 pour Box-Cox")
        transformed, _ = boxcox(np.array([creat_value, creat_value + 0.1]))
        return transformed[0]
    except Exception as e:
        print(f"⚠️ Échec Box-Cox sur Créatinine ({creat_value}) — tentative Yeo-Johnson. Erreur : {e}")
        try:
            transformed, _ = yeojohnson(np.array([creat_value, creat_value + 0.1]))
            return transformed[0]
        except Exception as e2:
            raise ValueError(f"❌ Créatinine invalide ({creat_value}) : transformation impossible. Erreur : {e2}")



def test_model_validation():
    """Vérifie que le modèle chargé possède les méthodes et attributs attendus."""
    try:
        # Chargement du modèle
        model = load_model()
        
        if model is None:
            print("❌ ERREUR: Impossible de charger le modèle")
            return False
        
        # Vérification des méthodes essentielles
        if not hasattr(model, 'predict'):
            print("❌ ERREUR: Le modèle n'a pas de méthode 'predict'")
            print(f"❌  TYPE: {model}")
            return False
        
        # Vérification des attributs attendus d'un RandomForest
        if hasattr(model, 'estimators_'):
            print("✅ Le modèle possède des estimateurs (RandomForest)")
        else:
            print("⚠️ AVERTISSEMENT: Le modèle ne semble pas être un RandomForest")
        
        # Test avec des données fictives
        test_data = pd.DataFrame({
            'Sexe': [1],
            # 'Personnels Médicaux/Pathologies virales (HB, HC, HIV)': [0],
            # 'Personnels Familiaux/HTA': [0],
            # 'Pathologies/Glaucome': [0],
            'Age': [50],
            'Créatinine (mg/L)': [10.0]
        })
        
        # Essai de prédiction
        try:
            prediction = model.predict(test_data)
            print(f"✅ Prédiction réussie: {prediction}")
        except Exception as e:
            print(f"❌ ERREUR lors de la prédiction: {str(e)}")
            return False
        
        print("✅ Validation du modèle réussie")
        return True
    
    except Exception as e:
        print(f"❌ ERREUR inattendue: {str(e)}")
        return False