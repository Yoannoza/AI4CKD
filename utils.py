from .schemas import PatientData
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

      
def get_llm():
    
    llm = AzureChatOpenAI(
        azure_deployment = os.getenv("DEPLOYMENT_NAME"),
        api_version = os.getenv("API_VERSION"), 
    )
    return llm


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

                    En te basant sur ces indicateurs, suis ces étapes:
                    1. Expliquer ce que signifient ces facteurs dans le contexte de la santé rénale
                    2. Dire comment ils ont contribué à la prédiction
                    3. Fournir des conseils pratiques ou des recommandations médicales adaptées

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
    
    response = get_llm().invoke(messages)
    return response.content
    


def get_feature_explanation(input_data: dict, prediction) -> dict:
    """
    Fonction pour expliquer les résultats du modèle
    
    Args:
        input_data: "données entrées par l'utilisateur"
        prediction: "Résultat de la prédiction"
    
    Returns:
        une dictionnaire qui contient les éléments qui ont impactés la prédiction
    """
    df_input = pd.DataFrame([input_data])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    # On prend la classe prédite
    predicted_class = model.predict(df_input)[0]
    shap_contributions = shap_values[predicted_class][0]

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


# def ask_agent(stade, input_data):
#     explanation = get_feature_explanation(input_data)

#     messages = [
#         {
#             "role": "system",
#             "content": system_prompt
#         },
#         {
#             "role": "user",
#             "content": f"""
# Voici les données du patient :

# - Stade prédit : {stade}
# - Paramètres médicaux : {input_data}
# - Explication du modèle (SHAP) : {explanation['top_features']}

# Fournis une explication claire et des recommandations adaptées.
# """
#         }
#     ]

#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=messages,
#         temperature=0.5
#     )

#     return response["choices"][0]["message"]["content"]