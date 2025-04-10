import pytest
from fastapi.testclient import TestClient
import pickle
import pandas as pd
from unittest.mock import patch, mock_open
import os
import json

# Import de l'application
from main import app, PatientData, load_model

client = TestClient(app)

# Données de test
test_patient_data = {
    "Sexe": 1,
    "Age": 65,
    "Creatinine": 18.5,
    "PathologiesVirales": 1,
    "HTAFamiliale": 1,
    "Glaucome": 0
}

# Mock du modèle pour les tests
class MockModel:
    def predict(self, X):
        # Retourne toujours le stade 3 pour les tests
        return [3]

# Tests de base
def test_root_endpoint():
    """Test de l'endpoint racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Bienvenue" in response.json()["message"]

# Test du chargement du modèle
@patch("builtins.open", mock_open(read_data="binary_model_data"))
@patch("pickle.load")
def test_load_model(mock_pickle_load):
    """Test du chargement du modèle."""
    mock_model = MockModel()
    mock_pickle_load.return_value = mock_model
    
    model = load_model()
    assert model is not None
    assert isinstance(model, MockModel)

# Test de l'endpoint de prédiction
@patch("main.load_model")
def test_predict_endpoint(mock_load_model):
    """Test de l'endpoint de prédiction."""
    # Configuration du mock
    mock_model = MockModel()
    mock_load_model.return_value = mock_model
    
    # Appel de l'API
    response = client.post("/predict", json=test_patient_data)
    
    # Vérifications
    assert response.status_code == 200
    result = response.json()
    assert "stage" in result
    assert "stage_name" in result
    assert "recommendation" in result
    assert "explanation" in result
    assert result["stage_num"] == 3  # Le mock retourne toujours 3

# Test du comportement en cas d'erreur de modèle
@patch("main.load_model")
def test_predict_model_error(mock_load_model):
    """Test du comportement en cas d'erreur de chargement du modèle."""
    # Configuration du mock pour simuler une erreur
    mock_load_model.return_value = None
    
    # Appel de l'API
    response = client.post("/predict", json=test_patient_data)
    
    # Vérifications
    assert response.status_code == 500
    assert "detail" in response.json()

# Test de l'endpoint chatbot
@patch("main.get_llm")
@patch("langgraph.prebuilt.create_react_agent")
def test_chatbot_endpoint(mock_create_agent, mock_get_llm):
    """Test de l'endpoint chatbot."""
    # Configuration des mocks
    class MockAgent:
        def invoke(self, *args, **kwargs):
            return {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Réponse de test"}]}
    
    mock_create_agent.return_value = MockAgent()
    
    # Appel de l'API
    response = client.post("/chatbot", json={"message": "Comment interpréter ces résultats?", "session_id": "test123"})
    
    # Vérifications
    assert response.status_code == 200
    result = response.json()
    assert "response" in result
    assert result["response"] == "Réponse de test"

# Test avec données complètes en intégration
@pytest.mark.integration
def test_full_integration():
    """Test d'intégration complet (à exécuter avec un modèle réel)."""
    # Ce test nécessite que le fichier de modèle soit présent
    if not os.path.exists("best_rf.pkl"):
        pytest.skip("Fichier de modèle non trouvé, test d'intégration ignoré")
    
    # Appel de l'API de prédiction
    prediction_response = client.post("/predict", json=test_patient_data)
    assert prediction_response.status_code == 200
    
    # Appel de l'API chatbot avec le contexte de la prédiction
    chatbot_response = client.post("/chatbot", json={"message": "Quelles sont les recommandations pour ce patient?"})
    assert chatbot_response.status_code == 200
    
    # Vérification que le contexte de prédiction est utilisé
    assert chatbot_response.json()["has_prediction_context"] == True

if __name__ == "__main__":
    # Pour exécuter manuellement: pytest -v test_main.py
    pytest.main(["-v"])