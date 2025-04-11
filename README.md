# 🧠 AI4CKD — API de Prédiction Intelligente pour la Maladie Rénale Chronique (CKD)

Bienvenue dans **AI4CKD**, une API RESTful construite avec ❤️ et **FastAPI** pour prédire les stades de la maladie rénale chronique, et discuter intelligemment des résultats via un chatbot LLM alimenté par **Google Gemini**.

> 🏆 Ce projet a été conçu par **l’Équipe 1** dans le cadre de l’**hackathon AI4CKD** organisé par l’**IFRI**.

---

## 🚀 Fonctionnalités principales

- 🔍 **Prédiction automatique du stade CKD** à partir de données patient
- 🧪 **Prétraitement intelligent** des données (Box-Cox, Yeo-Johnson, Scalers)
- 🤖 **Chatbot médical contextuel** basé sur les résultats précédents
- 📎 Support de l'import de fichiers CSV (optionnel)
- 🧠 Compatible avec LLM (LangChain + Google Generative AI)

---

## 📦 Installation

```bash
git clone https://github.com/Yoannoza/AI4CKD.git
cd ai4ckd-api
pip install -r requirements.txt
```

Crée un fichier `.env` avec ta clé API Gemini :

```
GEMINI_API=ta_cle_google_api
```

Assure-toi également d’avoir les fichiers suivants dans le répertoire :
- `best_model.pkl` — modèle entraîné
- `std_scaler.save` — StandardScaler sauvegardé
- `rob_scaler.save` — RobustScaler sauvegardé

---

## 🧪 Démarrer l’API

```bash
uvicorn main:app --reload
```

L’interface interactive Swagger est accessible ici :  
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧬 Endpoints

### `/predict` – POST

Prédit le stade CKD d’un patient.

#### Corps attendu (`application/json`)

```json
{
  "Sexe": 1,
  "Age": 45,
  "Creatinine": 1.2,
  "PathologiesVirales": 0,
  "HTAFamiliale": 1,
  "Glaucome": 0
}
```

#### Réponse

```json
{
  "stage": "Stade 2",
  "stage_name": "CKD 2",
  "stage_num": 1,
  "recommendation": "Contrôle des facteurs de risque et suivi biannuel.",
  "explanation": "Votre patient est classé au Stade 2..."
}
```

---

### `/chatbot` – POST

Discutez avec un chatbot contextuel alimenté par les dernières prédictions.

#### Corps attendu

```json
{
  "message": "Quel est le risque à ce stade ?"
}
```

#### Réponse

```json
{
  "response": "À ce stade, un suivi régulier est recommandé...",
  "has_prediction_context": true
}
```

---

### `/import_csv` – POST

Upload d’un fichier CSV contenant des données patients.

---

## 💡 Design original

Ce projet fusionne **ML**, **LLM**, **explicabilité**, et **UX médicale** :

- ✨ API centrée sur l’**interprétation clinique**
- 🧠 Agent LLM avec mémoire de session et prompt adaptatif
- 📊 Pipeline robuste pour les transformations statistiques

---

## 🛠️ Tech Stack

- FastAPI
- Scikit-learn, Pandas, SciPy
- LangChain + Gemini API (LLM)
- joblib, uvicorn
- Dotenv

---

## 🧪 Tester le modèle

Un test simple est inclus au lancement pour vérifier le chargement du modèle :

```bash
python main.py
```

---

## 🤝 Contribuer

Tu veux l'améliorer ? Ajouter une DB ? Multilingue ? Go for it.

---

## ⚠️ Avertissement

Cette API est un **outil d’aide à la décision** pour professionnels de santé. Elle **ne remplace pas un avis médical**.

---

## 📬 Contact

Made with ❤️ par **l’Équipe 1** 
📍 Projet développé pour l’hackathon **AI4CKD - IFRI 2025**  
