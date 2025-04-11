# API Prédiction CKD

Bienvenue dans le projet **API Prédiction CKD**, développé par l'équipe 1 IA lors du hackathon AI4CKD organisé par l'IFRI. Cette API permet d'estimer le stade de la maladie rénale chronique (CKD) chez un patient en fonction de plusieurs paramètres cliniques.

## Table des matières

- [API Prédiction CKD](#api-prédiction-ckd)
  - [Table des matières](#table-des-matières)
  - [Description du projet](#description-du-projet)
  - [Fonctionnalités](#fonctionnalités)
  - [Installation](#installation)
  - [Utilisation](#utilisation)
  - [Livrables](#livrables)
    - [Code source](#code-source)
    - [Interface utilisateur](#interface-utilisateur)
    - [Démonstration et rapport](#démonstration-et-rapport)
    - [Notebooks et modèles](#notebooks-et-modèles)
  - [Contribuer](#contribuer)

## Description du projet

L'API Prédiction CKD est conçue pour fournir une estimation du stade de la maladie rénale chronique chez un patient, basée sur des données cliniques telles que l'âge, le sexe, le niveau de créatinine, et d'autres facteurs de risque. L'API offre également des recommandations personnalisées en fonction du stade prédit.

## Fonctionnalités

- **Prédiction du stade CKD** : Estimation du stade de la maladie rénale chronique à partir des données patient.
- **Explications détaillées** : Fourniture d'explications sur les facteurs ayant contribué à la prédiction.
- **Recommandations personnalisées** : Suggestions adaptées au stade de CKD prédit.
- **Chatbot intégré** : Interaction avec un assistant virtuel pour des informations supplémentaires sur la CKD.

## Installation

1. Clonez le repository :

   ```bash
   git clone https://github.com/Yooannoza/AI4CKD.git
   cd AI4CKD
   ```

2. Créez un environnement virtuel et activez-le :

   ```bash
   python3 -m venv env
   source env/bin/activate  # Sur Windows, utilisez 'env\Scripts\activate'
   ```

3. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

4. Assurez-vous d'avoir un fichier `.env` contenant vos clés API et autres variables d'environnement nécessaires.

## Utilisation

1. Lancez l'API avec Uvicorn :

   ```bash
   uvicorn main:app --reload
   ```

2. Accédez à l'API à l'adresse `http://127.0.0.1:8000`.

3. La documentation interactive est disponible à `http://127.0.0.1:8000/docs`.

## Livrables

### Code source

Le code source complet de l'API est disponible dans le repository GitHub associé au projet.

### Interface utilisateur

Une interface web conviviale est disponible à l'adresse suivante :

[https://ai4ckd.vercel.app/](https://ai4ckd.vercel.app/)

### Démonstration et rapport

Une démonstration fonctionnelle de l'API, accompagnée d'un rapport détaillé, est accessible ici :

[https://drive.google.com/drive/folders/1-m5VMgGUGkVoc-7IOIwCf9q6lnZMPIgU?usp=sharing](https://drive.google.com/drive/folders/1-m5VMgGUGkVoc-7IOIwCf9q6lnZMPIgU?usp=sharing)

### Notebooks et modèles

Les notebooks Jupyter utilisés pour le développement et l'entraînement des modèles, ainsi que les modèles eux-mêmes, sont disponibles à ces liens :

- Notebook principal : [https://colab.research.google.com/drive/1igvaGyBt5VJDqGIt0hHFYG6godhanpSo?usp=sharing](https://colab.research.google.com/drive/1igvaGyBt5VJDqGIt0hHFYG6godhanpSo?usp=sharing)
- Modèles sauvegardés : [https://drive.google.com/drive/folders/1A8I4r2L3qcPlC9PB3jSbQXEJMCsnMfHM](https://drive.google.com/drive/folders/1A8I4r2L3qcPlC9PB3jSbQXEJMCsnMfHM)

## Contribuer

Les contributions sont les bienvenues. Pour proposer des améliorations :

1. Fork le repository.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/ma-fonctionnalite`).
3. Committez vos modifications (`git commit -am 'Ajoute une nouvelle fonctionnalité'`).
4. Poussez la branche (`git push origin feature/ma-fonctionnalite`).
5. Ouvrez une Pull Request.
