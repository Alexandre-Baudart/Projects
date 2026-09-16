import streamlit as st
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

API_URL = "https://projects-znbj.onrender.com"

# Titre
st.title("Heart Disease Prediction Demo")

# Sliders pour 5 features visibles
st.subheader("Adjust the following parameters:")

features = {
    "num_cols": ["age", "trestbps", "chol", "thalch", "oldpeak"],
    "cat_cols": ["cp", "restecg", "slope", "ca", "thal"],
    "binary_cols": ["sex", "fbs", "exang"]
}

features_names = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "cp",
    "restecg",
    "slope",
    "ca",
    "thal",
    "sex",
    "fbs",
    "exang",
]

fr_features = {
    "age": "Age",
    "trestbps": "Pression artérielle au repos",
    "chol": "Taux de cholestérol sérique total",
    "thalch": "Fréquence cardiaque maximale atteinte",
    "oldpeak": "Dépression du segment ST induite par l'effort par rapport au repos",
    "cp": "Type de douleur thoracique",
    "restecg": "Résultats de l'électrocardiogramme au repos",
    "slope": "Pente du segment ST au max de l'effort",
    "ca": "Nombre de vaisseaux majeurs colorés par fluoroscopie",
    "thal": "Résultats de la scintigraphie myocardique au Thallium",
    "sex": "Sexe",
    "fbs": "Glycémie à jeun supérieure à 120 mg/dl",
    "exang": "Angine induite par l'effort"
}

en_features = { "age": "Age", "trestbps": "Resting Blood Pressure", "chol": "Serum Cholesterol", "thalch": "Maximum Heart Rate", "oldpeak": "ST Depression Induced by Exercise", "cp": "Chest Pain Type", "restecg": "Resting Electrocardiogram Results", "slope": "ST Segment Slope at Peak Exercise", "ca": "Number of Major Vessels Colored by Fluoroscopy", "thal": "Thallium Stress Test Result", "sex": "Sex", "fbs": "Fasting Blood Sugar > 120 mg/dL", "exang": "Exercise-Induced Angina", }

num_features_metadata = {
    "age": { "min": 10.0, "max": 99.0, "default": 50.0, "step": 1.0, },
    "trestbps": { "min": 80.0, "max": 220.0, "default": 130.0, "step": 1.0, },
    "chol": { "min": 100.0, "max": 600.0, "default": 240.0, "step": 1.0, },
    "thalch": { "min": 60.0, "max": 220.0, "default": 150.0, "step": 1.0, },
    "oldpeak": { "min": 0.0, "max": 7.0, "default": 1.0, "step": 0.1, }
}

cat_en_values = {
    "cp": [
        "Asymptomatic",
        "Non-anginal",
        "Atypical angina",
        "Angina"
    ],
    "restecg": [
        "Normal",
        "ST-T abnormally",
        "LV hypertrophy"
    ],
    "slope": [
        "Upsloping",
        "Flat",
        "Downsloping"
    ],
    "ca": [
        '0',
        '1',
        '2',
        '3'
    ],
    "thal": [
        "Normal",
        "Reversable defect",
        "Fixed defect"
    ],
}

cat_fr_values = {
    "cp": [
        "Aucune douleur à la poitrine",
        "Douleur poitrine mais pas forcément cardiaque",
        "Douleur poitrine avec certaines caractéristiques de l'angine (pas toutes)",
        "Angine de poitrine"
    ],
    "restecg": [
        "Normal",
        "Anomalie de l'onde ST-T",
        "Hypertrophie ventriculaire gauche"
    ],
    "slope": [
        "Pente ascendante",
        "Pente plate/horizontale",
        "Pente descendante"
    ],
    "ca": [
        '0',
        '1',
        '2',
        '3'
    ],
    "thal": [
        "Normal",
        "Défaut réversible",
        "Défaut irréversible"
    ],
}

bin_fr_values_to_dataset_values = {
    "Non": False,
    "Oui": True,
    "Femme": "Female",
    "Homme": "Male",
}

bin_en_values_to_dataset_values = {
    "No": False,
    "Yes": True,
    "Female": "Female",
    "Male": "Male",
}

cat_fr_values_to_dataset_values = {
    "Angine de poitrine": "typical angina",
    "Douleur poitrine avec certaines caractéristiques de l'angine (pas toutes)" : "atypical angina",
    "Douleur poitrine mais pas forcément cardiaque": "non-anginal",
    "Aucune douleur à la poitrine": "asymptomatic",
    "Normal": "normal",
    "Anomalie de l'onde ST-T": "st-t abnormality",
    "Hypertrophie ventriculaire gauche": "lv hypertrophy",
    "Pente ascendante": "upsloping",
    "Pente plate/horizontale": "flat",
    "Pente descendante": "downsloping",
    "Défaut réversible": "reversable defect",
    "Défaut irréversible": "fixed defect"
}

cat_en_values_to_dataset_values = {
    "Typical angina": "typical angina",
    "Atypical angina" : "atypical angina",
    "Non-anginal": "non-anginal",
    "Asymptomatic": "asymptomatic",
    "Normal": "normal",
    "ST-T abnormality": "st-t abnormality",
    "LV hypertrophy": "lv hypertrophy",
    "Upsloping": "upsloping",
    "Flat": "flat",
    "Downsloping": "downsloping",
    "Reversable defect": "reversable defect",
    "Fixed defect": "fixed defect"
}

inputs = {}

for feat in features_names:

    # Variables numériques

    if feat in features["num_cols"]:
        metadata = num_features_metadata[feat]

        inputs[feat] = st.slider(
            en_features[feat].capitalize(),
            min_value=metadata["min"],
            max_value=metadata["max"],
            value=metadata["default"],
            step=metadata["step"],
            key=f"slider_{feat}"
        )

    # Variables catégorielles

    elif feat in features["cat_cols"]:
        option = st.selectbox( en_features[feat].capitalize(), options=cat_en_values[feat], key=f"select_{feat}")

        if feat == "ca":
            inputs[feat] = option
        else: inputs[feat] = cat_en_values_to_dataset_values[option]

    # Variables binaires

    elif feat in features["binary_cols"]:
        options = ( ["Female", "Male"] if feat == "sex" else ["No", "Yes"] )
        option = st.selectbox( en_features[feat].capitalize(), options=options, key=f"select_{feat}" )
        
        inputs[feat] = bin_en_values_to_dataset_values[option]
        
    # DataFrame final

input_df = pd.DataFrame([inputs])
input_df = input_df[features_names]

# Construction de l'input
if st.button("Prédire") :
    payload = {
        "data": input_df.to_dict(orient="records")
    }

    # Prediction & probabilities

    response = requests.post(
        f"{API_URL}/ml/predict", # http://127.0.0.1:8000/ml/predict
        json=payload,
        timeout=10
    )

    response.raise_for_status()

    result = response.json()
    pred = result["predictions"][0]
    probs = result["probabilities"][0]

    logger.info(result)

    # Display

    if pred == 0:
        st.success(
            f"**No heart disease predicted (based on the model)** – "
            f"Probability: **{probs[0]:.2%}**"
        )
    else:
        st.error(
            f"**Potential heart disease predicted (based on the model)** – "
            f"Probability: **{probs[1]:.2%}**"
        )

# Commande : python -m streamlit run project_app.py