import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load required files
rf_model = joblib.load("new_best_random_forest_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Top 4 features selected during training
TOP_FEATURES = ['PP', 'RMS', 'KU', 'IF']

def preprocess_input(data):
    """
    Function to preprocess input data:
    - Automatically select top 4 features (PP, RMS, KU, IF).
    - Scale the data using the saved StandardScaler.
    """
    selected_data = data[TOP_FEATURES]  # Select top 4 features
    return scaler.transform(selected_data)  # Scale the data

# Streamlit App Layout
st.set_page_config(page_title="Bearing Fault Detection", layout="wide")

# Header Section
st.markdown("<h2 style='text-align: center;'>Bearing Fault Detection Using ML</h2>", unsafe_allow_html=True)

# Manual Input Section
st.markdown("**Enter Individual Variable Values:**", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    rms = st.number_input("RMS (Root Mean Square)", value=0.0, step=0.1)
    ku = st.number_input("KU (Kurtosis)", value=0.0, step=0.1)

with col2:
    cf = st.number_input("CF (Crest Factor)", value=0.0, step=0.1)
    impulse = st.number_input("IF (Impulse Factor)", value=0.0, step=0.1)

with col3:
    pp = st.number_input("PP (Peak-to-Peak)", value=0.0, step=0.1)
    energy = st.number_input("EN (Energy)", value=0.0, step=0.1)

if st.button("Predict for Manual Input"):
    # Create DataFrame with all 6 features
    manual_input = pd.DataFrame([[rms, ku, cf, impulse, pp, energy]],
                                columns=['RMS', 'KU', 'CF', 'IF', 'PP', 'EN'])
    # Automatically select only the required features
    scaled_input = preprocess_input(manual_input)
    prediction = rf_model.predict(scaled_input)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    st.write(f"### Predicted Class: **{predicted_class}**")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file with all features (RMS, KU, CF, IF, PP, EN):", type=["csv"])

if uploaded_file:
    try:
        # Read uploaded data
        data = pd.read_csv(uploaded_file)

        # Automatically select and preprocess only the top 4 features
        scaled_features = preprocess_input(data)

        # Make predictions
        predictions = rf_model.predict(scaled_features)
        data['Prediction'] = label_encoder.inverse_transform(predictions)

        # Display results
        st.write("### Prediction Results:")
        st.dataframe(data)
        st.download_button("Download Results", data.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer Section
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed for AICTE QIP PG Certificate Program</p>", unsafe_allow_html=True)
