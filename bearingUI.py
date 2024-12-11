import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load required files
rf_model = joblib.load("C:\\Users\\arund\\Desktop\\Bearing ML Model\\random_forest_model.pkl")
label_encoder = joblib.load("C:\\Users\\arund\\Desktop\\Bearing ML Model\\label_encoder.pkl")
scaler = joblib.load("C:\\Users\\arund\\Desktop\\Bearing ML Model\\scaler.pkl")

def visualize_class_distribution(data):
    class_counts = data['Category'].value_counts()
    st.subheader("Class Distribution in Uploaded Dataset")
    st.bar_chart(class_counts)

def visualize_feature_importance():
    st.subheader("Feature Importance")
    feature_importance = rf_model.feature_importances_
    feature_names = ['RMS', 'KU', 'CF', 'IF', 'PP', 'EN']
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, feature_importance, color='teal')
    plt.xlabel("Importance")
    plt.title("Feature Importance in Random Forest Model")
    st.pyplot(plt)

# Streamlit App Layout
st.set_page_config(page_title="Bearing Fault Detection", layout="wide")

# Header Section
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("C:\\Users\\arund\\Desktop\\Bearing ML Model\\aicte_logo.png", width=100)
with col2:
    st.markdown("""<h2 style='text-align: center;'>AICTE QIP PG Certificate Program on Machine Learning</h2>""", unsafe_allow_html=True)
    st.markdown("""<h4 style='text-align: center;'>Centre: Indian Institute of Science (IISc), Bengaluru</h4>""", unsafe_allow_html=True)
    st.markdown("""<h3 style='text-align: center;'>Project Title: Machine Learning Application for Bearing Fault Detection</h3>""", unsafe_allow_html=True)
    
with col3:
    st.image("C:\\Users\\arund\\Desktop\\Bearing ML Model\\iisc_logo.png", width=100)

# Description Section
st.sidebar.markdown("""<h4>Dataset Features:</h4>""", unsafe_allow_html=True)
st.sidebar.markdown("""
- **RMS:** Root Mean Square
- **KU:** Kurtosis
- **CF:** Crest Factor
- **IF:** Impulse Factor
- **PP:** Peak-to-Peak
- **EN:** Energy
""")
st.sidebar.markdown("**Model:** Random Forest Classifier")
st.sidebar.markdown("**Accuracy Achieved:** 87%")
st.sidebar.markdown("**Developed by:** Arun C Dixit U & Nithin M")
st.sidebar.markdown("**Contact:** arundixitu@vvce.ac.in | 9900479762")

# Main Layout
st.write("This application detects faults in bearing systems using a Random Forest model trained on vibration dataset features. Upload a dataset or enter values manually to predict the bearing condition.")

# File Upload Section
st.markdown("""<h4>Upload Your Vibration Dataset (CSV format):</h4>""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop file here or browse", type=["csv"])

# Manual Input Section
st.markdown("**Enter Variable Values:**", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
with col1:
    rms = st.number_input("RMS", value=0.0, step=0.1)
with col2:
    ku = st.number_input("KU", value=0.0, step=0.1)
with col3:
    cf = st.number_input("CF", value=0.0, step=0.1)
with col4:
    impulse = st.number_input("IF", value=0.0, step=0.1)
with col5:
    pp = st.number_input("PP", value=0.0, step=0.1)
with col6:
    energy = st.number_input("EN", value=0.0, step=0.1)

probability_threshold = st.slider("Set Probability Threshold (%)", min_value=0, max_value=100, value=50, step=1)

if st.button("Predict for Manual Input"):
    manual_input = np.array([[rms, ku, cf, impulse, pp, energy]])
    scaled_input = scaler.transform(manual_input)
    probabilities = rf_model.predict_proba(scaled_input)[0] * 100
    predicted_class = rf_model.predict(scaled_input)[0]
    class_name = label_encoder.inverse_transform([predicted_class])[0]
    explanations = {
        "HB": "Healthy Bearing: Normal operation.",
        "IRD": "Inner Race Defect: Fault in the inner race.",
        "ORD": "Outer Race Defect: Fault in the outer race.",
        "RED": "Rolling Element Defect: Fault in rolling elements."
    }
    if probabilities.max() >= probability_threshold:
        st.write(f"### Predicted Class: {class_name} ({probabilities.max():.2f}%)")
        st.write(f"**Explanation:** {explanations[class_name]}")
    else:
        st.write(f"### Prediction Confidence Below Threshold ({probabilities.max():.2f}%)")

# File Processing Section
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    try:
        features = data[['RMS', 'KU', 'CF', 'IF', 'PP', 'EN']]
        scaled_features = scaler.transform(features)
        predictions = rf_model.predict(scaled_features)
        probabilities = rf_model.predict_proba(scaled_features) * 100
        data['Prediction'] = label_encoder.inverse_transform(predictions)
        data['Probability (%)'] = probabilities.max(axis=1)
        st.write("### Prediction Results")
        st.dataframe(data)
        st.download_button("Download Results", data.to_csv(index=False), file_name="results.csv", mime="text/csv")

        visualize_class_distribution(data)
        visualize_feature_importance()
    except Exception as e:
        st.error(f"Error processing file: {e}")

# Footer Section
st.markdown("""<hr style='border: 1px solid gray;'>""", unsafe_allow_html=True)
st.markdown(
    """<p style='text-align: center;'>Developed as a part of project work for AICTE QIP PG Certificate Program on Machine Learning at IISc, Bengaluru</p>""",
    unsafe_allow_html=True
)
