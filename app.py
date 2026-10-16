import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Medical Cost Predictor", page_icon="💡", layout="centered"
)

# Custom CSS for modern layout, cards, and soft shadows
st.markdown(
    """
    <style>
    .main {
        background-color: #F8FAFC;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.7rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2), 0 2px 4px -1px rgba(79, 70, 229, 0.1);
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #E2E8F0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained Linear Regression model safely using absolute pathing
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "linear.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# App Header
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="color: #1E293B; font-weight: 700;">💡 Medical Cost Predictor</h1>
        <p style="color: #64748B;">Enter the patient profile details below to estimate charges using your trained linear model.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Input Form Layout
with st.form("prediction_form"):
    st.markdown(
        "<div class='card'><h3 style='color: #1E293B; margin-bottom: 1rem;'>Patient Details</h3>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        sex = st.selectbox("Sex", ["female", "male"])
        bmi = st.number_input(
            "BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0
        )

    with col2:
        children = st.number_input(
            "Number of Children", min_value=0, max_value=10, value=0
        )
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox(
            "Region", ["southeast", "southwest", "northeast", "northwest"]
        )

    st.markdown("</div>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Calculate Prediction")

# Prediction logic
if submit_button:
    # Mapping categorical text strings to numeric indicators
    sex_val = 1 if sex == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0

    region_map = {
        "southeast": 0,
        "southwest": 1,
        "northeast": 2,
        "northwest": 3,
    }
    region_val = region_map.get(region, 0)

    # Constructing feature array matching exact feature names order: ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    features = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

    try:
        prediction = model.predict(features)
        st.markdown(
            f"""
            <div class='card' style='text-align: center; background-color: #EEF2FF; border: 1px solid #C7D2FE;'>
                <h3 style='color: #4F46E5; margin-bottom: 0.5rem;'>Estimated Charges</h3>
                <h1 style='color: #312E81; font-size: 2.2rem;'>${prediction[0]:,.2f}</h1>
            </div>
        """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Prediction Error: {e}")
