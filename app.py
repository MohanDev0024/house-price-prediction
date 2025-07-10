import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and feature names
model = joblib.load('house_price_model.pkl')
feature_names = joblib.load('feature_names.pkl')  # This must be saved during training

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")

st.markdown("Fill in the details below to estimate the price of a house:")

# Collect input features from user
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
GarageCars = st.slider("Garage Capacity (cars)", 0, 4, 2)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)
FullBath = st.slider("Number of Full Bathrooms", 0, 4, 2)

# Create a DataFrame for the user input
input_data = {
    'OverallQual': OverallQual,
    'GrLivArea': GrLivArea,
    'GarageCars': GarageCars,
    'TotalBsmtSF': TotalBsmtSF,
    'FullBath': FullBath
}

input_df = pd.DataFrame([input_data])

# Prepare full input with all expected feature columns
# Add missing features with 0 values
full_input = pd.DataFrame(columns=feature_names)
full_input = pd.concat([full_input, input_df], ignore_index=True)
full_input = full_input.fillna(0)  # Fill any missing columns with zero

# Predict and display result
if st.button("Predict Price"):
    prediction = model.predict(full_input)[0]
    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")
