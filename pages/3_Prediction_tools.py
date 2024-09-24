import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Function to read the model from the uploaded file
def read_model(obj):
    return joblib.load(obj)

# Streamlit app interface
st.header("Under development :p")

# File uploader for ML model
mlmodel = st.file_uploader("Upload your ML model", type=['sav'])

# Load the model when uploaded 
if mlmodel:
    mlmodel = read_model(mlmodel)
    st.success("Model successfully loaded!")
    st.header("Input variables now")
    input_data = {}
    for feature in mlmodel.feature_names_in_:
        input_data[feature] = st.text_input(f"Value for: {feature}")
    predict_button = st.button("Predict using model!")
    if predict_button:
        try:
            input_df = pd.DataFrame([input_data], dtype=float)
            prediction = mlmodel.predict(input_df)
            st.write(f"Prediction in Normal Value: {prediction[0]}")
            st.write(f"Prediction in Exponential Value: {np.exp(prediction[0])}")   
        except Exception as e:
            st.error(f"Error making prediction: {e}")
