import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Function to read the model from the uploaded file
@st.cache_data
def read_model(obj):
    return joblib.load(obj)

@st.cache_data
def read_file(obj) : 
    return pd.read_excel(obj)

# Streamlit app interface
st.header("Prediction tools")
st.subheader("This page dedicated to use a pre trained machine learning model to predict a new dataset")
st.subheader("Step 1 . Upload machine learning model")

# File uploader for ML model
mlmodel = st.file_uploader("Upload your ML model", type=['sav'])

# Load the model when uploaded 
if mlmodel:
    mlmodel = read_model(mlmodel)
    st.success("Model successfully loaded!")
    st.subheader("Step 2. Upload your desired data")
    dataprocess = st.file_uploader("upload your data and make sure it contains {} in the column".format(mlmodel.feature_names_in_), type = ['xlsx'])
    if(dataprocess) : 
        dataprocess = read_file(dataprocess)
        if all(feature in dataprocess.columns for feature in mlmodel.feature_names_in_): 
            dataprocess.dropna(inplace = True) 
            prediction = mlmodel.predict(dataprocess[mlmodel.feature_names_in_])
            dataprocess['prediction_exp'] = np.exp(prediction)
            dataprocess['prediction'] = prediction
            st.subheader("Step 3 fetch prediction result")
            st.write(dataprocess)
        else : 
            not_exist_feature = [x for x in mlmodel.feature_names_in_ if x not in dataprocess.columns] 
            st.error("Please provide {} in your column as it is not exist and try again".format(not_exist_feature) )
            # dataprocess['prediction'] = prediction
        
    
