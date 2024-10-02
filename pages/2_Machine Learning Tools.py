import pandas as pd 
import io
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
import streamlit as st 
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

from helper import GovalMachineLearning, filter_dataframe, transform_dataframe


st.title("AutoML") 
st.text("This app aims to automate the machine learning training model using a custom data")
# st.text("This app aims to automate the machine learning training model using a custom data")
st.subheader("Step 1 : ")

def fsd_2(y_true, y_pred):
  y_true = np.exp(y_true)
  y_pred = np.exp(y_pred)
  pe = (y_true - y_pred)/y_true
  ape = np.abs(pe)
  return np.std(ape)


def train_model(X , y, data) : 
    return RandomForestRegressor().fit(data[X], data[y])

def upload_data() : 
    uploaded = st.file_uploader("Please upload your data : ", 'xlsx')    
    return uploaded 
def click_predict() : 
    st.session_state['predict_button'] = True

st.session_state['predict_button'] = False

@st.cache_data
def load_data(data) : 
    return pd.read_excel(data) 
uploaded = upload_data()  
if (uploaded) :     
    df = load_data(uploaded)
    st.subheader("Step 2 (Optional): ")
    st.text("Data filtration") 
    df = filter_dataframe(df) 
    st.subheader("Step 3 (Optional)") 
    st.text("Data transformation")
    df = transform_dataframe(df)
    st.subheader("Step 4 select dependent and independent variable") 
    y = st.selectbox("dependent variable : ", df.columns) 
    x = st.multiselect("independent variable : ", [x for x in df.columns if x!=y])
    st.subheader("Model Random Forest Hyperparameters")
    bootstrap = st.radio("Bootstrap (True/False):", (True, False))
    max_features = st.radio("Max Features:", ('sqrt', 'log2'))
    max_depth = st.slider("Max Depth:", min_value=5, max_value=50, value=10)
    n_estimators = st.slider("Number of Estimators:", min_value=10, max_value=100, value=50)
    predict_button = st.button("start training", on_click=click_predict())
    if (predict_button) :  
        product = GovalMachineLearning(df,x,y,RandomForestRegressor(max_features=max_features , max_depth = max_depth, n_estimators = n_estimators, bootstrap = bootstrap, random_state = 42)) 
        mlmodel = product[0]
        
        kfoldresult = product[1]
        joblib_filename = "Machinelearningmodel{}.sav".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_buffer = io.BytesIO()
        joblib.dump(mlmodel, model_buffer)
        model_buffer.seek(0)          
        st.write("Try!!:")
        featimp = mlmodel.feature_importances_
        featname = mlmodel.feature_names_in_
        feature_importance_df = pd.DataFrame({'Feature': featname, 'Importance': featimp})

        # Sort the DataFrame by feature importance in descending order
        st.subheader("Feature importance")
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Create a bar plot to visualize feature importances
        plt.figure(figsize=(8, 5))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.tight_layout()  
        st.pyplot(plt)  
        plt.clf()  
        st.download_button(
            label="Download trained model",
            data=model_buffer,
            file_name=joblib_filename,
            mime='application/octet-stream'
        )

        
        




