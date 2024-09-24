import pandas as pd 
import io
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
import streamlit as st 
import joblib
from datetime import datetime

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
    print(st.session_state['predict_button'])

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
    predict_button = st.button("start training", on_click=click_predict())
    if (st.session_state['predict_button'] == True) :  
        try : 
            product = GovalMachineLearning(df,x,y,RandomForestRegressor()) 
            mlmodel = product[0]
            kfoldresult = product[1]
            joblib_filename = "Machinelearningmodel{}.sav".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
            model_buffer = io.BytesIO()
            joblib.dump(mlmodel, model_buffer)
            model_buffer.seek(0)          
            col1, col2 = st.columns(2)
            with col1:
                    st.write("KFOLD evaluation:")
                    st.dataframe(kfoldresult)
            with col2: 
                st.download_button(
                    label="Download trained model",
                    data=model_buffer,
                    file_name=joblib_filename,
                    mime='application/octet-stream'
                )
        except : 
            st.warning("model ready to prepare")
        
        




