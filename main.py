import pandas as pd 
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import io
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
import streamlit as st 
import joblib
from datetime import datetime

from helper import GovalMachineLearning


st.title("AutoML") 
st.text("This app aims to automate the machine learning training model using a custom data")
# st.text("This app aims to automate the machine learning training model using a custom data")
st.subheader("Step 1 : ")


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    try : 
        modify = st.checkbox("Add filters")
        if not modify:
            return df
        df = df.copy()
        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
        modification_container = st.container()
        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]
    except : 
        st.warning("This column cannot be filtered, might be filled with missing values to prevent this : ")
        st.text("ensure the column contains a value ")
    return df

def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers transform numeric columns
    (logarithm, inverse, or squared). The transformations create new columns
    rather than modifying the original column.
    
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: DataFrame with new transformed columns
    """
    modify = st.checkbox("Add transformations")
    if not modify:
        return df
    
    df = df.copy()

    # Create container for transformations
    transformation_container = st.container()
    with transformation_container:
        to_transform_columns = st.multiselect("Select columns to transform", df.columns)
        
        for column in to_transform_columns:
            if is_numeric_dtype(df[column]):
                transformation_type = st.selectbox(
                    f"Choose transformation for {column}",
                    ("None", "Logarithm", "Inverse", "Squared"),
                )
                transformation_name = {
                    'Logarithm' : 'ln',
                    'Inverse' : 'inv', 
                    'Squared' : 'sq', 
                    "None" : " "
                    
                }
                new_column_name = f"{transformation_name[transformation_type]}_{column}"

                if transformation_type == "Logarithm":
                    try:
                        df[new_column_name] = np.log(df[column])
                        st.write(f"Column '{new_column_name}' created.")
                    except ValueError:
                        st.warning(f"Cannot apply logarithm to column {column}. It may contain non-positive values.")
                
                elif transformation_type == "Inverse":
                    try:
                        df[new_column_name] = 1 / df[column]
                        st.write(f"Column '{new_column_name}' created.")
                    except ZeroDivisionError:
                        st.warning(f"Cannot apply inverse to column {column}. It may contain zeros.")
                
                elif transformation_type == "Squared":
                    df[new_column_name] = df[column] ** 2
                    st.write(f"Column '{new_column_name}' created.")
            else:
                st.warning(f"Column {column} is not numeric and cannot be transformed.")
    
    return df


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
        
        




