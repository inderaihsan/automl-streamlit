from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score,KFold
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import streamlit as st
import pandas as pd




def fsd_2(y_true, y_pred):
  y_true = np.exp(y_true)
  y_pred = np.exp(y_pred)
  pe = (y_true - y_pred)/y_true
  ape = np.abs(pe)
  return np.std(ape)

def evaluate(actual, predicted, squared = False, model = None):
    """
    Calculate various regression evaluation metrics, including FSD (Forecast Standard Deviation).

    Parameters:
    actual (array-like): The actual target values.
    predicted (array-like): The predicted target values.

    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    if (squared == True):
        actual = np.exp(actual)
        predicted = np.exp(predicted)
    #calculate percentage error and absolute percentage error
    pe = ((actual-predicted)/actual)
    ape = np.abs(pe)
    n = len(actual)
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actual - predicted))
    mae = mean_absolute_error(actual, predicted)

    # Calculate MSE (Mean Squared Error)
    mse = mean_squared_error(actual ,predicted)

    # Calculate R-squared (R2)
    r2 = r2_score(actual, predicted)

    # Calculate MAPE (Median Absolute Percentage Error)
    mape = np.median(ape)


    # Calculate FSD (Forecast Standard Deviation)
    fsd = np.std(ape)


    #pe10 and rt20 :

    r20 = [x for x in ape if x>=0.2]
    r10 = [x for x in ape if x<=0.1]
    rt20 = len(r20)/n
    pe10 = len(r10)/n
    # Create a dictionary to store the metrics

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'MAPE': mape,
        'FSD': fsd,
        'PE10' : pe10,
        'RT20' : rt20

    }
    return metrics


def GovalMachineLearning(data, X, y, algorithm) :
  columns = X
  columns.append(y)
#   if data[columns].isna().values.any():
  st.warning("Checking missing/infinity value and  attempting to remove them...")
  data.replace([np.inf, -np.inf], np.nan, inplace=True)
  data.dropna(subset=[*X, y], inplace=True)
  st.write("Missing and infinity values removal successful. Current number of rows:", len(data))
  if y in X :
    loading_bar = st.progress(value = 0, text = "Model training....")
    y_name = y
    X = [x for x in X if (x!=y)]
    data_model = data[columns].dropna()
    X = data_model[X]
    y = data_model[y]
    model = algorithm.fit(X,y)
    prediction_train = model.predict(X)
    data['prediction'] = prediction_train
    st.scatter_chart(
    data,
    x="prediction",
    y=y_name,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    mod = algorithm.fit(X_train, y_train)
    prediction_test = mod.predict(X_test)
    test_score = evaluate(y_test, prediction_test, squared = True) 
    train_score = evaluate(y, prediction_train, squared = True) 
    col1, col2 = st.columns(2)

# Display tables in the respective columns
    with col1:
        st.write("Train score:")
        st.dataframe(pd.DataFrame(train_score, index = [0]).transpose())

    with col2:
        st.write("Test score (30%):")
        st.dataframe(pd.DataFrame(test_score, index = [0]).transpose())
    i=0
    evaluation_result = {
        'R2' : [],
        'Fold' : [],
        'FSD' : [],
        'PE10' : [],
        'RT20' : []
    }
    kf = KFold(n_splits=10, shuffle = True, random_state = 404)
    for train_index, test_index in kf.split(X):
    #   st.write("Fold:", i+1)
      X_train = X.iloc[train_index, :]
      y_train = y.iloc[train_index]
      X_test = X.iloc[test_index]
      y_test = y.iloc[test_index]
      mod = model.fit(X_train, y_train)
      prediction = mod.predict(X_test)
      fold_result = (evaluate(y_test, prediction, squared=True))
      evaluation_result['Fold'].append(i)
      evaluation_result['R2'].append(fold_result['R2'])
      evaluation_result['FSD'].append(fold_result['FSD'])
      evaluation_result['PE10'].append(fold_result['PE10'])
      evaluation_result['RT20'].append(fold_result['RT20'])
    #   st.write("training in KFOLD.... fold", i+1)
      i = i+1
      loading_bar.progress(i*10, text='Model training in cross validation....')
#   evaluation_result = pd.DataFrame(evaluation_result)
#   st.dataframe(evaluation_result)
  return [model, evaluation_result]