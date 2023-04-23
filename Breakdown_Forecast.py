# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:40:21 2023
@author: purva

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from lightgbm import LGBMClassifier

st.title('Breakdown Forecast')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

input_file = st.file_uploader("", type="csv")

y_pred = [] 

if input_file is not None:
    input_data = pd.read_csv(input_file)
#Read data
#SCADA data
    scada_df = pd.read_csv('scada_data.csv')
    scada_df['DateTime'] = pd.to_datetime(scada_df['DateTime'])
# scada_df.set_index('DateTime', inplace=True)

    fault_df = pd.read_csv('fault_data.csv')
    fault_df['DateTime'] = pd.to_datetime(fault_df['DateTime'])
# fault_df.set_index('DateTime', inplace=True)

    fault_df.Fault.unique()
# Combine scada and fault data
    df_combine = scada_df.merge(fault_df, on='Time', how='outer')

# Replace records that has no fault label (NaN) as 'NF' (no fault)
    df_combine['Fault'] = df_combine['Fault'].replace(np.nan, 'NF')


    df_nf = df_combine[df_combine.Fault=='NF'].sample(300)


    df_f = df_combine[df_combine.Fault!='NF']

# Combine no fault and faulty dataframes
    df_combine = pd.concat((df_nf, df_f), axis=0).reset_index(drop=True)


# Drop irrelevant features
    train_df = df_combine.drop(columns=['DateTime_x', 'Time', 'Error', 'WEC: ava. windspeed', 
                                    'WEC: ava. available P from wind',
                                    'WEC: ava. available P technical reasons',
                                    'WEC: ava. Available P force majeure reasons',
                                    'WEC: ava. Available P force external reasons',
                                    'WEC: max. windspeed', 'WEC: min. windspeed', 
                                    'WEC: Operating Hours', 'WEC: Production kWh',
                                    'WEC: Production minutes', 'DateTime_y'])
# Feature and target
# X = df_combine.iloc[:,3:-2]
# y = df_combine.iloc[:,-1]
    X = train_df.iloc[:,:-1]
    y = train_df.iloc[:,-1]

# Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Make pipeline of SMOTE, scaling, and classifier
    pipe = make_pipeline(SMOTE(), StandardScaler(), LGBMClassifier(random_state=42))
# Define multiple scoring metrics
    scoring = {
    'acc': 'accuracy',
    'prec_macro': 'precision_macro',
    'rec_macro': 'recall_macro',
    'f1_macro': 'f1_macro'
    }

# Stratified K-Fold 
    stratkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Return a dictionary of all scorings
    cv_scores = cross_validate(pipe, X_train, y_train, cv=stratkfold, scoring=scoring)





    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(input_data)
    st.write(y_pred)
    

# Load the trained model

# Load the input data



# Predict on test set








    # # Make predictions using the input data
    # predictions = model.predict(input_data)

    # # Display the predictions
    # st.write(predictions,input_data['DateTime'])
