# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:06:55 2023

@author: Starchild
"""
import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np

# Title
st.header("RandomForest+SMOTE ACLF death prediction model")

#input
INR=st.sidebar.number_input("INR(Norm:0.8-1.2)")
bilirubin=st.sidebar.number_input("Total bilirubin(mg/dL,Norm:0.3-1.9,model range:0-50)")
resp_rate=st.sidebar.number_input("resp_rate(bpm,Norm:12-20,model range:0-70)")
creatinine=st.sidebar.number_input("creatinine(mg/dL,Norm:0.5-1.2,model range:0-150)")
sbp=st.sidebar.number_input("sbp(mmHg,Norm:90-120,model range:0-400)")
heart_rate=st.sidebar.number_input("heart rate(bpm,Norm:60-100,model range:0-300)")
temperature=st.sidebar.number_input("temperature(°C,Norm:36.5-37.5,model range:35-50)")
mbp=st.sidebar.number_input("mbp(mmHg,Norm:70-100,model range:0-300)")
age=st.sidebar.number_input("age(year)")
albumin=st.sidebar.number_input("albumin(g/dL,Norm:3.5-5,model range:0-10)")
spo2=st.sidebar.number_input("spo2(%,Norm:95-100,model range:0-100)")
dbp=st.sidebar.number_input("dbp(mmHg,Norm:60-80,model range:0-300)")


with open('14RandomForest.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('14data_max.pkl', 'rb') as f:
    data_max = pickle.load(f)
with open('14data_min.pkl', 'rb') as f:
    data_min = pickle.load(f)
with open('14RandomForest_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    # Store inputs into dataframe
    columns = ['INR','bilirubin','resp_rate','creatinine','sbp',
                          'heart_rate', 'temperature','mbp','age','albumin',
                          'spo2','dbp']
    X = pd.DataFrame([[INR,bilirubin,resp_rate,creatinine,sbp,
                          heart_rate, temperature,mbp,age,albumin,
                          spo2,dbp]], 
                     columns =columns )
    st.write('Raw data:')
    st.dataframe(X)
    X = (X-data_min)/(data_max-data_min)
    st.write('Normalized data:')
    st.dataframe(X)
    # Get prediction
    prediction = clf.predict(X)
    pred=clf.predict_proba(X)[0][1]
    shap_values2 = explainer(X)
    
    # Output prediction
    
    st.text(f"The probability of death of the patient is {pred}.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig=shap.plots.bar(shap_values2[0])
    st.pyplot(fig)
    
    
    
    
    
    
    
