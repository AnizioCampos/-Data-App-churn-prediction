import pandas as pd
import streamlit as st
from pycaret.classification import *
from catboost import CatBoostClassifier


# loading the trained model.
model = load_model('modelo-final')
#model = pickle.load('model/modelo-final')

# carregando uma amostra dos dados.
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') 
#classifier = pickle.load(pickle_in)


# título
st.title("Data App - Churn prediction")

# subtítulo
st.markdown("This is a Data App used to display the Machine Learning solution for the customer churn prediction problem. ")


st.image('churn.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')

#cabeçalho barra lateral
st.sidebar.subheader("Set customer attributes to predict if he or she will churn of not")


# mapeando dados do usuário para cada atributo
gender = st.sidebar.selectbox("Gender", ('male', 'female'))
SeniorCitizen = st.sidebar.selectbox("Senior citizen", ('yes', 'no'))
Partner = st.sidebar.selectbox("Partner", ('yes', 'no'))
Dependents = st.sidebar.selectbox("Dependents", ('yes', 'no'))
tenure = st.sidebar.number_input('Months in the company', value=0)
PhoneService = st.sidebar.selectbox('Phone service', ('yes', 'no'))
TotalCharges = st.sidebar.number_input('Total charges', value=0)
MonthlyCharges = st.sidebar.number_input('Monthly charges', value=0)


# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()

    data_teste["gender"] = [gender]
    data_teste["SeniorCitizen"] = [SeniorCitizen]
    data_teste["Partner"] = [Partner]    
    data_teste["Dependents"] = [Dependents]
    data_teste["tenure"] = [tenure]	
    data_teste["PhoneService"] = [PhoneService]
    data_teste["TotalCharges"] = [TotalCharges]
    data_teste["MonthlyCharges"] = [MonthlyCharges]
    
    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = predict_model( model, data = data_teste)["Label"]

    st.subheader("Will the client churn?")
    
    result = str(result[0])

    st.write(result)