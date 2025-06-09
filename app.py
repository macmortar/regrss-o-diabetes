# -*- coding: utf-8 -*-
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

# Função para criar e salvar o modelo
def criarModelo():
    # Carregar o dataset Diabetes
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target)

    # Dividir em treino/teste (pode só usar treino aqui, pois o app é só para predição depois)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar e treinar o modelo (pode ajustar os parâmetros conforme seu GridSearch)
    modelo = MLPRegressor(hidden_layer_sizes=(150,100), activation='relu',
                          solver='adam', learning_rate_init=0.001, max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)

    # Salvar o modelo
    joblib.dump(modelo, 'model.pkl')

# Treinar e salvar o modelo ao iniciar o app
criarModelo()

# Carregar o modelo salvo
modelo_carregado = joblib.load('model.pkl')

# Título da aplicação
st.title('Predição da Progressão da Diabetes')

st.write("""
Este aplicativo utiliza um modelo de regressão (MLPRegressor) para prever a progressão da diabetes um ano após a medição inicial.
Preencha os valores abaixo e clique em "Prever" para obter a previsão.
""")

# Inputs do usuário
age = st.number_input('Idade normalizada (age)', value=0.0)
sex = st.number_input('Sexo (0 = feminino, 1 = masculino)', value=0.0)
bmi = st.number_input('Índice de massa corporal (bmi)', value=0.0)
bp = st.number_input('Pressão arterial média (bp)', value=0.0)
s1 = st.number_input('Medida sérica S1', value=0.0)
s2 = st.number_input('Medida sérica S2', value=0.0)
s3 = st.number_input('Medida sérica S3', value=0.0)
s4 = st.number_input('Medida sérica S4', value=0.0)
s5 = st.number_input('Medida sérica S5', value=0.0)
s6 = st.number_input('Medida sérica S6', value=0.0)

# Preparar os dados de entrada
entrada = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

# Fazer a previsão
if st.button('Prever'):
    previsao = modelo_carregado.predict(entrada)
    st.write(f'Progressão prevista da diabetes (após 1 ano): **{previsao[0]:.2f}**')
