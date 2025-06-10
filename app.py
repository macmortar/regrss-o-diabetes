# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Treinar e salvar o modelo se não existir
MODEL_PATH = 'model.pkl'

@st.cache_data(show_spinner=False)
def train_and_save_model(path):
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = MLPRegressor(
        hidden_layer_sizes=(100,), activation='relu', solver='adam',
        max_iter=500, random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    return model

try:
    modelo = joblib.load(MODEL_PATH)
except FileNotFoundError:
    modelo = train_and_save_model(MODEL_PATH)

# Interface Streamlit
st.title('Predição da Progressão da Diabetes')

st.write("""
Este aplicativo treina um modelo de regressão (MLPRegressor) e, em seguida, permite prever a progressão da diabetes um ano após a medição inicial.
Os dados são carregados do conjunto `load_diabetes` da scikit-learn.

Preencha os valores abaixo e clique em "Prever" para obter a previsão.
""")

# Inputs do usuário
feature_names = load_diabetes().feature_names
inputs = []
for name in feature_names:
    val = st.number_input(f'{name}', value=0.0)
    inputs.append(val)

entrada = np.array([inputs])

if st.button('Prever'):
    previsao = modelo.predict(entrada)
    st.write(f'Progressão prevista da diabetes (após 1 ano): **{previsao[0]:.2f}**')

# Observação:
# O modelo é salvo em 'model.pkl' e recarregado em execuções subsequentes.

