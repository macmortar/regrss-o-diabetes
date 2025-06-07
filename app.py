import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Título
st.title('Simulador de Progressão da Diabetes')

# Carregar dados
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(150,100), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Interface para o usuário colocar valores
st.sidebar.header('Defina os valores das variáveis')

age = st.sidebar.slider('Age', float(X['age'].min()), float(X['age'].max()), 0.0)
sex = st.sidebar.slider('Sex', float(X['sex'].min()), float(X['sex'].max()), 0.0)
bmi = st.sidebar.slider('BMI', float(X['bmi'].min()), float(X['bmi'].max()), 0.0)
bp = st.sidebar.slider('Blood Pressure', float(X['bp'].min()), float(X['bp'].max()), 0.0)
s1 = st.sidebar.slider('S1', float(X['s1'].min()), float(X['s1'].max()), 0.0)
s2 = st.sidebar.slider('S2', float(X['s2'].min()), float(X['s2'].max()), 0.0)
s3 = st.sidebar.slider('S3', float(X['s3'].min()), float(X['s3'].max()), 0.0)
s4 = st.sidebar.slider('S4', float(X['s4'].min()), float(X['s4'].max()), 0.0)
s5 = st.sidebar.slider('S5', float(X['s5'].min()), float(X['s5'].max()), 0.0)
s6 = st.sidebar.slider('S6', float(X['s6'].min()), float(X['s6'].max()), 0.0)

# Botão de previsão
if st.sidebar.button('Prever Progressão da Diabetes'):
    # Monta um array com os valores informados
    input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
    # Faz a previsão
    prediction = model.predict(input_data)
    # Mostra o resultado
    st.subheader('Resultado da Previsão')
    st.write(f'Progressão estimada da diabetes (após 1 ano): {prediction[0]:.2f}')