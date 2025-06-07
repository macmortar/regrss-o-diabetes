import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Título do App
st.title('🔍 Simulador de Progressão da Diabetes')

# Carregar dados
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Treinar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(150,100), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Sidebar - Interface para entrada de dados
st.sidebar.header('📋 Defina os valores das variáveis')

# Função para coletar os valores
def user_input_features():
    age = st.sidebar.slider('Age (idade normalizada)', float(X['age'].min()), float(X['age'].max()), 0.0)
    sex = st.sidebar.slider('Sex (0=feminino, 1=masculino)', float(X['sex'].min()), float(X['sex'].max()), 0.0)
    bmi = st.sidebar.slider('BMI (Índice de Massa Corporal)', float(X['bmi'].min()), float(X['bmi'].max()), 0.0)
    bp = st.sidebar.slider('Blood Pressure (Pressão arterial média)', float(X['bp'].min()), float(X['bp'].max()), 0.0)
    s1 = st.sidebar.slider('S1 (medida sérica 1)', float(X['s1'].min()), float(X['s1'].max()), 0.0)
    s2 = st.sidebar.slider('S2 (medida sérica 2)', float(X['s2'].min()), float(X['s2'].max()), 0.0)
    s3 = st.sidebar.slider('S3 (medida sérica 3)', float(X['s3'].min()), float(X['s3'].max()), 0.0)
    s4 = st.sidebar.slider('S4 (medida sérica 4)', float(X['s4'].min()), float(X['s4'].max()), 0.0)
    s5 = st.sidebar.slider('S5 (medida sérica 5)', float(X['s5'].min()), float(X['s5'].max()), 0.0)
    s6 = st.sidebar.slider('S6 (medida sérica 6)', float(X['s6'].min()), float(X['s6'].max()), 0.0)
    
    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'bp': bp,
        's1': s1,
        's2': s2,
        's3': s3,
        's4': s4,
        's5': s5,
        's6': s6
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Coletar os inputs do usuário
input_df = user_input_features()

# Botão para gerar a previsão
if st.sidebar.button('Prever Progressão da Diabetes'):
    # Previsão
    prediction = model.predict(input_df)
    # Exibir resultado
    st.subheader('🩺 Resultado da Previsão')
    st.write(f'➡️ Progressão estimada da diabetes (após 1 ano): **{prediction[0]:.2f}**')
    
    st.write('Valores de entrada:')
    st.write(input_df)

# Rodapé
st.write('---')
st.caption('App desenvolvido com Streamlit + Scikit-learn')