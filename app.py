import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Título do app
st.title('Projeto de Regressão - Predição de Progressão da Diabetes')

# Carregar dados
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Mostrar dataset
if st.checkbox("Mostrar dados brutos"):
    st.write(X.head())

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sidebar para parâmetros do modelo
st.sidebar.header("Parâmetros do MLPRegressor")
hidden_layer_sizes = st.sidebar.selectbox("Camadas ocultas", [(50,), (100,), (150,100)])
activation = st.sidebar.selectbox("Função de ativação", ['relu', 'tanh'])
solver = st.sidebar.selectbox("Solver", ['adam', 'lbfgs'])
learning_rate_init = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.001, step=0.001)
max_iter = st.sidebar.selectbox("Max Iter", [500, 1000])

# Treinar modelo
model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    learning_rate_init=learning_rate_init,
    max_iter=max_iter,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Mostrar resultados
st.subheader("Métricas do Modelo")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.2f}")
