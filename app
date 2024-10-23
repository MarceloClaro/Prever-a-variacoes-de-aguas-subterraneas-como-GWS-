import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_percentage_error, r2_score, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso

# Função para carregar e tratar os dados
def carregar_dados(file):
    data = pd.read_csv(file)
    st.write("Informações dos Dados:")
    st.write(data.describe())
    
    # Verificar e tratar valores nulos
    if data.isnull().sum().sum() > 0:
        st.warning("Os dados contêm valores nulos. Eles serão preenchidos com a média.")
        data.fillna(data.mean(), inplace=True)
    
    return data

# Função para normalizar os dados
def normalizar_dados(X_train, X_val, X_test, metodo='MinMax'):
    scaler = MinMaxScaler() if metodo == 'MinMax' else StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

# Função para calcular métricas de regressão
def calcular_metricas_regressao(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    erro_medio = np.mean(np.abs(y_test - y_pred))
    return mse, mape, r2, erro_medio

# Exibir métricas de comparação
def exibir_metricas(mse, mape, r2, erro_medio):
    st.write(f"**Erro Médio Quadrado (MSE):** {mse:.4f}")
    st.write(f"**Erro Percentual Absoluto Médio (MAPE):** {mape:.4f}")
    st.write(f"**Coeficiente de Determinação (R²):** {r2:.4f}")
    st.write(f"**Erro Médio:** {erro_medio:.4f}")

# Comparar com o artigo
def comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo):
    st.write("### Comparação com o Artigo:")
    st.write(f"MSE no Artigo: {mse_artigo}, MSE do Modelo: {mse:.4f}")
    st.write(f"MAPE no Artigo: {mape_artigo}, MAPE do Modelo: {mape:.4f}")
    st.write(f"R² no Artigo: {r2_artigo}, R² do Modelo: {r2:.4f}")
    st.write(f"Erro Médio no Artigo: {erro_medio_artigo}, Erro Médio do Modelo: {erro_medio:.4f}")
    
    if abs(r2 - r2_artigo) > 0.1:
        st.warning("Atenção: O R² do modelo está significativamente diferente do valor apresentado no artigo.")
    if mse > mse_artigo * 1.2:
        st.warning("O MSE do modelo é muito maior que o do artigo. Considere ajustar os hiperparâmetros.")

# Função para exibir a importância das features
def mostrar_importancia_features(modelo, X):
    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
        features = X.columns
        importancia_df = pd.DataFrame({'Features': features, 'Importância': importancias})
        importancia_df = importancia_df.sort_values(by='Importância', ascending=False)
        
        st.write("### Importância das Variáveis (Features):")
        fig, ax = plt.subplots()
        importancia_df.plot(kind='bar', x='Features', y='Importância', legend=False, ax=ax)
        plt.title("Importância das Features")
        plt.ylabel("Importância")
        plt.tight_layout()
        st.pyplot(fig)

# Função para otimização de hiperparâmetros com Grid Search
def grid_search_model(modelo, X_train, y_train, param_grid):
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    st.write("Melhores parâmetros encontrados:", grid_search.best_params_)
    return grid_search.best_estimator_

# Função para empilhamento de modelos (Stacking)
def stacking_model(X_train, y_train):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', XGBClassifier(n_estimators=100)),
        ('cat', CatBoostClassifier(n_estimators=100, verbose=0))
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=XGBClassifier())
    stacking_model.fit(X_train, y_train)
    return stacking_model

# Função para detectar e remover outliers usando o Z-Score
def remover_outliers(X, y, limiar=3):
    z_scores = np.abs((X - X.mean()) / X.std())
    filtro = (z_scores < limiar).all(axis=1)
    return X[filtro], y[filtro]

# Função para aplicar SMOTE para balanceamento
def aplicar_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

# Função para configurar o sidebar
def configurar_sidebar():
    st.sidebar.title("Configurações dos Modelos")
    modelo_tipo = st.sidebar.selectbox('Escolha o Modelo', ['XGBoost', 'Random Forest', 'CatBoost', 'Stacking'])
    tipo_problema = st.sidebar.selectbox('Escolha o Tipo de Problema', ['Classificação', 'Regressão'])
    
    n_estimators = st.sidebar.slider('Número de Árvores (n_estimators)', 100, 1000, 300)
    learning_rate = st.sidebar.slider('Taxa de Aprendizado (learning_rate)', 0.01, 0.3, 0.1)
    max_depth = st.sidebar.slider('Profundidade Máxima (max_depth)', 3, 10, 6)
    early_stopping_rounds = st.sidebar.slider('Parada Precoce (early_stopping_rounds)', 10, 100, 50)
    l2_reg = st.sidebar.slider('Regularização L2 (Weight Decay)', 0.01, 1.0, 0.1)

    # Configurações do XGBoost (opcionais)
    if modelo_tipo == 'XGBoost':
        subsample = st.sidebar.slider('Subsample (Taxa de Amostragem)', 0.5, 1.0, 0.8)
        colsample_bytree = st.sidebar.slider('ColSample ByTree (Taxa de Colunas por Árvore)', 0.5, 1.0, 0.8)

    # Valores de comparação com o artigo fornecidos pelo usuário
    st.sidebar.subheader("Valores do Artigo para Comparação")
    mse_artigo = st.sidebar.number_input('MSE do Artigo', min_value=0.0, value=0.03)
    mape_artigo = st.sidebar.number_input('MAPE do Artigo', min_value=0.0, value=0.02)
    r2_artigo = st.sidebar.number_input('R² do Artigo', min_value=0.0, max_value=1.0, value=0.95)
    erro_medio_artigo = st.sidebar.number_input('Erro Médio do Artigo', min_value=0.0, value=0.01)

    return modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, early_stopping_rounds, l2_reg, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo

# Função principal
def main():
    modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, early_stopping_rounds, l2_reg, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo = configurar_sidebar()

    uploaded_file = st.sidebar.file_uploader("Carregue seus dados em CSV", type=["csv"])

    if uploaded_file:
        data = carregar_dados(uploaded_file)
        st.write("Pré-visualização dos Dados Carregados:")
        st.write(data.head())

        if 'target' in data.columns:
            X = data.drop(columns=['target'])
            y = data['target']
        else:
            st.error("O arquivo deve conter a coluna 'target' como variável alvo.")
            st.stop()

        # Remover outliers
        X, y = remover_outliers(X, y)

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Normalizar os dados de entrada
        metodo_normalizacao = st.sidebar.selectbox('Método de Normalização', ['MinMax', 'Z-Score'])
        X_train_scaled, X_val_scaled, X_test_scaled = normalizar_dados(X_train, X_val, X_test, metodo=metodo_normalizacao)

        # Aplicar SMOTE para balanceamento em problemas de classificação
        if tipo_problema == 'Classificação':
            aplicar_smote_toggle = st.sidebar.checkbox("Aplicar SMOTE para Balanceamento?", value=False)
            if aplicar_smote_toggle:
                X_train_scaled, y_train = aplicar_smote(X_train_scaled, y_train)

        # Escolher o modelo baseado no tipo de problema
        if tipo_problema == 'Regressão':
            # Definir parâmetros do modelo de regressão
            modelo_kwargs = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'reg_lambda': l2_reg
            }

            # Treinamento do modelo de regressão
            if modelo_tipo == 'XGBoost':
                modelo_kwargs['subsample'] = st.sidebar.slider('Subsample', 0.5, 1.0, 0.8)
                modelo_kwargs['colsample_bytree'] = st.sidebar.slider('ColSample ByTree', 0.5, 1.0, 0.8)
                modelo = XGBRegressor(**modelo_kwargs)
            elif modelo_tipo == 'CatBoost':
                modelo = CatBoostRegressor(**modelo_kwargs, verbose=0)
            elif modelo_tipo == 'Random Forest':
                modelo = RandomForestRegressor(**modelo_kwargs)
            elif modelo_tipo == 'Stacking':
                modelo = stacking_model(X_train_scaled, y_train)

            # Aplicar Grid Search para otimização de hiperparâmetros
            if st.sidebar.checkbox('Otimizar Hiperparâmetros com Grid Search?'):
                param_grid = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.3],
                }
                modelo = grid_search_model(modelo, X_train_scaled, y_train, param_grid)

            # Treinar o modelo
            modelo.fit(X_train_scaled, y_train)

            # Fazer previsões no conjunto de teste
            y_pred = modelo.predict(X_test_scaled)

            # Calcular métricas de desempenho de regressão
            mse, mape, r2, erro_medio = calcular_metricas_regressao(y_test, y_pred)
            exibir_metricas(mse, mape, r2, erro_medio)

            # Comparar com os valores do artigo
            comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo)

            # Exibir a importância das features
            mostrar_importancia_features(modelo, X)

            # Exibir gráfico de dispersão de previsões vs valores reais
            plotar_dispersao_previsoes(y_test, y_pred)

        elif tipo_problema == 'Classificação':
            # Definir parâmetros do modelo de classificação
            modelo_kwargs = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'reg_lambda': l2_reg
            }

            # Treinamento do modelo de classificação
            if modelo_tipo == 'XGBoost':
                modelo_kwargs['subsample'] = st.sidebar.slider('Subsample', 0.5, 1.0, 0.8)
                modelo_kwargs['colsample_bytree'] = st.sidebar.slider('ColSample ByTree', 0.5, 1.0, 0.8)
                modelo = XGBClassifier(**modelo_kwargs)
            elif modelo_tipo == 'CatBoost':
                modelo = CatBoostClassifier(**modelo_kwargs, verbose=0)
            elif modelo_tipo == 'Random Forest':
                modelo = RandomForestClassifier(**modelo_kwargs)
            elif modelo_tipo == 'Stacking':
                modelo = stacking_model(X_train_scaled, y_train)

            # Aplicar Grid Search para otimização de hiperparâmetros
            if st.sidebar.checkbox('Otimizar Hiperparâmetros com Grid Search?'):
                param_grid = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.3],
                }
                modelo = grid_search_model(modelo, X_train_scaled, y_train, param_grid)

            # Treinar o modelo
            modelo.fit(X_train_scaled, y_train)

            # Fazer previsões no conjunto de teste
            y_pred = modelo.predict(X_test_scaled)

            # Calcular métricas de classificação
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            st.write(f"**Acurácia no Conjunto de Teste:** {acc:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**Precisão:** {precision:.4f}")
            st.write(f"**Revocação:** {recall:.4f}")

            # Exibir matriz de confusão
            st.write("### Matriz de Confusão:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            plt.title('Matriz de Confusão')
            fig.colorbar(cax)
            plt.ylabel('Verdadeiros')
            plt.xlabel('Previstos')

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i][j]), va='center', ha='center')

            st.pyplot(fig)

            # Exibir a importância das features
            mostrar_importancia_features(modelo, X)

# Função para exibir gráfico de dispersão (para regressão)
def plotar_dispersao_previsoes(y_test, y_pred):
    st.write("### Dispersão: Previsões vs Valores Reais")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Previsões')
    plt.title('Previsões vs Valores Reais')
    st.pyplot(fig)

# Executar a função principal
if __name__ == "__main__":
    main()
