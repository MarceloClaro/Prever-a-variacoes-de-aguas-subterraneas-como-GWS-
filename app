import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Função para carregar os dados e exibir informações iniciais
def carregar_dados(file):
    data = pd.read_csv(file)
    st.write("Informações dos Dados:")
    st.write(data.describe())
    
    # Tratar valores nulos
    if data.isnull().sum().sum() > 0:
        st.warning("Os dados contêm valores nulos. Eles serão preenchidos com a média.")
        data.fillna(data.mean(), inplace=True)
    
    return data

# Função para normalizar os dados
def normalizar_dados(X_train, X_val, X_test, metodo='MinMax'):
    scaler = MinMaxScaler() if metodo == 'MinMax' else StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

# Função para calcular o erro médio
def calcular_erro_medio(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))

# Função para calcular todas as métricas de regressão
def calcular_metricas_regressao(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    erro_medio = calcular_erro_medio(y_test, y_pred)
    return mse, mape, r2, erro_medio

# Exibição das métricas de comparação
def exibir_metricas(mse, mape, r2, erro_medio):
    st.write(f"**Erro Médio Quadrado (MSE):** {mse:.4f}")
    st.write(f"**Erro Percentual Absoluto Médio (MAPE):** {mape:.4f}")
    st.write(f"**Coeficiente de Determinação (R²):** {r2:.4f}")
    st.write(f"**Erro Médio:** {erro_medio:.4f}")

# Comparar os resultados do modelo com o artigo
def comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo):
    st.write("### Comparação com o Artigo:")
    st.write(f"MSE no Artigo: {mse_artigo}, MSE do Modelo: {mse:.4f}")
    st.write(f"MAPE no Artigo: {mape_artigo}, MAPE do Modelo: {mape:.4f}")
    st.write(f"R² no Artigo: {r2_artigo}, R² do Modelo: {r2:.4f}")
    st.write(f"Erro Médio no Artigo: {erro_medio_artigo}, Erro Médio do Modelo: {erro_medio:.4f}")
    
    # Alerta visual se houver grande diferença entre o artigo e os resultados
    if abs(r2 - r2_artigo) > 0.1:
        st.warning("Atenção: O R² do modelo está significativamente diferente do valor apresentado no artigo.")
    if mse > mse_artigo * 1.2:
        st.warning("O MSE do modelo é muito maior que o do artigo. Considere ajustar os hiperparâmetros.")

# Função para exibir a importância das variáveis (features)
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

# Plotar gráfico de dispersão para prever vs. real
def plotar_dispersao_previsoes(y_test, y_pred):
    st.write("### Dispersão: Previsões vs Valores Reais")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Previsões')
    plt.title('Previsões vs Valores Reais')
    st.pyplot(fig)

# Função para configurar o sidebar e permitir inserção manual dos valores do artigo
def configurar_sidebar():
    st.sidebar.title("Configurações dos Modelos")
    modelo_tipo = st.sidebar.selectbox('Escolha o Modelo', ['XGBoost', 'Random Forest', 'CatBoost'])
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

# Função principal para carregar e processar os dados
def main():
    # Configuração do sidebar
    modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, early_stopping_rounds, l2_reg, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo = configurar_sidebar()

    # Upload do arquivo CSV
    uploaded_file = st.sidebar.file_uploader("Carregue seus dados em CSV", type=["csv"])

    if uploaded_file:
        # Carregar os dados
        data = carregar_dados(uploaded_file)
        st.write("Pré-visualização dos Dados Carregados:")
        st.write(data.head())

        # Separar as variáveis de entrada (X) e a variável alvo (y)
        if 'target' in data.columns:
            X = data.drop(columns=['target'])
            y = data['target']
        else:
            st.error("O arquivo deve conter a coluna 'target' como variável alvo.")
            st.stop()

        # Dividir os dados em treino, validação e teste
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

        # Normalizar os dados de entrada
        metodo_normalizacao = st.sidebar.selectbox('Método de Normalização', ['MinMax', 'Z-Score'])
        X_train_scaled, X_val_scaled, X_test_scaled = normalizar_dados(X_train, X_val, X_test, metodo=metodo_normalizacao)

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
            else:
                modelo = RandomForestRegressor(**modelo_kwargs)

            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)

            # Calcular métricas de desempenho de regressão
            mse, mape, r2, erro_medio = calcular_metricas_regressao(y_test, y_pred)
            exibir_metricas(mse, mape, r2, erro_medio)

            # Comparação com os valores do artigo
            comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo)

            # Exibir gráficos
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
            else:
                modelo = RandomForestClassifier(**modelo_kwargs)

            modelo.fit(X_train_scaled, y_train)
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

            # Exibir histogramas das previsões
            plotar_histograma_previsoes(y_test, y_pred)

        # Exibir a importância das features
        mostrar_importancia_features(modelo, X)

# Executar a função principal
if __name__ == "__main__":
    main()
