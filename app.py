import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)

# Função para carregar e tratar os dados
def carregar_dados(file):
    try:
        data = pd.read_csv(file, parse_dates=True)
        st.write("Informações dos Dados:")
        st.write(data.describe())
        
        # Verificar e tratar valores nulos
        if data.isnull().sum().sum() > 0:
            st.warning("Os dados contêm valores nulos. Eles serão preenchidos com a média ou moda, conforme apropriado.")
            num_cols = data.select_dtypes(include=['float64', 'int64']).columns
            cat_cols = data.select_dtypes(include=['object']).columns
            data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
            data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
        
        return data
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        logging.exception("Erro ao carregar os dados")
        return None

# Função para extrair características temporais
def extrair_caracteristicas_temporais(dataframe, coluna_tempo):
    try:
        dataframe[coluna_tempo] = pd.to_datetime(dataframe[coluna_tempo])
        dataframe['ano'] = dataframe[coluna_tempo].dt.year
        dataframe['mes'] = dataframe[coluna_tempo].dt.month
        dataframe['dia'] = dataframe[coluna_tempo].dt.day
        dataframe['dia_da_semana'] = dataframe[coluna_tempo].dt.weekday
        dataframe['estacao'] = dataframe[coluna_tempo].dt.month % 12 // 3 + 1
        return dataframe
    except Exception as e:
        st.error(f"Erro ao extrair características temporais: {e}")
        logging.exception("Erro ao extrair características temporais")
        return dataframe

# Função para preparar os dados (pré-processamento)
def preparar_dados(X, y):
    try:
        # Identificar colunas numéricas e categóricas
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        # Pipelines para colunas numéricas e categóricas
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('power_transformer', PowerTransformer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Transformador de colunas
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        # Aplicar pré-processamento
        X_processed = preprocessor.fit_transform(X)
        return X_processed, preprocessor
    except Exception as e:
        st.error(f"Erro no pré-processamento dos dados: {e}")
        logging.exception("Erro no pré-processamento dos dados")
        return None, None

# Função para calcular métricas de regressão
def calcular_metricas_regressao(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, mape, r2

# Exibir métricas de comparação
def exibir_metricas_regressao(mse, rmse, mae, mape, r2):
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.4f}")
    st.write(f"**Coeficiente de Determinação (R²):** {r2:.4f}")

# Função para exibir a importância das features
def mostrar_importancia_features(modelo, X, preprocessor):
    try:
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
            # Obter os nomes das features após o pré-processamento
            num_features = preprocessor.transformers_[0][2]
            cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out()
            features = np.concatenate([num_features, cat_features])
            importancia_df = pd.DataFrame({'Features': features, 'Importância': importancias})
            importancia_df = importancia_df.sort_values(by='Importância', ascending=False)
            
            st.write("### Importância das Variáveis (Features):")
            fig, ax = plt.subplots(figsize=(10, 6))
            importancia_df.plot(kind='barh', x='Features', y='Importância', legend=False, ax=ax)
            plt.title("Importância das Features")
            plt.xlabel("Importância")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.warning("Não foi possível exibir a importância das features.")
        logging.exception("Erro ao exibir a importância das features")

# Função para otimização de hiperparâmetros com Randomized Search
def otimizar_modelo(modelo, X_train, y_train, param_distributions):
    try:
        scoring = 'neg_mean_squared_error'
        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=param_distributions,
            n_iter=20,
            cv=tscv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        st.write("Melhores parâmetros encontrados:", random_search.best_params_)
        return random_search.best_estimator_
    except Exception as e:
        st.error(f"Erro na otimização do modelo: {e}")
        logging.exception("Erro na otimização do modelo")
        return modelo

# Função para empilhamento de modelos (Stacking)
def stacking_model():
    try:
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('xgb', XGBRegressor(n_estimators=100)),
            ('cat', CatBoostRegressor(n_estimators=100, verbose=0))
        ]
        final_estimator = XGBRegressor(n_estimators=100)
        modelo = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
        return modelo
    except Exception as e:
        st.error(f"Erro ao criar o modelo de empilhamento: {e}")
        logging.exception("Erro ao criar o modelo de empilhamento")
        return None

# Função para configurar o sidebar
def configurar_sidebar():
    st.sidebar.title("Configurações do Modelo")
    modelo_tipo = st.sidebar.selectbox('Escolha o Modelo', ['XGBoost', 'Random Forest', 'CatBoost', 'Stacking'])
    
    n_estimators = st.sidebar.slider('Número de Árvores (n_estimators)', 100, 1000, 300, step=50)
    learning_rate = st.sidebar.slider('Taxa de Aprendizado (learning_rate)', 0.01, 0.3, 0.1)
    max_depth = st.sidebar.slider('Profundidade Máxima (max_depth)', 3, 20, 6)
    l2_reg = st.sidebar.slider('Regularização L2 (reg_lambda)', 0.0, 1.0, 0.1)
    
    # Configurações do XGBoost (opcionais)
    if modelo_tipo == 'XGBoost':
        subsample = st.sidebar.slider('Subsample (Taxa de Amostragem)', 0.5, 1.0, 0.8)
        colsample_bytree = st.sidebar.slider('ColSample ByTree', 0.5, 1.0, 0.8)
    else:
        subsample = None
        colsample_bytree = None

    return modelo_tipo, n_estimators, learning_rate, max_depth, l2_reg, subsample, colsample_bytree

# Função principal
def main():
    st.title("Previsão de Variações de Águas Subterrâneas (GWS)")
    modelo_tipo, n_estimators, learning_rate, max_depth, l2_reg, subsample, colsample_bytree = configurar_sidebar()

    uploaded_file = st.sidebar.file_uploader("Carregue seus dados em CSV", type=["csv"])

    if uploaded_file:
        data = carregar_dados(uploaded_file)
        if data is not None:
            st.write("Pré-visualização dos Dados Carregados:")
            st.write(data.head())

            # Selecionar a coluna de data, se disponível
            colunas_data = [col for col in data.columns if 'data' in col.lower() or 'date' in col.lower()]
            if colunas_data:
                coluna_data = st.sidebar.selectbox('Selecione a coluna de data', colunas_data)
                data = extrair_caracteristicas_temporais(data, coluna_data)
            else:
                st.warning("Não foi encontrada nenhuma coluna de data para extrair características temporais.")

            # Selecionar a coluna alvo
            coluna_alvo = st.sidebar.selectbox('Selecione a coluna alvo (GWS)', data.columns)

            # Usar a coluna selecionada como variável alvo
            if coluna_alvo in data.columns:
                X = data.drop(columns=[coluna_alvo])
                y = data[coluna_alvo]
            else:
                st.error(f"A coluna {coluna_alvo} não foi encontrada no arquivo CSV.")
                st.stop()

            # Remover outliers (opcional)
            remover_outliers_toggle = st.sidebar.checkbox("Remover Outliers?", value=False)
            if remover_outliers_toggle:
                X, y = remover_outliers(X, y)
                st.write("Outliers removidos.")

            # Pré-processar os dados
            X_processed, preprocessor = preparar_dados(X, y)
            if X_processed is None:
                st.stop()

            # Dividir os dados em conjuntos de treino e teste mantendo a ordem temporal
            split_index = int(0.8 * len(X_processed))
            X_train_full, X_test = X_processed[:split_index], X_processed[split_index:]
            y_train_full, y_test = y[:split_index], y[split_index:]

            # Escolher o modelo
            modelo_kwargs = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'reg_lambda': l2_reg
            }
            if subsample and colsample_bytree:
                modelo_kwargs['subsample'] = subsample
                modelo_kwargs['colsample_bytree'] = colsample_bytree

            if modelo_tipo == 'XGBoost':
                modelo = XGBRegressor(**modelo_kwargs)
            elif modelo_tipo == 'CatBoost':
                modelo = CatBoostRegressor(**modelo_kwargs, verbose=0)
            elif modelo_tipo == 'Random Forest':
                modelo = RandomForestRegressor(**modelo_kwargs)
            elif modelo_tipo == 'Stacking':
                modelo = stacking_model()

            # Aplicar Randomized Search para otimização de hiperparâmetros
            if st.sidebar.checkbox('Otimizar Hiperparâmetros?'):
                param_distributions = {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [4, 6, 8, 10, 12],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'reg_lambda': [0.0, 0.1, 0.5, 1.0]
                }
                modelo = otimizar_modelo(modelo, X_train_full, y_train_full, param_distributions)

            # Treinar o modelo usando TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for train_index, val_index in tscv.split(X_train_full):
                X_train, X_val = X_train_full[train_index], X_train_full[val_index]
                y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
                modelo.fit(X_train, y_train)
                y_pred_val = modelo.predict(X_val)
                mse_val = mean_squared_error(y_val, y_pred_val)
                scores.append(mse_val)
            st.write(f"Validação Cruzada Temporal (MSE): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

            # Ajustar o modelo nos dados completos de treinamento
            modelo.fit(X_train_full, y_train_full)

            # Fazer previsões no conjunto de teste
            y_pred = modelo.predict(X_test)

            # Calcular métricas de desempenho de regressão
            mse, rmse, mae, mape, r2 = calcular_metricas_regressao(y_test, y_pred)
            exibir_metricas_regressao(mse, rmse, mae, mape, r2)

            # Exibir a importância das features
            mostrar_importancia_features(modelo, X, preprocessor)

            # Exibir gráfico de dispersão de previsões vs valores reais
            plotar_dispersao_previsoes(y_test, y_pred)

            # Exibir gráfico de resíduos
            plotar_residuos(y_test, y_pred)

            # Análise de resíduos
            testar_estacionariedade(y_test - y_pred)

        else:
            st.write("Erro ao processar os dados.")
    else:
        st.write("Por favor, carregue um arquivo CSV para começar.")

# Função para exibir gráfico de dispersão (para regressão)
def plotar_dispersao_previsoes(y_test, y_pred):
    st.write("### Dispersão: Previsões vs Valores Reais")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Previsões')
    plt.title('Previsões vs Valores Reais')
    st.pyplot(fig)

# Função para plotar resíduos
def plotar_residuos(y_test, y_pred):
    st.write("### Resíduos: Valores Reais vs Resíduos")
    residuos = y_test - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuos, edgecolors=(0, 0, 0))
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Previsões')
    ax.set_ylabel('Resíduos')
    plt.title('Resíduos vs Previsões')
    st.pyplot(fig)

# Função para testar estacionariedade dos resíduos
from statsmodels.tsa.stattools import adfuller

def testar_estacionariedade(residuos):
    st.write("### Teste de Estacionariedade dos Resíduos")
    result = adfuller(residuos)
    st.write(f"Estatística ADF: {result[0]:.4f}")
    st.write(f"Valor-p: {result[1]:.4f}")
    for key, value in result[4].items():
        st.write(f"Valor Crítico ({key}): {value:.4f}")
    if result[1] < 0.05:
        st.write("Os resíduos são estacionários (rejeita-se a hipótese nula).")
    else:
        st.write("Os resíduos não são estacionários (não se rejeita a hipótese nula).")

# Executar a função principal
if __name__ == "__main__":
    main()
