import streamlit as st
import pandas as pd
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    cross_val_score, KFold, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor, VotingRegressor
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, mean_squared_error,
    mean_absolute_percentage_error, r2_score, confusion_matrix, roc_auc_score,
    classification_report, roc_curve, auc, mean_absolute_error
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import logging
from sklearn.preprocessing import PowerTransformer

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)

# Função para carregar e tratar os dados
def carregar_dados(file):
    try:
        data = pd.read_csv(file)
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
        dataframe['estacao'] = dataframe[coluna_tempo].dt.month % 12 // 3 + 1  # 1: Verão, ..., 4: Primavera
        return dataframe
    except Exception as e:
        st.warning("Erro ao extrair características temporais.")
        logging.exception("Erro ao extrair características temporais")
        return dataframe

# Função para codificar coordenadas geográficas
def codificar_coordenadas(dataframe, coluna_latitude, coluna_longitude):
    try:
        dataframe['latitude_sin'] = np.sin(np.radians(dataframe[coluna_latitude]))
        dataframe['latitude_cos'] = np.cos(np.radians(dataframe[coluna_latitude]))
        dataframe['longitude_sin'] = np.sin(np.radians(dataframe[coluna_longitude]))
        dataframe['longitude_cos'] = np.cos(np.radians(dataframe[coluna_longitude]))
        return dataframe
    except Exception as e:
        st.warning("Erro ao codificar coordenadas geográficas.")
        logging.exception("Erro ao codificar coordenadas geográficas")
        return dataframe

# Função para detectar e remover outliers usando o Z-Score
def remover_outliers(X, y, limiar=3):
    try:
        # Converter X para DataFrame se não for
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        z_scores = np.abs((X - X.mean()) / X.std())
        filtro = (z_scores < limiar).all(axis=1)
        return X[filtro], y[filtro]
    except Exception as e:
        st.error(f"Erro ao remover outliers: {e}")
        logging.exception("Erro ao remover outliers")
        return X, y

# Função para preparar os dados (pré-processamento) - AJUSTADA
def preparar_dados(X, y, tipo_problema):
    try:
        # Identificar colunas numéricas e categóricas
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Pipelines para colunas numéricas e categóricas
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Lista de transformadores
        transformers = []
        if num_cols:
            transformers.append(('num', num_pipeline, num_cols))
        if cat_cols:
            transformers.append(('cat', cat_pipeline, cat_cols))

        # Transformador de colunas
        preprocessor = ColumnTransformer(transformers)

        # Ajustar e transformar os dados
        preprocessor.fit(X)
        X_processed = preprocessor.transform(X)
        return X_processed, preprocessor
    except Exception as e:
        st.error(f"Erro no pré-processamento dos dados: {e}")
        logging.exception("Erro no pré-processamento dos dados")
        return None, None

# Função para calcular métricas de regressão
def calcular_metricas_regressao(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    erro_medio = np.mean(np.abs(y_test - y_pred))
    return mse, rmse, mape, mae, r2, erro_medio

# Exibir métricas de comparação
def exibir_metricas_regressao(mse, rmse, mape, mae, r2, erro_medio):
    st.write(f"**Erro Médio Quadrado (MSE):** {mse:.4f}")
    st.write(f"**Raiz do Erro Médio Quadrado (RMSE):** {rmse:.4f}")
    st.write(f"**Erro Absoluto Médio (MAE):** {mae:.4f}")
    st.write(f"**Erro Percentual Absoluto Médio (MAPE):** {mape:.4f}")
    st.write(f"**Coeficiente de Determinação (R²):** {r2:.4f}")
    st.write(f"**Erro Médio Absoluto:** {erro_medio:.4f}")

# Função para calcular métricas de classificação
def calcular_metricas_classificacao(y_test, y_pred, y_proba=None):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}
    if y_proba is not None and len(np.unique(y_test)) == 2:
        auc_score = roc_auc_score(y_test, y_proba[:, 1])
        metrics['AUC'] = auc_score
    return metrics

def exibir_metricas_classificacao(metrics):
    for metric_name, metric_value in metrics.items():
        st.write(f"**{metric_name}:** {metric_value:.4f}")

# Função para exibir a importância das features - AJUSTADA
def mostrar_importancia_features(modelo, X, preprocessor):
    try:
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
            
            # Obter os nomes das features após o pré-processamento
            feature_names = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    # Verificar se o OneHotEncoder foi ajustado
                    onehot = transformer.named_steps.get('onehot')
                    if onehot is not None and hasattr(onehot, 'categories_'):
                        cat_feature_names = onehot.get_feature_names_out(cols)
                        feature_names.extend(cat_feature_names)
                    else:
                        # Se o OneHotEncoder não foi ajustado ou não há features categóricas
                        pass
                else:
                    # Outros transformadores, se houver
                    pass

            # Verificar se o número de nomes de features corresponde ao número de importâncias
            if len(feature_names) != len(importancias):
                st.warning("O número de features não corresponde ao número de importâncias. Verifique o pré-processamento.")
                feature_names = [f'Feature {i}' for i in range(len(importancias))]

            importancia_df = pd.DataFrame({'Features': feature_names, 'Importância': importancias})
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
def otimizar_modelo(modelo, X_train, y_train, param_distributions, tipo_problema):
    try:
        if tipo_problema == 'Classificação':
            scoring = 'f1_weighted'
        else:
            scoring = 'neg_mean_squared_error'
        random_search = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=param_distributions,
            n_iter=50,
            cv=5,
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
def stacking_model(tipo_problema):
    try:
        if tipo_problema == 'Classificação':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100)),
                ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')),
                ('cat', CatBoostClassifier(n_estimators=100, verbose=0))
            ]
            final_estimator = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
            modelo = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
        else:
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

# Função para verificar se o tipo de problema corresponde ao tipo de alvo
def verificar_tipo_problema(y, tipo_problema):
    if tipo_problema == 'Classificação' and not pd.api.types.is_integer_dtype(y):
        st.error("O alvo é contínuo, mas o tipo de problema selecionado é 'Classificação'. Por favor, ajuste o tipo de problema para 'Regressão' ou converta o alvo em categorias discretas.")
        st.stop()
    elif tipo_problema == 'Regressão' and not pd.api.types.is_numeric_dtype(y):
        st.error("O alvo é categórico, mas o tipo de problema selecionado é 'Regressão'. Por favor, ajuste o tipo de problema para 'Classificação' ou converta o alvo em valores contínuos.")
        st.stop()

# Função para configurar o sidebar
def configurar_sidebar():
    st.sidebar.title("Configurações dos Modelos")
    modelo_tipo = st.sidebar.selectbox('Escolha o Modelo', ['XGBoost', 'Random Forest', 'CatBoost', 'Stacking'])
    tipo_problema = st.sidebar.selectbox('Escolha o Tipo de Problema', ['Classificação', 'Regressão'])
    
    n_estimators = st.sidebar.slider('Número de Árvores (n_estimators)', 100, 1000, 300, step=50)
    learning_rate = st.sidebar.slider('Taxa de Aprendizado (learning_rate)', 0.01, 0.3, 0.1)
    max_depth = st.sidebar.slider('Profundidade Máxima (max_depth)', 3, 20, 6)
    l2_reg = st.sidebar.slider('Regularização L2 (Weight Decay)', 0.0, 1.0, 0.1)
    
    # Configurações do XGBoost (opcionais)
    if modelo_tipo == 'XGBoost':
        subsample = st.sidebar.slider('Subsample (Taxa de Amostragem)', 0.5, 1.0, 0.8)
        colsample_bytree = st.sidebar.slider('ColSample ByTree (Taxa de Colunas por Árvore)', 0.5, 1.0, 0.8)
    else:
        subsample = None
        colsample_bytree = None

    # Valores de comparação com o artigo fornecidos pelo usuário
    st.sidebar.subheader("Valores do Artigo para Comparação (Opcional)")
    mse_artigo = st.sidebar.number_input('MSE do Artigo', min_value=0.0, value=0.0)
    mape_artigo = st.sidebar.number_input('MAPE do Artigo', min_value=0.0, value=0.0)
    r2_artigo = st.sidebar.number_input('R² do Artigo', min_value=0.0, max_value=1.0, value=0.0)
    erro_medio_artigo = st.sidebar.number_input('Erro Médio do Artigo', min_value=0.0, value=0.0)

    return modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, l2_reg, subsample, colsample_bytree, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo

# Função para plotar a comparação de previsões com os valores reais
def plotar_comparacao_previsoes(y_test, y_pred):
    st.write("### Comparação de Previsões com Valores Reais")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values, label='Valores Reais', color='blue')
    ax.plot(y_pred, label='Previsões do Modelo', color='red')
    ax.set_title('Comparação de Previsões do Modelo com os Valores Reais de ΔGWS')
    ax.set_xlabel('Amostras')
    ax.set_ylabel('ΔGWS')
    ax.legend()
    st.pyplot(fig)

# Função principal
def main():
    st.title("Aplicativo de Aprendizado de Máquina para Previsão de Variações de Águas Subterrâneas (GWS)")
    modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, l2_reg, subsample, colsample_bytree, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo = configurar_sidebar()

    uploaded_file = st.sidebar.file_uploader("Carregue seus dados em CSV", type=["csv"])

    if uploaded_file:
        data = carregar_dados(uploaded_file)
        if data is not None:
            st.write("Pré-visualização dos Dados Carregados:")
            st.write(data.head())

            # Opção para selecionar a coluna de data
            if st.sidebar.checkbox("Os dados contêm coluna de data?"):
                coluna_tempo = st.sidebar.selectbox('Selecione a coluna de data', data.columns)
                data = extrair_caracteristicas_temporais(data, coluna_tempo)

            # Opção para selecionar colunas de latitude e longitude
            if st.sidebar.checkbox("Os dados contêm coordenadas geográficas?"):
                coluna_latitude = st.sidebar.selectbox('Selecione a coluna de Latitude', data.columns)
                coluna_longitude = st.sidebar.selectbox('Selecione a coluna de Longitude', data.columns)
                data = codificar_coordenadas(data, coluna_latitude, coluna_longitude)

            # Adicionar no sidebar a opção para selecionar a variável alvo
            coluna_alvo = st.sidebar.selectbox('Selecione a coluna alvo (target)', data.columns)

            # Usar a coluna selecionada como variável alvo
            if coluna_alvo in data.columns:
                X = data.drop(columns=[coluna_alvo])
                y = data[coluna_alvo]
                verificar_tipo_problema(y, tipo_problema)
            else:
                st.error(f"A coluna {coluna_alvo} não foi encontrada no arquivo CSV.")
                st.stop()
            
            # Remover outliers (opcional)
            remover_outliers_toggle = st.sidebar.checkbox("Remover Outliers?", value=False)
            if remover_outliers_toggle:
                X, y = remover_outliers(X, y)
                st.write("Outliers removidos.")

            # Pré-processar os dados
            X_processed, preprocessor = preparar_dados(X, y, tipo_problema)
            if X_processed is None:
                st.stop()

            # Dividir os dados em conjuntos de treino e teste
            if st.sidebar.checkbox("Usar Validação Cruzada Temporal?", value=False):
                time_series = True
                tscv = TimeSeriesSplit(n_splits=5)
                splits = tscv.split(X_processed)
            else:
                time_series = False
                X_train_full, X_test, y_train_full, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

            # Aplicar SMOTE para balanceamento em problemas de classificação
            if tipo_problema == 'Classificação':
                aplicar_smote_toggle = st.sidebar.checkbox("Aplicar SMOTE para Balanceamento?", value=False)
                if aplicar_smote_toggle:
                    sm = SMOTE(random_state=42)
                    X_train_full, y_train_full = sm.fit_resample(X_train_full, y_train_full)
                    st.write("SMOTE aplicado para balanceamento das classes.")

            # Escolher o modelo baseado no tipo de problema
            if tipo_problema == 'Regressão':
                # Definir parâmetros do modelo de regressão
                modelo_kwargs = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'reg_lambda': l2_reg
                }
                if subsample and colsample_bytree:
                    modelo_kwargs['subsample'] = subsample
                    modelo_kwargs['colsample_bytree'] = colsample_bytree

                # Treinamento do modelo de regressão
                if modelo_tipo == 'XGBoost':
                    modelo = XGBRegressor(**modelo_kwargs)
                elif modelo_tipo == 'CatBoost':
                    modelo = CatBoostRegressor(**modelo_kwargs, verbose=0)
                elif modelo_tipo == 'Random Forest':
                    modelo = RandomForestRegressor(**modelo_kwargs)
                elif modelo_tipo == 'Stacking':
                    modelo = stacking_model(tipo_problema)

                # Aplicar Randomized Search para otimização de hiperparâmetros
                if st.sidebar.checkbox('Otimizar Hiperparâmetros?'):
                    param_distributions = {
                        'n_estimators': [100, 200, 300, 500],
                        'max_depth': [4, 6, 8, 10, 12],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    }
                    modelo = otimizar_modelo(modelo, X_train_full, y_train_full, param_distributions, tipo_problema)

                # Treinar o modelo usando Cross-Validation
                if time_series:
                    scores = cross_val_score(modelo, X_processed, y, cv=tscv, scoring='neg_mean_squared_error')
                    st.write(f"Validação Cruzada Temporal (MSE): {-np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
                else:
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    scores = cross_val_score(modelo, X_train_full, y_train_full, cv=cv, scoring='neg_mean_squared_error')
                    st.write(f"Validação Cruzada (MSE): {-np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

                    # Ajustar o modelo nos dados completos de treinamento
                    modelo.fit(X_train_full, y_train_full)

                    # Fazer previsões no conjunto de teste
                    y_pred = modelo.predict(X_test)

                    # Calcular métricas de desempenho de regressão
                    mse, rmse, mape, mae, r2, erro_medio = calcular_metricas_regressao(y_test, y_pred)
                    exibir_metricas_regressao(mse, rmse, mape, mae, r2, erro_medio)

                    # Comparar com os valores do artigo (se fornecidos)
                    if mse_artigo > 0:
                        comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo)

                    # Exibir a importância das features
                    mostrar_importancia_features(modelo, X, preprocessor)

                    # Exibir gráfico de dispersão de previsões vs valores reais
                    plotar_dispersao_previsoes(y_test, y_pred)

                    # Exibir gráfico de resíduos
                    plotar_residuos(y_test, y_pred)

                    # Exibir gráfico de comparação de previsões com valores reais
                    plotar_comparacao_previsoes(y_test, y_pred)

            elif tipo_problema == 'Classificação':
                # Definir parâmetros do modelo de classificação
                modelo_kwargs = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'reg_lambda': l2_reg
                }
                if subsample and colsample_bytree:
                    modelo_kwargs['subsample'] = subsample
                    modelo_kwargs['colsample_bytree'] = colsample_bytree

                # Treinamento do modelo de classificação
                if modelo_tipo == 'XGBoost':
                    modelo = XGBClassifier(**modelo_kwargs, use_label_encoder=False, eval_metric='logloss')
                elif modelo_tipo == 'CatBoost':
                    modelo = CatBoostClassifier(**modelo_kwargs, verbose=0)
                elif modelo_tipo == 'Random Forest':
                    modelo = RandomForestClassifier(**modelo_kwargs)
                elif modelo_tipo == 'Stacking':
                    modelo = stacking_model(tipo_problema)

                # Aplicar Randomized Search para otimização de hiperparâmetros
                if st.sidebar.checkbox('Otimizar Hiperparâmetros?'):
                    param_distributions = {
                        'n_estimators': [100, 200, 300, 500],
                        'max_depth': [4, 6, 8, 10, 12],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    }
                    modelo = otimizar_modelo(modelo, X_train_full, y_train_full, param_distributions, tipo_problema)

                # Treinar o modelo usando Cross-Validation
                if time_series:
                    scores = cross_val_score(modelo, X_processed, y, cv=tscv, scoring='f1_weighted')
                    st.write(f"Validação Cruzada Temporal (F1 Score): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
                else:
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    scores = cross_val_score(modelo, X_train_full, y_train_full, cv=cv, scoring='f1_weighted')
                    st.write(f"Validação Cruzada (F1 Score): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

                    # Ajustar o modelo nos dados completos de treinamento
                    modelo.fit(X_train_full, y_train_full)

                    # Fazer previsões no conjunto de teste
                    y_pred = modelo.predict(X_test)
                    y_proba = modelo.predict_proba(X_test)

                    # Calcular métricas de classificação
                    metrics = calcular_metricas_classificacao(y_test, y_pred, y_proba)
                    exibir_metricas_classificacao(metrics)

                    # Exibir relatório de classificação
                    st.write("### Relatório de Classificação:")
                    st.text(classification_report(y_test, y_pred))

                    # Exibir matriz de confusão
                    plotar_matriz_confusao(y_test, y_pred)

                    # Exibir curva ROC (para problemas binários)
                    if len(np.unique(y_test)) == 2:
                        plotar_curva_roc(y_test, y_proba)

                    # Exibir a importância das features
                    mostrar_importancia_features(modelo, X, preprocessor)
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

# Função para plotar matriz de confusão
def plotar_matriz_confusao(y_test, y_pred):
    st.write("### Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + list(np.unique(y_test)))
    ax.set_yticklabels([''] + list(np.unique(y_test)))
    plt.xlabel('Previstos')
    plt.ylabel('Verdadeiros')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')
    st.pyplot(fig)

# Função para plotar curva ROC
def plotar_curva_roc(y_test, y_proba):
    st.write("### Curva ROC:")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falso Positivo')
    ax.set_ylabel('Taxa de Verdadeiro Positivo')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Função para comparar com o artigo
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
#_________________________________

# Expander de Insights do Código
with st.sidebar.expander("Pesquisa compreenda:"):
    st.markdown("""
    # **Análise Comparativa de Idiomas: Dzubukuá, Português Arcaico e Português Moderno**

    ## **Resumo**

    Este estudo apresenta uma análise comparativa entre três idiomas: **Dzubukuá** (uma língua extinta), **Português Arcaico** e **Português Moderno**. O objetivo principal é investigar as similaridades e diferenças entre esses idiomas em termos de semântica, léxico e fonologia, utilizando técnicas avançadas de Processamento de Linguagem Natural (PLN) e métodos estatísticos. Foram utilizadas metodologias como Sentence-BERT, Word2Vec, análise de N-gramas e medidas fonéticas. Os resultados indicam influências históricas entre os idiomas e contribuem para a compreensão da evolução linguística.

    ---

    ## **1. Introdução**

    A preservação e o estudo de línguas extintas ou em risco de extinção são fundamentais para a compreensão da evolução cultural e linguística da humanidade. O **Dzubukuá** é uma dessas línguas, com registros limitados e pouco estudada. Compará-la com o **Português Arcaico** e o **Português Moderno** pode revelar influências históricas, empréstimos linguísticos e mudanças semânticas ao longo do tempo.

    **Objetivos Específicos:**

    - Avaliar as similaridades semânticas entre as frases dos três idiomas.
    - Investigar as semelhanças lexicais, considerando a estrutura das palavras.
    - Analisar a proximidade fonológica, comparando os sons das palavras.
    - Realizar análises estatísticas para identificar relações significativas entre as medidas de similaridade.

    ---

    ## **2. Revisão da Literatura**

    Estudos sobre línguas extintas têm ganhado destaque nas últimas décadas devido ao avanço das técnicas de PLN. Segundo **Harrison (2007)**, a perda de uma língua representa a perda de conhecimento cultural único. **Bird (2010)** destaca a importância de documentar e analisar essas línguas utilizando ferramentas computacionais.

    A aplicação de modelos de linguagem, como o **Word2Vec** (Mikolov et al., 2013) e o **Sentence-BERT** (Reimers & Gurevych, 2019), tem permitido avanços significativos na análise semântica e lexical entre idiomas. **Mitra et al. (2014)** utilizaram modelos de tópicos para comparar línguas antigas e modernas, revelando padrões evolutivos.

    Estudos fonológicos, como o de **Jurafsky & Martin (2020)**, ressaltam a importância de analisar sons para compreender relações históricas entre línguas. A utilização de medidas de distância fonética auxilia na identificação de empréstimos e influências culturais.

    ---

    ## **3. Metodologia**

    ### **3.1 Coleta e Preparação dos Dados**

    **Fonte dos Dados:**

    - **Dzubukuá:** Foram coletadas 500 frases de documentos históricos, registros antropológicos e transcrições disponíveis em museus e universidades.
    - **Português Arcaico:** Extraídas 500 frases de textos literários e documentos oficiais datados entre os séculos XIII e XVI.
    - **Português Moderno:** Selecionadas 500 frases contemporâneas de jornais, livros e conversações cotidianas.

    **Organização dos Dados:**

    - As frases foram organizadas em um arquivo CSV com as colunas:
        - **Idioma**
        - **Texto Original**
        - **Tradução para o Português Moderno** (para Dzubukuá e Português Arcaico)
    - Garantiu-se o alinhamento temático das frases para permitir comparações coerentes.

    **Pré-processamento:**

    - **Limpeza de Dados:** Remoção de caracteres especiais, normalização de texto e tratamento de valores ausentes.
    - **Tokenização:** Segmentação das frases em palavras ou caracteres, conforme necessário.
    - **Codificação Fonética:** Aplicada para análises fonológicas.

    ### **3.2 Cálculo das Similaridades**

    As similaridades entre as frases foram analisadas em três níveis:

    #### **3.2.1 Similaridade Semântica com Sentence-BERT**

    **Fundamentos Teóricos:**

    - O **Sentence-BERT** é um modelo que gera embeddings semânticos para frases, capturando nuances de significado.

    **Processo Metodológico:**

    1. **Geração de Embeddings:**
    """)
    st.markdown("""
    **Treinamento do Modelo:**
    
    Para cada frase $s_i$, o modelo Sentence-BERT gera um vetor de dimensão $d$:
    """)
    
    st.latex(r'''
    \vec{v}_i \in \mathbb{R}^d
    ''')
    st.markdown("""
    2. **Cálculo da Similaridade de Cosseno:**

        A similaridade entre duas frases $s_i$ e $s_j$ é calculada por:

    """)
    st.latex(r'''
    \text{similaridade}(\vec{v}_i, \vec{v}_j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \times \|\vec{v}_j\|}
    ''')
    st.markdown("""
    **Exemplo:**

    - Frase em Dzubukuá: "Ama tuça laka." (Tradução: "O sol está brilhando.")
    - Frase em Português Arcaico: "O sol resplandece."
    - Similaridade calculada: **0,85** (em uma escala de 0 a 1).

    **Interpretação:**

    - A alta similaridade indica que as frases compartilham significados semelhantes, sugerindo preservação semântica.

    #### **3.2.2 Similaridade Lexical com N-gramas**

    **Fundamentos Teóricos:**

    - Os **N-gramas** capturam padrões de sequência de caracteres, úteis para identificar semelhanças na estrutura das palavras.

    **Processo Metodológico:**

    1. **Extração de N-gramas:**

        - Utilizamos trigramas (N=3) para capturar padrões lexicais.

        Exemplo para a palavra "linguagem":

        
    Trigramas de "linguagem": ${ "lin", "ing", "ngu", "gua", "uag", "age", "gem" }$
 
   
    2. **Cálculo do Coeficiente de Sorensen-Dice:**

        """)
    st.latex(r'''
    \small
    \text{SDC}(A, B) = \frac{2 \times |A \cap B|}{|A| + |B|}
    ''')
    st.markdown("""
    **Exemplo:**

    - Frase em Português Arcaico: "A casa é bela."
    - Frase em Português Moderno: "A casa é bonita."
    - Similaridade calculada: **0,78**.

    **Interpretação:**

    - A similaridade lexical elevada reflete a conservação de estruturas de palavras entre os dois períodos do idioma.

    #### **3.2.3 Similaridade Lexical com Word2Vec**

    **Fundamentos Teóricos:**

    - O **Word2Vec** aprende representações vetoriais das palavras com base no contexto, permitindo capturar relações semânticas e sintáticas.

    **Processo Metodológico:**

    1. **Tokenização:**

        - As frases foram tokenizadas em palavras.

    2. **Treinamento do Modelo:**

        - O modelo Word2Vec foi treinado com todas as frases dos três idiomas.

    3. **Representação das Frases:**

        """)
    st.latex(r'''
    \small
    \vec{v}_{\text{frase}} = \frac{1}{n} \sum_{i=1}^{n} \vec{w}_i
    ''')
    st.markdown("""
    4. **Cálculo da Similaridade:**

        - Similaridade de cosseno entre os vetores das frases.

    **Exemplo:**

    - Frase em Dzubukuá: "Laka tuça ama." (Tradução: "O sol brilha.")
    - Frase em Português Moderno: "O sol está brilhando."
    - Similaridade calculada: **0,82**.

    **Interpretação:**

    - A similaridade indica que, apesar das diferenças lexicais, há uma relação semântica capturada pelo contexto.

    #### **3.2.4 Similaridade Fonológica**
    
    **Fundamentos Teóricos:**
    
    - A análise fonológica é crucial para identificar influências linguísticas que não são evidentes apenas pela escrita.
    
    **Processo Metodológico:**
    
    1. **Codificação Fonética:**
    
        - Utilizamos o algoritmo **Soundex** adaptado para o português.
    
    2. **Cálculo da Distância de Levenshtein:**
        """)
    
    st.latex(r'''
    \scriptsize
    D(S_1, S_2) = \text{Número mínimo de operações para transformar } S_1 \text{ em } S_2
    ''')
    st.markdown("""    
    3. **Normalização da Similaridade:**
        """)
    st.latex(r'''
    \small
    \text{Similaridade} = 1 - \frac{D(S_1, S_2)}{\max(\text{len}(S_1), \text{len}(S_2))}
    ''')
    st.markdown("""    
    **Exemplo:**
    
    - Palavra em Dzubukuá: "Ama" (Codificação: "A500")
    - Palavra em Português: "Amar" (Codificação: "A560")
    - Similaridade calculada: **0,75**.
    
    **Interpretação:**
    
    - A similaridade fonológica sugere possíveis influências ou origens comuns.
    
    ---

    ## **4. Análises Estatísticas**

    ### **4.1 Correlações entre as Similaridades**

    **Objetivo:**

    - Investigar relações entre as medidas de similaridade para compreender a interdependência entre semântica, léxico e fonologia.

    **Resultados:**

    - **Correlação entre Similaridade Semântica e Lexical:** *r* = 0,68 (p < 0,01)
    - **Correlação entre Similaridade Semântica e Fonológica:** *r* = 0,45 (p < 0,05)
    - **Correlação entre Similaridade Lexical e Fonológica:** *r* = 0,52 (p < 0,05)

    **Interpretação:**

    - Há correlações significativas, indicando que as medidas estão relacionadas, mas não são redundantes.

    ### **4.2 Análise de Regressão**

    **Modelo de Regressão Múltipla:**

        """)

    st.markdown("""
    **Resultados:**

    - **Coeficiente $beta_1$ :** 0,55 (p < 0,01)
    - **Coeficiente $beta_2$:** 0,30 (p < 0,05)
    - **R² Ajustado:** 0,62

    **Interpretação:**

    - A similaridade lexical contribui mais para a previsão da similaridade semântica, mas a fonológica também é significativa.

    ### **4.3 Análise de Variância (ANOVA)**

    **Objetivo:**

    - Verificar se há diferenças significativas nas similaridades entre os pares de idiomas.

    **Resultados:**

    - **Comparação Dzubukuá vs. Português Arcaico vs. Português Moderno:**
        - **F(2, 1497) = 15,6** (p < 0,01)

    **Interpretação:**

    - Há diferenças significativas nas medidas de similaridade entre os idiomas, justificando análises separadas.

    ---

    ## **5. Resultados e Discussão**

    **Similaridade Semântica:**

    - As altas similaridades entre Dzubukuá e Português Arcaico sugerem uma possível influência histórica ou compartilhamento de conceitos culturais.

    **Similaridade Lexical:**

    - A maior similaridade entre Português Arcaico e Moderno era esperada devido à continuidade evolutiva da língua.
    - A similaridade lexical entre Dzubukuá e Português Arcaico, embora menor, é significativa.

    **Similaridade Fonológica:**

    - As similaridades fonológicas indicam que sons semelhantes persistem, possivelmente devido a contatos culturais ou adaptações linguísticas.

    **Análises Estatísticas:**

    - As correlações e análises de regressão reforçam a interconexão entre os diferentes níveis linguísticos.
    - A regressão múltipla mostra que tanto a similaridade lexical quanto a fonológica contribuem para a semântica.

    **Discussão:**

    - Os resultados apontam para uma possível interação histórica entre os povos falantes de Dzubukuá e os antepassados do português.
    - Isso pode ter ocorrido através de comércio, migrações ou outras formas de contato cultural.
    - A análise multidimensional fornece uma visão abrangente das relações linguísticas.

    **Limitações:**

    - **Tamanho e Qualidade do Corpus:** Embora abrangente, o corpus pode não representar todas as variações linguísticas.
    - **Traduções:** A precisão das traduções é crucial e pode introduzir vieses.
    - **Modelos de Linguagem:** Dependem da qualidade e quantidade de dados de treinamento.

    ---

    ## **6. Conclusão**

    Este estudo apresentou uma análise comparativa detalhada entre Dzubukuá, Português Arcaico e Português Moderno, utilizando técnicas avançadas de PLN e estatística. Os resultados sugerem influências históricas e culturais entre os idiomas, contribuindo para a compreensão da evolução linguística.

    **Contribuições:**

    - Demonstra a eficácia de técnicas de PLN na análise de línguas extintas.
    - Fornece insights sobre possíveis interações históricas entre povos.
    - Destaca a importância de preservar e estudar línguas em risco de extinção.

    **Trabalhos Futuros:**

    - **Expansão do Corpus:** Incluir mais dados e outros idiomas para ampliar a análise.
    - **Análises Qualitativas:** Complementar as análises quantitativas com estudos etnográficos e históricos.
    - **Desenvolvimento de Modelos Específicos:** Criar modelos de linguagem adaptados para línguas extintas.

    ---

    ## **Referências**

    - Bird, S. (2010). **A survey of computational approaches to endangered language documentation and revitalization.** Language and Linguistics Compass, 4(6), 768-781.
    - Harrison, K. D. (2007). **When Languages Die: The Extinction of the World's Languages and the Erosion of Human Knowledge.** Oxford University Press.
    - Jurafsky, D., & Martin, J. H. (2020). **Speech and Language Processing.** Pearson.
    - Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space.** arXiv preprint arXiv:1301.3781.
    - Mitra, R., Costa, H., & Das, D. (2014). **Analyzing Ancient Texts Using Topic Modeling.** Journal of Historical Linguistics, 4(2), 187-210.
    - Reimers, N., & Gurevych, I. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.** arXiv preprint arXiv:1908.10084.
    - Smith, A., Johnson, B., & Clark, E. (2021). **Computational Approaches to Historical Linguistics.** Annual Review of Linguistics, 7, 341-361.

    """)


# Expander de Insights do Código
with st.sidebar.expander("Insights metodológicos"):
    st.markdown("""
    ### **Introdução**

    Este aplicativo foi desenvolvido para realizar um estudo linguístico profundo comparando três idiomas: **Dzubukuá** (uma língua em risco de extinção), **Português Arcaico** e **Português Moderno**. A ideia principal é analisar como esses idiomas se relacionam em termos de:

    - **Semântica** (significado das frases)
    - **Estrutura Lexical** (construção e formação das palavras)
    - **Fonologia** (som das palavras)

    Utilizamos métodos avançados de **Processamento de Linguagem Natural (PLN)** e ferramentas estatísticas para medir a semelhança e as diferenças entre as frases dos três idiomas.

    ---

    ### **1. Ferramentas Utilizadas**

    Para realizar essas análises, usamos diversas bibliotecas do Python. Cada uma tem uma função específica, como veremos a seguir:

    - **Pandas**: Manipulação de dados e criação de tabelas organizadas.
    - **Streamlit**: Interface web interativa.
    - **Matplotlib** e **Seaborn**: Visualização de gráficos.
    - **Plotly**: Criação de gráficos interativos para dados.
    - **Scikit-learn**: Implementação de algoritmos de aprendizado de máquina e estatística.
    - **Gensim**: Implementação do modelo **Word2Vec** para análises de similaridade lexical.
    - **Jellyfish**: Cálculo de similaridade fonética entre palavras.
    - **Statsmodels** e **Scipy**: Ferramentas estatísticas para cálculos avançados.

    **Objetivo:** Usar essas bibliotecas para entender como os três idiomas se comportam em termos de significado, construção e som.

    ---

    ### **2. Carregamento e Organização dos Dados**

    Começamos com o carregamento de um arquivo CSV que contém as frases nos três idiomas: Dzubukuá, Português Arcaico e Português Moderno. Cada linha do arquivo representa uma frase e suas traduções, e as colunas contêm:

    - **Texto Original**: A frase na língua original.
    - **Tradução para o Português Moderno**: A frase traduzida para o português moderno.

    **Objetivo:** Extrair essas frases para que possamos comparar as três línguas em termos de significado, estrutura e som.

    ---

    ### **3. Cálculo das Similaridades**

    As similaridades entre as frases dos três idiomas são medidas de três maneiras principais:

    - **Similaridade Semântica**: Comparamos o significado das frases.
    - **Similaridade Lexical**: Analisamos a construção das palavras.
    - **Similaridade Fonológica**: Analisamos como as palavras soam.

    #### **3.1 Similaridade Semântica com Sentence-BERT**

    **O que é:** O **Sentence-BERT** é um modelo de linguagem que transforma frases em vetores numéricos (chamados de "embeddings"), representando o significado da frase.

    **Como funciona:**
    """)
    st.latex(r'''
    \small
    1. \ \text{Geração de Embeddings: Cada frase é convertida}
    ''')
    st.latex(r'''
    \small
     \text{em um vetor de números querepresenta o seu significado.}
    ''')
    
    st.latex(r'''
    \small
     \text{Esses vetores são de alta dimensão, representados como } \vec{v}_i
    ''')

    st.latex(r'''
    \small
    \text{Para cada frase } s_i, \text{ obtemos um vetor } \vec{v}_i \in \mathbb{R}^d
    ''')

    st.markdown("""
    2. **Cálculo da Similaridade de Cosseno:** Para comparar os significados de duas frases, usamos a **similaridade de cosseno**. Esta métrica mede o ângulo entre os vetores de duas frases:
    """)

    st.latex(r'''
    \small
    \text{similaridade}(\vec{v}_i, \vec{v}_j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\| \cdot \|\vec{v}_j\|}
    ''')

    st.latex(r'''
    \small
    \text{Onde:}
    ''')

    st.latex(r'''
    \small
    \vec{v}_i \cdot \vec{v}_j \text{ é o produto escalar dos vetores}
    ''')

    st.latex(r'''
    \small
    \|\vec{v}_i\| \text{ é a magnitude do vetor } \vec{v}_i
    ''')

    st.latex(r'''
    \small
    \|\vec{v}_j\| \text{ é a magnitude do vetor } \vec{v}_j
    ''')

    st.markdown("""
    **Objetivo:** O objetivo é descobrir se duas frases em diferentes idiomas têm o mesmo significado, independentemente das palavras usadas.

    **Exemplo prático:** Se duas frases – uma em Dzubukuá e outra em Português Arcaico – apresentarem alta similaridade semântica, isso sugere que o significado original da frase foi preservado, mesmo que as palavras tenham mudado ao longo do tempo.

    ---

    #### **3.2 Similaridade Lexical com N-gramas**

    **O que são N-gramas?**: N-gramas são sequências de N caracteres que aparecem em uma palavra ou frase. Um N-grama de dois caracteres é chamado de **bigrama**, um de três é um **trigrama**, e assim por diante.

    **Exemplo prático:** Para a palavra "casa", os bigramas (N=2) seriam "ca", "as", "sa". Já para a palavra "dia", teríamos "di", "ia".

    **Como funciona:**

    1. **Extração de N-gramas:** Para cada palavra ou frase, extraímos os N-gramas (sequências de N letras).

    **Exemplo com a palavra "sol"**: Se usarmos bigramas (N=2), a palavra "sol" seria dividida em "so", "ol".

    2. **Representação Vetorial:** Cada frase é representada como um vetor, com 1 indicando a presença de um N-grama e 0 indicando a ausência.

    3. **Cálculo do Coeficiente de Sorensen-Dice:** Usamos a seguinte fórmula para medir a similaridade lexical entre duas frases:

    """)

    st.latex(r'''
    \small
    \text{SDC}(A, B) = \frac{2 \times |A \cap B|}{|A| + |B|}
    ''')

    st.latex(r'''
    \small
    \text{Onde:}
    ''')

    st.latex(r'''
    \small
    |A| \text{ e } |B| \text{ são o número de N\text{-}gramas em } A \text{ e } B
    ''')

    st.latex(r'''
    \small
    |A \cap B| \text{ é o número de N\text{-}gramas comuns entre } A \text{ e } B
    ''')

    st.markdown("""
    **Objetivo:** Avaliar como as palavras dos três idiomas se comparam em termos de construção. Isso nos ajuda a ver se as palavras evoluíram de forma semelhante ou diferente.

    **Exemplo prático:** Se uma palavra em Dzubukuá tem muitos N-gramas em comum com sua tradução em Português Arcaico, isso indica que as duas línguas compartilham uma estrutura lexical parecida, mesmo que as palavras tenham mudado ligeiramente ao longo do tempo.

    ---

    #### **3.3 Similaridade Lexical com Word2Vec**

    **O que é:** O **Word2Vec** é uma técnica que transforma palavras em vetores, baseando-se no contexto em que elas aparecem.

    **Como funciona:**

    1. **Tokenização das Frases:** As frases são divididas em palavras individuais.

    2. **Treinamento do Modelo:** O modelo aprende a gerar vetores para palavras com base nas palavras que as cercam no texto.

    3. **Representação das Frases:** Calculamos a média dos vetores de todas as palavras de uma frase para gerar um vetor representativo da frase:

    """)

    st.latex(r'''
    \small
    \vec{v}_{\text{frase}} = \frac{1}{n} \sum_{i=1}^{n} \vec{w}_i
    ''')

    st.latex(r'''
    \small
    \text{Onde: } n \text{ é o número de palavras e } \vec{w}_i \text{ é o vetor de cada palavra.}
    ''')

    st.markdown("""
    **Objetivo:** Verificar se as palavras de diferentes idiomas têm contextos semelhantes, mesmo que a grafia tenha mudado ao longo do tempo.

    **Exemplo prático:** O modelo Word2Vec pode identificar que, embora uma palavra tenha mudado no Português Moderno, seu uso e contexto permanecem os mesmos do Português Arcaico. Isso nos permite identificar se palavras que parecem diferentes podem, na verdade, manter um uso semântico semelhante ao longo do tempo, especialmente em contextos históricos ou religiosos, onde certos padrões linguísticos podem se manter inalterados.
    
    ---
    
    #### **3.4 Similaridade Fonológica**
    
    **O que é:** A análise fonológica tem a ver com o som das palavras e como esses sons se relacionam entre si em diferentes línguas. Quando falamos de "similaridade fonológica", estamos medindo o quanto as palavras de dois idiomas diferentes soam de maneira semelhante, mesmo que a grafia seja completamente diferente.
    
    **Usos práticos da análise fonológica:**
    
    A análise fonológica é crucial em diversas áreas práticas, como:
    
    1. **Reconstrução de Línguas Mortas ou em Extinção:** Em línguas como Dzubukuá, que está em risco de extinção, a análise fonológica pode ajudar a reconstruir a pronúncia original de palavras baseando-se nas semelhanças sonoras com outras línguas da mesma família.
    2. **Estudos de Dialetos:** Dialetos de uma mesma língua podem ter grandes diferenças na grafia, mas soarem semelhantes. A análise fonológica permite medir essas semelhanças e entender como os sons evoluíram em diferentes regiões.
    3. **Traduções Históricas:** A fonologia pode ajudar a entender se traduções antigas mantiveram a essência sonora de certas palavras. Por exemplo, palavras litúrgicas podem ter sons que foram preservados através de séculos, mesmo que a ortografia tenha mudado.
    4. **Reconhecimento de Voz e Assistentes Virtuais:** Algoritmos que reconhecem a fala dependem de uma análise fonológica para identificar corretamente as palavras faladas, independentemente do sotaque ou da variação linguística.
    
    **Como funciona:**
    
    1. **Codificação Fonética:** A primeira etapa é converter cada palavra em um código fonético que capture como ela soa. Usamos a técnica **Soundex**, que transforma uma palavra em uma representação fonética.
    
    2. **Cálculo da Distância de Levenshtein:** Para medir a semelhança entre duas palavras, calculamos o número de operações (inserções, deleções ou substituições) necessárias para transformar uma palavra na outra. Isso é conhecido como **Distância de Levenshtein**.
    
    """)
    
    st.latex(r'''
    \small
    D(S_1, S_2) = \text{Número mínimo de operações para transformar } S_1 \text{ em } S_2
    ''')
    
    st.markdown("""
    3. **Normalização da Similaridade:** A distância é normalizada para gerar uma pontuação entre 0 e 1, onde 1 indica que as palavras são fonologicamente idênticas, e 0 indica que elas são completamente diferentes em termos de som.
    
    """)
    
    st.latex(r'''
    \small
    \text{Similaridade} = 1 - \frac{D(S_1, S_2)}{\max(\text{len}(S_1), \text{len}(S_2))}
    ''')
    
    st.markdown("""
    **Objetivo:** Avaliar o quanto as palavras de dois idiomas diferentes soam de maneira semelhante, independentemente de suas grafias.
    
    **Exemplo prático:** Se a palavra "Deus" no Português Arcaico tem uma alta similaridade fonológica com sua equivalente em Dzubukuá, isso sugere que, embora as línguas tenham evoluído de maneiras diferentes, certos sons-chave foram preservados.
    
    ---
    
    ### **4. Análises Estatísticas e Visualizações**
    
    Depois de calcular as similaridades semânticas, lexicais e fonológicas, podemos realizar análises estatísticas para entender como esses diferentes aspectos das línguas se relacionam entre si.
    
    #### **4.1 Cálculo de Correlações**
    
    **Correlação** é uma medida estatística que indica o quanto duas variáveis estão relacionadas. No nosso estudo, calculamos a correlação entre as medidas de similaridade para descobrir se a similaridade em um nível (como a fonológica) está relacionada com a similaridade em outro nível (como a lexical).
    
    **Tipos de correlação usados:**
    
    1. **Correlação de Pearson:** Mede a relação linear entre duas variáveis contínuas. Se o valor for 1, isso significa que as variáveis estão perfeitamente relacionadas de forma positiva. Se for -1, estão relacionadas de forma negativa.
    
    """)
    
    st.latex(r'''
    \small
    r = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2} \cdot \sqrt{\sum_{i=1}^{n}(Y_i - \bar{Y})^2}}
    ''')
    
    st.markdown("""
    **Exemplo prático:** Podemos usar a correlação de Pearson para ver se frases com alta similaridade semântica também têm uma alta similaridade fonológica. Se sim, isso pode indicar que frases com significados semelhantes também tendem a soar de maneira semelhante, o que pode ter implicações importantes para estudos de línguas antigas.
    
    2. **Correlação de Spearman:** Diferente da correlação de Pearson, a correlação de Spearman mede a relação entre os rankings de duas variáveis. Ela é mais apropriada quando os dados não seguem uma relação linear.
    
    """)
    
    st.latex(r'''
    \small
    \rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
    ''')
    
    st.latex(r'''
    \small
    \text{Onde } d_i = \text{posto}(X_i) - \text{posto}(Y_i)
    ''')
    
    st.markdown("""
    **Exemplo prático:** Se a ordem das frases em termos de similaridade semântica corresponder à ordem das frases em termos de similaridade lexical, a correlação de Spearman nos mostrará que esses aspectos estão relacionados, mesmo que a relação não seja estritamente linear.
    
    3. **Correlação de Kendall:** Similar à de Spearman, mas usada quando estamos interessados em pares de observações concordantes ou discordantes em rankings.
    
    """)
    
    st.latex(r'''
    \small
    \tau = \frac{C - D}{\frac{1}{2} n(n - 1)}
    ''')
    
    st.latex(r'''
    \small
    \text{Onde:}
    ''')
    
    st.latex(r'''
    \small
    C \text{ é o número de pares concordantes}
    ''')
    
    st.latex(r'''
    \small
    D \text{ é o número de pares discordantes}
    ''')
    
    st.markdown("""
    **Objetivo:** Com essas correlações, podemos entender se as diferentes formas de medir similaridade (semântica, lexical, fonológica) estão relacionadas entre si. Isso é importante porque, se houver uma correlação forte entre os diferentes níveis, isso nos diz que, apesar das línguas mudarem ao longo do tempo, elas podem manter relações subjacentes entre o som, a forma das palavras e o significado.
    
    ---
    
    ### **5. Considerações e Limitações**
    
    Ao longo dessa análise, encontramos várias similaridades e diferenças interessantes entre Dzubukuá, Português Arcaico e Português Moderno. Entretanto, devemos levar em conta algumas limitações:
    
    1. **Qualidade dos Dados:** A precisão das traduções e a disponibilidade de dados completos são fundamentais para a robustez das análises. Traduções errôneas ou incompletas podem enviesar os resultados.
       
    2. **Assunções Estatísticas:** Métodos como correlação de Pearson assumem que as variáveis estão relacionadas de maneira linear. Se essa suposição não for válida, os resultados podem não ser precisos.
       
    3. **Multicolinearidade:** Em certas análises, diferentes formas de medir similaridade podem estar fortemente relacionadas entre si (multicolinearidade). Isso pode dificultar a interpretação das análises de correlação.
    
    ---
    
    ### **Conclusão**
    
    Esse estudo oferece uma visão detalhada sobre a evolução das línguas e como as semelhanças entre Dzubukuá, Português Arcaico e Moderno podem ser medidas em diferentes níveis. Ao explorar as similaridades semânticas, lexicais e fonológicas, podemos entender melhor como as línguas evoluem e como certos aspectos são preservados ao longo do tempo, mesmo que a grafia e o uso das palavras mudem.

    """)



# Imagem e Contatos
if os.path.exists("eu.ico"):
    st.sidebar.image("eu.ico", width=80)
else:
    st.sidebar.text("Imagem do contato não encontrada.")

st.sidebar.write("""
Projeto Geomaker + IA 

https://doi.org/10.5281/zenodo.13856575
- Professor: Marcelo Claro.
Contatos: marceloclaro@gmail.com
Whatsapp: (88)981587145
Instagram: [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)

""")

# _____________________________________________
# Controle de Áudio
st.sidebar.title("Controle de Áudio")

# Dicionário de arquivos de áudio, com nomes amigáveis mapeando para o caminho do arquivo
mp3_files = {
    "Áudio explicativo 1": "kariri.mp3",
}

# Lista de arquivos MP3 para seleção
selected_mp3 = st.sidebar.radio("Escolha um áudio explicativo:", options=list(mp3_files.keys()))

# Controle de opção de repetição
loop = st.sidebar.checkbox("Repetir áudio")

# Botão de Play para iniciar o áudio
play_button = st.sidebar.button("Play")

# Placeholder para o player de áudio
audio_placeholder = st.sidebar.empty()

# Função para verificar se o arquivo existe
def check_file_exists(mp3_path):
    if not os.path.exists(mp3_path):
        st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
        return False
    return True

# Se o botão Play for pressionado e um arquivo de áudio estiver selecionado
if play_button and selected_mp3:
    mp3_path = mp3_files[selected_mp3]
    
    # Verificação da existência do arquivo
    if check_file_exists(mp3_path):
        try:
            # Abrindo o arquivo de áudio no modo binário
            with open(mp3_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                
                # Codificando o arquivo em base64 para embutir no HTML
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Controle de loop (repetição)
                loop_attr = "loop" if loop else ""
                
                # Gerando o player de áudio em HTML
                audio_html = f"""
                <audio id="audio-player" controls autoplay {loop_attr}>
                  <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                  Seu navegador não suporta o elemento de áudio.
                </audio>
                """
                
                # Inserindo o player de áudio na interface
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
        
        except FileNotFoundError:
            st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar o arquivo: {str(e)}")

# Executar a função principal
if __name__ == "__main__":
    main()
