import streamlit as st
import os
import logging
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    cross_val_score, KFold, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, mean_squared_error,
    mean_absolute_percentage_error, r2_score, confusion_matrix, roc_auc_score,
    classification_report, roc_curve, auc, mean_absolute_error
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE
import joblib  # Para exportar modelos

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definições de funções auxiliares

def carregar_dados(file):
    try:
        data = pd.read_csv(file)
        st.write("### Informações dos Dados:")
        st.write(data.describe())
        
        # Verificar e tratar valores nulos
        if data.isnull().sum().sum() > 0:
            st.warning("Os dados contêm valores nulos. Eles serão preenchidos com a média ou moda, conforme apropriado.")
            num_cols = data.select_dtypes(include=['float64', 'int64']).columns
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            data[num_cols] = data[num_cols].fillna(data[num_cols].mean())
            data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
        
        return data
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        logging.exception("Erro ao carregar os dados")
        return None

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

def codificar_coordenadas(dataframe, coluna_latitude, coluna_longitude):
    try:
        # Verificar se as colunas selecionadas são numéricas
        if not pd.api.types.is_numeric_dtype(dataframe[coluna_latitude]):
            raise TypeError(f"A coluna '{coluna_latitude}' não é numérica.")
        if not pd.api.types.is_numeric_dtype(dataframe[coluna_longitude]):
            raise TypeError(f"A coluna '{coluna_longitude}' não é numérica.")
        
        dataframe['latitude_sin'] = np.sin(np.radians(dataframe[coluna_latitude]))
        dataframe['latitude_cos'] = np.cos(np.radians(dataframe[coluna_latitude]))
        dataframe['longitude_sin'] = np.sin(np.radians(dataframe[coluna_longitude]))
        dataframe['longitude_cos'] = np.cos(np.radians(dataframe[coluna_longitude]))
        return dataframe
    except Exception as e:
        st.error(f"Erro ao codificar coordenadas geográficas: {e}")
        logging.exception("Erro ao codificar coordenadas geográficas")
        st.stop()  # Parar a execução do app

def remover_outliers(X, y, limiar=3):
    try:
        # Selecionar apenas colunas numéricas
        colunas_numericas = X.select_dtypes(include=['float64', 'int64']).columns
        if colunas_numericas.empty:
            st.warning("Não há colunas numéricas para remover outliers.")
            return X, y
        
        z_scores = np.abs((X[colunas_numericas] - X[colunas_numericas].mean()) / X[colunas_numericas].std())
        filtro = (z_scores < limiar).all(axis=1)
        return X[filtro], y[filtro]
    except Exception as e:
        st.error(f"Erro ao remover outliers: {e}")
        logging.exception("Erro ao remover outliers")
        st.stop()

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
        return X_processed, preprocessor, num_cols, cat_cols
    except Exception as e:
        st.error(f"Erro no pré-processamento dos dados: {e}")
        logging.exception("Erro no pré-processamento dos dados")
        st.stop()

def calcular_metricas_regressao(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    erro_medio = np.mean(np.abs(y_test - y_pred))
    return mse, rmse, mape, mae, r2, erro_medio

def exibir_metricas_regressao(mse, rmse, mape, mae, r2, erro_medio):
    st.write(f"**Erro Médio Quadrado (MSE):** {mse:.4f}")
    st.write(f"**Raiz do Erro Médio Quadrado (RMSE):** {rmse:.4f}")
    st.write(f"**Erro Absoluto Médio (MAE):** {mae:.4f}")
    st.write(f"**Erro Percentual Absoluto Médio (MAPE):** {mape:.4f}")
    st.write(f"**Coeficiente de Determinação (R²):** {r2:.4f}")
    st.write(f"**Erro Médio Absoluto:** {erro_medio:.4f}")

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
                        feature_names.extend(cols)
                else:
                    # Outros transformadores, se houver
                    feature_names.extend(cols)

            # Verificar se o número de nomes de features corresponde ao número de importâncias
            if len(feature_names) != len(importancias):
                st.warning("O número de features não corresponde ao número de importâncias. Verifique o pré-processamento.")
                feature_names = [f'Feature {i}' for i in range(len(importancias))]

            importancia_df = pd.DataFrame({'Features': feature_names, 'Importância': importancias})
            importancia_df = importancia_df.sort_values(by='Importância', ascending=False)
            
            st.write("### Importância das Variáveis (Features):")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importância', y='Features', data=importancia_df, ax=ax, palette='viridis')
            plt.title("Importância das Features")
            plt.xlabel("Importância")
            plt.ylabel("Features")
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.warning("Não foi possível exibir a importância das features.")
        logging.exception("Erro ao exibir a importância das features")

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
        st.write("### Melhores Parâmetros Encontrados:")
        st.write(random_search.best_params_)
        return random_search.best_estimator_
    except Exception as e:
        st.error(f"Erro na otimização do modelo: {e}")
        logging.exception("Erro na otimização do modelo")
        return modelo

def stacking_model(tipo_problema):
    try:
        if tipo_problema == 'Classificação':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)),
                ('cat', CatBoostClassifier(n_estimators=100, verbose=0, random_state=42))
            ]
            final_estimator = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
            modelo = StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=5)
        else:
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
                ('cat', CatBoostRegressor(n_estimators=100, verbose=0, random_state=42))
            ]
            final_estimator = XGBRegressor(n_estimators=100, random_state=42)
            modelo = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
        return modelo
    except Exception as e:
        st.error(f"Erro ao criar o modelo de empilhamento: {e}")
        logging.exception("Erro ao criar o modelo de empilhamento")
        return None

def verificar_tipo_problema(y, tipo_problema):
    if tipo_problema == 'Classificação' and not pd.api.types.is_integer_dtype(y):
        st.error("O alvo é contínuo, mas o tipo de problema selecionado é 'Classificação'. Por favor, ajuste o tipo de problema para 'Regressão' ou converta o alvo em categorias discretas.")
        st.stop()
    elif tipo_problema == 'Regressão' and not pd.api.types.is_numeric_dtype(y):
        st.error("O alvo é categórico, mas o tipo de problema selecionado é 'Regressão'. Por favor, ajuste o tipo de problema para 'Classificação' ou converta o alvo em valores contínuos.")
        st.stop()

def configurar_sidebar():
    st.sidebar.title("Configurações dos Modelos")
    modelo_tipo = st.sidebar.selectbox('Escolha o Modelo', ['XGBoost', 'Random Forest', 'CatBoost', 'Stacking'])
    tipo_problema = st.sidebar.selectbox('Escolha o Tipo de Problema', ['Classificação', 'Regressão'])
    
    n_estimators = st.sidebar.slider('Número de Árvores (n_estimators)', 100, 1000, 300, step=50)
    learning_rate = st.sidebar.slider('Taxa de Aprendizado (learning_rate)', 0.01, 0.3, 0.1, step=0.01)
    max_depth = st.sidebar.slider('Profundidade Máxima (max_depth)', 3, 20, 6)
    l2_reg = st.sidebar.slider('Regularização L2 (Weight Decay)', 0.0, 1.0, 0.1, step=0.1)
    
    # Inicializar todas as variáveis como None
    gamma = min_child_weight = subsample = colsample_bytree = reg_alpha = reg_lambda = None
    min_samples_split = min_samples_leaf = max_features = depth = l2_leaf_reg = border_count = None
    
    # Adição de mais hiperparâmetros específicos
    if modelo_tipo == 'XGBoost':
        gamma = st.sidebar.slider('Gamma', 0.0, 5.0, 0.0, step=0.1)
        min_child_weight = st.sidebar.slider('Min Child Weight', 1, 10, 1)
        subsample = st.sidebar.slider('Subsample (Taxa de Amostragem)', 0.5, 1.0, 0.8, step=0.05)
        colsample_bytree = st.sidebar.slider('ColSample ByTree (Taxa de Colunas por Árvore)', 0.5, 1.0, 0.8, step=0.05)
        reg_alpha = st.sidebar.slider('Regularização Alpha', 0.0, 1.0, 0.0, step=0.1)
        reg_lambda = st.sidebar.slider('Regularização Lambda', 0.0, 1.0, 1.0, step=0.1)
    elif modelo_tipo == 'Random Forest':
        min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
        min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, 20, 1)
        max_features = st.sidebar.selectbox('Max Features', ['auto', 'sqrt', 'log2'])
    elif modelo_tipo == 'CatBoost':
        depth = st.sidebar.slider('Depth', 3, 10, 6)
        l2_leaf_reg = st.sidebar.slider('L2 Leaf Reg', 1, 10, 3)
        border_count = st.sidebar.slider('Border Count', 32, 255, 32)
    else:
        # Parâmetros para Stacking ou outros modelos
        pass  # As variáveis já estão inicializadas como None
    
    # Valores de comparação com o artigo fornecidos pelo usuário
    st.sidebar.subheader("Valores do Artigo para Comparação (Opcional)")
    mse_artigo = st.sidebar.number_input('MSE do Artigo', min_value=0.0, value=0.0, format="%.4f")
    mape_artigo = st.sidebar.number_input('MAPE do Artigo', min_value=0.0, value=0.0, format="%.4f")
    r2_artigo = st.sidebar.number_input('R² do Artigo', min_value=0.0, max_value=1.0, value=0.0, format="%.4f")
    erro_medio_artigo = st.sidebar.number_input('Erro Médio do Artigo', min_value=0.0, value=0.0, format="%.4f")

    return (modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, l2_reg, 
            subsample, colsample_bytree, gamma, min_child_weight, reg_alpha, reg_lambda,
            min_samples_split, min_samples_leaf, max_features, depth, l2_leaf_reg, border_count,
            mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo)

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

def plotar_dispersao_previsoes(y_test, y_pred):
    st.write("### Dispersão: Previsões vs Valores Reais")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, edgecolor='k')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel('Valores Reais')
    ax.set_ylabel('Previsões')
    plt.title('Previsões vs Valores Reais')
    st.pyplot(fig)

def plotar_residuos(y_test, y_pred):
    st.write("### Resíduos: Valores Reais vs Resíduos")
    residuos = y_test - y_pred
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuos, ax=ax, edgecolor='k')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Previsões')
    ax.set_ylabel('Resíduos')
    plt.title('Resíduos vs Previsões')
    st.pyplot(fig)

def plotar_matriz_confusao(y_test, y_pred):
    st.write("### Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previstos')
    plt.ylabel('Verdadeiros')
    st.pyplot(fig)

def plotar_curva_roc(y_test, y_proba):
    st.write("### Curva ROC:")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falso Positivo')
    ax.set_ylabel('Taxa de Verdadeiro Positivo')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

def plotar_curvas_aprendizado(modelo, X, y, tipo_problema):
    from sklearn.model_selection import learning_curve
    st.write("### Curvas de Aprendizado")
    fig, ax = plt.subplots(figsize=(10, 6))
    scoring = 'f1_weighted' if tipo_problema == 'Classificação' else 'neg_mean_squared_error'
    train_sizes, train_scores, test_scores = learning_curve(
        modelo, X, y, cv=5, scoring=scoring,
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Score de Treino')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Score de Validação')
    ax.set_xlabel('Tamanho do Conjunto de Treino')
    ax.set_ylabel('Score')
    ax.set_title('Curvas de Aprendizado')
    ax.legend(loc='best')
    st.pyplot(fig)

def exportar_modelo(modelo, preprocessor):
    try:
        # Criar um pipeline completo para exportação
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', modelo)
        ])
        # Salvar o pipeline usando joblib
        joblib.dump(pipeline, 'modelo_trained.pkl')
        st.success("Modelo treinado exportado com sucesso!")
        with open('modelo_trained.pkl', 'rb') as f:
            st.download_button('Download Modelo Treinado', f, file_name='modelo_trained.pkl')
    except Exception as e:
        st.error(f"Erro ao exportar o modelo: {e}")
        logging.exception("Erro ao exportar o modelo")

def exportar_resultados(y_test, y_pred):
    try:
        resultados = pd.DataFrame({
            'Valores Reais': y_test,
            'Previsões': y_pred
        })
        resultados.to_csv('resultados.csv', index=False)
        st.success("Resultados exportados com sucesso!")
        with open('resultados.csv', 'rb') as f:
            st.download_button('Download Resultados', f, file_name='resultados.csv')
    except Exception as e:
        st.error(f"Erro ao exportar os resultados: {e}")
        logging.exception("Erro ao exportar os resultados")

def comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo):
    st.write("### Comparação com Valores do Artigo:")
    comparison_df = pd.DataFrame({
        'Métricas': ['MSE', 'MAPE', 'R²', 'Erro Médio'],
        'Seu Modelo': [mse, mape, r2, erro_medio],
        'Artigo': [mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo]
    })
    st.table(comparison_df)

# Função principal

def main():
    try:
        # Definir o caminho do ícone e configurar a página
        icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto
        if os.path.exists(icon_path):
            st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
            logging.info(f"Ícone {icon_path} carregado com sucesso.")
        else:
            # Se o ícone não for encontrado, carrega sem favicon
            st.set_page_config(page_title="Geomaker", layout="wide")
            logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")
        
        # Layout da página
        if os.path.exists('capa.png'):
            st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_container_width=True)
            logging.info("Imagem 'capa.png' carregada com sucesso.")
        else:
            st.warning("Imagem 'capa.png' não encontrada.")
            logging.warning("Imagem 'capa.png' não encontrada.")
        
        if os.path.exists("logo.png"):
            st.sidebar.image("logo.png", width=200, use_container_width=True)
            logging.info("Imagem 'logo.png' exibida na sidebar.")
        else:
            st.sidebar.text("Imagem do logotipo não encontrada.")
            logging.warning("Imagem 'logo.png' não encontrada na sidebar.")
        
        st.title("Aplicativo de Aprendizado de Máquina para Previsão de Variações de Águas Subterrâneas (GWS)")
        st.write("Este aplicativo permite treinar modelos de classificação e regressão para prever variações nas águas subterrâneas, com ferramentas avançadas de análise e visualização.")
        
        with st.expander("Transformações de Dados e Engenharia de Features"):
            st.write("""
            # Transformações de Dados e Engenharia de Features

            ## Introdução

            Este aplicativo incorpora técnicas avançadas de pré-processamento de dados e engenharia de features para melhorar a precisão das previsões. As principais transformações incluem:

            - **Tratamento de Valores Nulos:** Preenchimento com média ou moda, conforme apropriado.
            - **Codificação de Variáveis Categóricas:** Utilização de One-Hot Encoding.
            - **Escalonamento de Features:** Aplicação de StandardScaler para normalizar os dados.
            - **Engenharia de Features Temporais:** Extração de ano, mês, dia, dia da semana e estação a partir de colunas de data.
            - **Codificação de Coordenadas Geográficas:** Conversão de latitude e longitude em componentes seno e cosseno.
            - **Remoção de Outliers:** Filtragem de dados com base no z-score.
            """)
        
        # Configurar o sidebar e obter as configurações
        (modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, l2_reg, 
         subsample, colsample_bytree, gamma, min_child_weight, reg_alpha, reg_lambda,
         min_samples_split, min_samples_leaf, max_features, depth, l2_leaf_reg, border_count,
         mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo) = configurar_sidebar()
    
        # Upload de arquivo CSV
        uploaded_file = st.sidebar.file_uploader("Carregue seus dados em CSV", type=["csv"])
    
        if uploaded_file:
            data = carregar_dados(uploaded_file)
            if data is not None:
                st.write("### Pré-visualização dos Dados Carregados:")
                st.write(data.head())

                # Opção para selecionar a coluna de data
                if st.sidebar.checkbox("Os dados contêm coluna de data?"):
                    colunas_temporais = data.select_dtypes(include=['object', 'datetime64[ns]', 'category']).columns.tolist()
                    if not colunas_temporais:
                        st.error("Nenhuma coluna de data encontrada. Verifique os tipos de dados.")
                        st.stop()
                    coluna_tempo = st.sidebar.selectbox('Selecione a coluna de data', colunas_temporais)
                    data = extrair_caracteristicas_temporais(data, coluna_tempo)

                # Opção para selecionar colunas de latitude e longitude
                if st.sidebar.checkbox("Os dados contêm coordenadas geográficas?"):
                    # Filtrar apenas colunas numéricas para seleção
                    colunas_numericas = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    if not colunas_numericas:
                        st.error("Não há colunas numéricas disponíveis para selecionar como Latitude e Longitude.")
                        st.stop()
                    
                    coluna_latitude = st.sidebar.selectbox('Selecione a coluna de Latitude', colunas_numericas)
                    coluna_longitude = st.sidebar.selectbox('Selecione a coluna de Longitude', colunas_numericas)
                    data = codificar_coordenadas(data, coluna_latitude, coluna_longitude)

                # Selecionar a coluna alvo
                colunas_alvo = data.columns.tolist()
                coluna_alvo = st.sidebar.selectbox('Selecione a coluna alvo (target)', colunas_alvo)

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
                    st.write("### Outliers removidos.")

                # Pré-processar os dados
                X_processed, preprocessor, num_cols, cat_cols = preparar_dados(X, y, tipo_problema)
                if X_processed is None:
                    st.stop()

                # Dividir os dados em conjuntos de treino e teste
                if st.sidebar.checkbox("Usar Validação Cruzada Temporal?", value=False):
                    time_series = True
                    tscv = TimeSeriesSplit(n_splits=5)
                    
                    # Definir X_train_full e y_train_full como todo o conjunto de dados processados
                    X_train_full = X_processed
                    y_train_full = y
                    logging.info("Definidas X_train_full e y_train_full para Validação Cruzada Temporal.")
                else:
                    time_series = False
                    X_train_full, X_test, y_train_full, y_test = train_test_split(
                        X_processed, y, test_size=0.2, random_state=42
                    )
                    logging.info("Dividido o conjunto de dados em treino e teste.")

                # Aplicar SMOTE para balanceamento em problemas de classificação
                if tipo_problema == 'Classificação' and not time_series:
                    aplicar_smote_toggle = st.sidebar.checkbox("Aplicar SMOTE para Balanceamento?", value=False)
                    if aplicar_smote_toggle:
                        sm = SMOTE(random_state=42)
                        X_train_full, y_train_full = sm.fit_resample(X_train_full, y_train_full)
                        st.write("### SMOTE aplicado para balanceamento das classes.")

                # Escolher o modelo baseado no tipo de problema
                if tipo_problema == 'Regressão':
                    # Definir parâmetros do modelo de regressão
                    modelo_kwargs = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'reg_lambda': reg_lambda if reg_lambda is not None else 1.0
                    }
                    if modelo_tipo == 'XGBoost':
                        modelo_kwargs.update({
                            'gamma': gamma,
                            'min_child_weight': min_child_weight,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree,
                            'reg_alpha': reg_alpha
                        })
                    elif modelo_tipo == 'Random Forest':
                        modelo_kwargs.update({
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'max_features': max_features
                        })
                    elif modelo_tipo == 'CatBoost':
                        modelo_kwargs.update({
                            'depth': depth,
                            'l2_leaf_reg': l2_leaf_reg,
                            'border_count': border_count
                        })

                    # Treinamento do modelo de regressão
                    if modelo_tipo == 'XGBoost':
                        modelo = XGBRegressor(**modelo_kwargs, random_state=42)
                    elif modelo_tipo == 'CatBoost':
                        modelo = CatBoostRegressor(**modelo_kwargs, verbose=0, random_state=42)
                    elif modelo_tipo == 'Random Forest':
                        modelo = RandomForestRegressor(**modelo_kwargs, random_state=42)
                    elif modelo_tipo == 'Stacking':
                        modelo = stacking_model(tipo_problema)
                    
                    if modelo is None:
                        st.stop()

                    # Aplicar Randomized Search para otimização de hiperparâmetros
                    if st.sidebar.checkbox('Otimizar Hiperparâmetros?'):
                        param_distributions = {}
                        if modelo_tipo == 'XGBoost':
                            param_distributions = {
                                'n_estimators': [100, 200, 300, 500],
                                'max_depth': [4, 6, 8, 10, 12],
                                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                                'gamma': [0, 0.1, 0.2, 0.3],
                                'min_child_weight': [1, 3, 5, 7],
                                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                                'reg_alpha': [0, 0.1, 0.5, 1, 1.5, 2]
                            }
                        elif modelo_tipo == 'Random Forest':
                            param_distributions = {
                                'n_estimators': [100, 200, 300, 500],
                                'max_depth': [None, 10, 20, 30, 40, 50],
                                'min_samples_split': [2, 5, 10, 15],
                                'min_samples_leaf': [1, 2, 4, 6],
                                'max_features': ['auto', 'sqrt', 'log2']
                            }
                        elif modelo_tipo == 'CatBoost':
                            param_distributions = {
                                'depth': [4, 6, 8, 10],
                                'l2_leaf_reg': [1, 3, 5, 7, 9],
                                'border_count': [32, 64, 128, 256]
                            }
                        elif modelo_tipo == 'Stacking':
                            # Para empilhamento, geralmente otimiza-se os hiperparâmetros dos modelos base individualmente
                            # Aqui, podemos optar por não otimizar ou definir parâmetros fixos
                            param_distributions = {}

                        if param_distributions:
                            if time_series:
                                st.warning("Otimização de hiperparâmetros não está implementada para Validação Cruzada Temporal.")
                            else:
                                modelo = otimizar_modelo(modelo, X_train_full, y_train_full, param_distributions, tipo_problema)

                    # Treinar o modelo usando Cross-Validation
                    if time_series:
                        scores = cross_val_score(modelo, X_processed, y, cv=tscv, scoring='neg_mean_squared_error')
                        st.write(f"### Validação Cruzada Temporal (MSE): {-np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
                    else:
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        scores = cross_val_score(modelo, X_train_full, y_train_full, cv=cv, scoring='neg_mean_squared_error')
                        st.write(f"### Validação Cruzada (MSE): {-np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

                        # Ajustar o modelo nos dados completos de treinamento
                        modelo.fit(X_train_full, y_train_full)
                        logging.info("Modelo de regressão treinado com sucesso.")

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

                        # Exibir curvas de aprendizado
                        plotar_curvas_aprendizado(modelo, X_train_full, y_train_full, tipo_problema)

                        # Exportar modelo treinado
                        exportar_modelo(modelo, preprocessor)

                        # Exportar resultados
                        exportar_resultados(y_test, y_pred)

                elif tipo_problema == 'Classificação':
                    # Definir parâmetros do modelo de classificação
                    modelo_kwargs = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'reg_lambda': reg_lambda if reg_lambda is not None else 1.0
                    }
                    if modelo_tipo == 'XGBoost':
                        modelo_kwargs.update({
                            'gamma': gamma,
                            'min_child_weight': min_child_weight,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree,
                            'reg_alpha': reg_alpha
                        })
                    elif modelo_tipo == 'Random Forest':
                        modelo_kwargs.update({
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'max_features': max_features
                        })
                    elif modelo_tipo == 'CatBoost':
                        modelo_kwargs.update({
                            'depth': depth,
                            'l2_leaf_reg': l2_leaf_reg,
                            'border_count': border_count
                        })

                    # Treinamento do modelo de classificação
                    if modelo_tipo == 'XGBoost':
                        modelo = XGBClassifier(**modelo_kwargs, use_label_encoder=False, eval_metric='logloss', random_state=42)
                    elif modelo_tipo == 'CatBoost':
                        modelo = CatBoostClassifier(**modelo_kwargs, verbose=0, random_state=42)
                    elif modelo_tipo == 'Random Forest':
                        modelo = RandomForestClassifier(**modelo_kwargs, random_state=42)
                    elif modelo_tipo == 'Stacking':
                        modelo = stacking_model(tipo_problema)
                    
                    if modelo is None:
                        st.stop()

                    # Aplicar Randomized Search para otimização de hiperparâmetros
                    if st.sidebar.checkbox('Otimizar Hiperparâmetros?'):
                        param_distributions = {}
                        if modelo_tipo == 'XGBoost':
                            param_distributions = {
                                'n_estimators': [100, 200, 300, 500],
                                'max_depth': [4, 6, 8, 10, 12],
                                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                                'gamma': [0, 0.1, 0.2, 0.3],
                                'min_child_weight': [1, 3, 5, 7],
                                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                                'reg_alpha': [0, 0.1, 0.5, 1, 1.5, 2]
                            }
                        elif modelo_tipo == 'Random Forest':
                            param_distributions = {
                                'n_estimators': [100, 200, 300, 500],
                                'max_depth': [None, 10, 20, 30, 40, 50],
                                'min_samples_split': [2, 5, 10, 15],
                                'min_samples_leaf': [1, 2, 4, 6],
                                'max_features': ['auto', 'sqrt', 'log2']
                            }
                        elif modelo_tipo == 'CatBoost':
                            param_distributions = {
                                'depth': [4, 6, 8, 10],
                                'l2_leaf_reg': [1, 3, 5, 7, 9],
                                'border_count': [32, 64, 128, 256]
                            }
                        elif modelo_tipo == 'Stacking':
                            # Para empilhamento, geralmente otimiza-se os hiperparâmetros dos modelos base individualmente
                            # Aqui, podemos optar por não otimizar ou definir parâmetros fixos
                            param_distributions = {}

                        if param_distributions:
                            if time_series:
                                st.warning("Otimização de hiperparâmetros não está implementada para Validação Cruzada Temporal.")
                            else:
                                modelo = otimizar_modelo(modelo, X_train_full, y_train_full, param_distributions, tipo_problema)

                    # Treinar o modelo usando Cross-Validation
                    if time_series:
                        if tipo_problema == 'Classificação':
                            scoring = 'f1_weighted'
                        else:
                            scoring = 'neg_mean_squared_error'
                        scores = cross_val_score(modelo, X_processed, y, cv=tscv, scoring=scoring)
                        metric_name = 'F1 Score' if tipo_problema == 'Classificação' else 'MSE'
                        st.write(f"### Validação Cruzada Temporal ({metric_name}): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
                    else:
                        if tipo_problema == 'Classificação':
                            scoring = 'f1_weighted'
                        else:
                            scoring = 'neg_mean_squared_error'
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        scores = cross_val_score(modelo, X_train_full, y_train_full, cv=cv, scoring=scoring)
                        metric_name = 'F1 Score' if tipo_problema == 'Classificação' else 'MSE'
                        st.write(f"### Validação Cruzada ({metric_name}): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

                        # Ajustar o modelo nos dados completos de treinamento
                        modelo.fit(X_train_full, y_train_full)
                        logging.info("Modelo de classificação treinado com sucesso.")

                        # Fazer previsões no conjunto de teste
                        y_pred = modelo.predict(X_test)
                        y_proba = modelo.predict_proba(X_test) if hasattr(modelo, 'predict_proba') else None

                        # Calcular métricas de classificação
                        metrics = calcular_metricas_classificacao(y_test, y_pred, y_proba)
                        exibir_metricas_classificacao(metrics)

                        # Exibir relatório de classificação
                        st.write("### Relatório de Classificação:")
                        st.text(classification_report(y_test, y_pred))

                        # Exibir matriz de confusão
                        plotar_matriz_confusao(y_test, y_pred)

                        # Exibir curva ROC (para problemas binários)
                        if y_proba is not None and len(np.unique(y_test)) == 2:
                            plotar_curva_roc(y_test, y_proba)

                        # Exibir a importância das features
                        mostrar_importancia_features(modelo, X, preprocessor)

                        # Exibir gráfico de dispersão de previsões vs valores reais
                        plotar_dispersao_previsoes(y_test, y_pred)

                        # Exibir gráfico de resíduos (opcional para classificação)
                        # Pode-se adaptar conforme necessário

                        # Exibir gráfico de comparação de previsões com valores reais
                        # Pode-se adaptar conforme necessário

                        # Exibir curvas de aprendizado
                        plotar_curvas_aprendizado(modelo, X_train_full, y_train_full, tipo_problema)

                        # Exportar modelo treinado
                        exportar_modelo(modelo, preprocessor)

                        # Exportar resultados
                        exportar_resultados(y_test, y_pred)

            # Após treinar e avaliar o modelo, adicionar a seção de previsão
            st.header("🔮 Fazer Previsões com Novos Dados")

            with st.form(key='prediction_form'):
                st.write("Insira os valores para as features abaixo e clique em **Prever** para obter a previsão.")

                # Criar um dicionário para armazenar os inputs do usuário
                user_input = {}

                # Entradas para colunas numéricas
                for col in num_cols:
                    # Obter os valores mínimos e máximos para definir os limites dos sliders
                    min_val = float(X[col].min()) if not X[col].isnull().all() else 0.0
                    max_val = float(X[col].max()) if not X[col].isnull().all() else 1.0
                    mean_val = float(X[col].mean()) if not X[col].isnull().all() else 0.0
                    user_input[col] = st.number_input(f'{col}', value=mean_val, min_value=min_val, max_value=max_val, format="%.4f")

                # Entradas para colunas categóricas
                for col in cat_cols:
                    unique_vals = data[col].dropna().unique().tolist()
                    user_input[col] = st.selectbox(f'{col}', options=unique_vals)

                submit_button = st.form_submit_button(label='Prever')

            if submit_button:
                try:
                    # Criar um DataFrame com os inputs do usuário
                    input_df = pd.DataFrame([user_input])

                    # Pré-processar os dados de entrada
                    X_new_processed = preprocessor.transform(input_df)

                    # Realizar a previsão
                    if tipo_problema == 'Regressão':
                        prediction = modelo.predict(X_new_processed)
                        st.success(f"**Previsão de ΔGWS:** {prediction[0]:.4f}")
                    elif tipo_problema == 'Classificação':
                        prediction = modelo.predict(X_new_processed)
                        prediction_proba = modelo.predict_proba(X_new_processed)
                        st.success(f"**Classe Predita:** {prediction[0]}")
                        st.write(f"**Probabilidades:** {prediction_proba[0]}")
                except Exception as e:
                    st.error(f"Erro ao realizar a previsão: {e}")
                    logging.exception("Erro ao realizar a previsão")

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
        logging.exception("Erro inesperado no main")

    # Imagem e Contatos
    if os.path.exists("eu.ico"):
        st.sidebar.image("eu.ico", width=80, use_container_width=True)
        logging.info("Imagem 'eu.ico' exibida na sidebar.")
    else:
        st.sidebar.text("Imagem do contato não encontrada.")
        logging.warning("Imagem 'eu.ico' não encontrada na sidebar.")

    st.sidebar.write("""
    ### Projeto Geomaker + IA 

    [DOI:10.5281/zenodo.13856575](https://doi.org/10.5281/zenodo.13856575)
    - **Professor:** Marcelo Claro.
    - **Contatos:** marceloclaro@gmail.com
    - **Whatsapp:** (88) 98158-7145
    - **Instagram:** [marceloclaro.geomaker](https://www.instagram.com/marceloclaro.geomaker/)
    """)

    # Controle de Áudio
    st.sidebar.title("Controle de Áudio")

    # Dicionário de arquivos de áudio, com nomes amigáveis mapeando para o caminho do arquivo
    mp3_files = {
        "Áudio explicativo 1": "kariri.mp3",
        # Adicione mais arquivos conforme necessário
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
                    logging.info(f"Áudio '{mp3_path}' reproduzido.")
            except FileNotFoundError:
                st.sidebar.error(f"Arquivo {mp3_path} não encontrado.")
                logging.error(f"Arquivo {mp3_path} não encontrado.")
            except Exception as e:
                st.sidebar.error(f"Erro ao carregar o arquivo: {str(e)}")
                logging.exception("Erro ao carregar o arquivo de áudio.")

# Executar a função principal
if __name__ == "__main__":
    main()
