import streamlit as st
import os
import logging
import base64
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
from sklearn.preprocessing import PowerTransformer

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
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
    
    # Definir o caminho do ícone
    icon_path = "logo.png"  # Verifique se o arquivo logo.png está no diretório correto
    
    # Verificar se o arquivo de ícone existe antes de configurá-lo
    if os.path.exists(icon_path):
        st.set_page_config(page_title="Geomaker", page_icon=icon_path, layout="wide")
        logging.info(f"Ícone {icon_path} carregado com sucesso.")
    else:
        # Se o ícone não for encontrado, carrega sem favicon
        st.set_page_config(page_title="Geomaker", layout="wide")
        logging.warning(f"Ícone {icon_path} não encontrado, carregando sem favicon.")
    
    # Layout da página
    if os.path.exists('capa.png'):
        st.image('capa.png', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always')
    else:
        st.warning("Imagem 'capa.png' não encontrada.")
    
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", width=200)
    else:
        st.sidebar.text("Imagem do logotipo não encontrada.")
    
    #___________________________________________________________
    st.title("Aplicativo de Aprendizado de Máquina para Previsão de Variações de Águas Subterrâneas (GWS)")
    st.write("Este aplicativo permite treinar um modelo de classificação de imagens e aplicar algoritmos de clustering para análise comparativa.")
    with st.expander("Transformações de Dados e Aumento de Dados no Treinamento de Redes Neurais"):
        st.write("""
        # Descrição Meticulosa do Código e do Aplicativo para Previsão de Variações de Águas Subterrâneas (ΔGWS)

        ## Introdução
        
        Este documento fornece uma descrição detalhada do código e do aplicativo desenvolvidos para prever variações nas águas subterrâneas (ΔGWS) utilizando técnicas de aprendizado de máquina. O objetivo principal é criar uma ferramenta robusta e interativa que permita a pesquisadores e profissionais analisar e prever mudanças nos níveis de águas subterrâneas com base em diversos fatores ambientais e climáticos.
        
        O aplicativo é construído em Python e utiliza o framework **Streamlit** para criar uma interface web interativa. Ele incorpora técnicas avançadas de pré-processamento de dados, engenharia de features, seleção de modelos, otimização de hiperparâmetros e avaliação de desempenho. O código foi projetado para ser flexível, permitindo ao usuário escolher entre diferentes modelos de aprendizado de máquina e ajustar os hiperparâmetros conforme necessário.
        
        ## Objetivos do Código e do Aplicativo
        
        - **Prever Variações nas Águas Subterrâneas (ΔGWS)**: Utilizar modelos de aprendizado de máquina para prever mudanças nos níveis de águas subterrâneas com base em dados históricos e fatores ambientais.
        
        - **Interface Interativa**: Fornecer uma interface fácil de usar onde os usuários podem carregar seus próprios dados, selecionar variáveis-alvo, ajustar modelos e visualizar resultados.
        
        - **Engenharia de Features Específicas**: Incorporar características temporais e espaciais para melhorar a precisão das previsões, considerando a natureza temporal e geográfica dos dados.
        
        - **Suporte a Diferentes Tipos de Problemas**: Permitir tanto tarefas de regressão quanto de classificação, dependendo da natureza do problema e dos dados disponíveis.
        
        - **Avaliação e Comparação de Modelos**: Fornecer métricas de desempenho detalhadas e visualizações que ajudam a entender a eficácia do modelo, incluindo comparações com resultados de pesquisas ou artigos científicos.
        
        ## Especificações Técnicas e Científicas
        
        ### 1. **Importação de Bibliotecas**
        
        O código começa com a importação de várias bibliotecas essenciais:
        
        - **Streamlit**: Framework para criação de aplicativos web interativos em Python.
        - **Pandas e NumPy**: Para manipulação e análise de dados.
        - **Scikit-learn**: Biblioteca principal para aprendizado de máquina, incluindo modelos, pré-processamento, seleção de modelos e métricas.
        - **XGBoost e CatBoost**: Bibliotecas especializadas para modelos de gradient boosting, eficientes para tarefas de regressão e classificação.
        - **Matplotlib**: Para criação de gráficos e visualizações.
        - **Imbalanced-learn (SMOTE)**: Para lidar com problemas de desbalanceamento de classes em tarefas de classificação.
        - **Logging**: Para registro de informações e tratamento de exceções.
        
        ### 2. **Carregamento e Tratamento de Dados**
        
        #### Função `carregar_dados(file)`
        
        - **Objetivo**: Carregar dados a partir de um arquivo CSV fornecido pelo usuário e realizar um tratamento inicial.
        - **Processo**:
          - Leitura do arquivo CSV utilizando `pd.read_csv`.
          - Exibição de informações estatísticas dos dados usando `data.describe()`.
          - Verificação e tratamento de valores nulos:
            - Para colunas numéricas, valores nulos são preenchidos com a média (`mean`).
            - Para colunas categóricas, valores nulos são preenchidos com a moda (`mode`).
        - **Justificativa Científica**:
          - O tratamento de valores nulos é crucial para evitar erros durante o treinamento do modelo e para garantir que a informação disponível seja utilizada de forma eficaz.
          - A imputação de valores é uma técnica padrão em pré-processamento de dados.
        
        ### 3. **Engenharia de Features**
        
        #### a) Extração de Características Temporais
        
        ##### Função `extrair_caracteristicas_temporais(dataframe, coluna_tempo)`
        
        - **Objetivo**: Extrair informações temporais de uma coluna de data/hora.
        - **Processo**:
          - Conversão da coluna de data para o tipo `datetime`.
          - Extração de componentes temporais:
            - Ano, mês, dia, dia da semana e estação do ano.
        - **Justificativa Científica**:
          - Dados de águas subterrâneas são influenciados por padrões sazonais e temporais.
          - A inclusão dessas características pode melhorar significativamente a capacidade do modelo de capturar tendências e padrões temporais.
        
        #### b) Codificação de Coordenadas Geográficas
        
        ##### Função `codificar_coordenadas(dataframe, coluna_latitude, coluna_longitude)`
        
        - **Objetivo**: Transformar coordenadas geográficas em representações que possam ser interpretadas pelos modelos.
        - **Processo**:
          - Cálculo das componentes seno e cosseno das latitudes e longitudes.
        - **Justificativa Científica**:
          - Coordenadas geográficas são circulares por natureza.
          - A utilização de funções seno e cosseno permite representar a posição geográfica de forma contínua e preservando a relação espacial.
        
        ### 4. **Pré-Processamento dos Dados**
        
        ##### Função `preparar_dados(X, y, tipo_problema)`
        
        - **Objetivo**: Preparar os dados para o treinamento, incluindo imputação, escalonamento e codificação.
        - **Processo**:
          - Identificação de colunas numéricas e categóricas.
          - Criação de pipelines de pré-processamento para cada tipo de dado:
            - **Numéricos**: Imputação da média e padronização (StandardScaler).
            - **Categóricos**: Imputação da moda e codificação one-hot (OneHotEncoder).
          - Uso do `ColumnTransformer` para aplicar transformações apropriadas a cada tipo de coluna.
        - **Justificativa Científica**:
          - O pré-processamento adequado é essencial para o desempenho dos modelos de aprendizado de máquina.
          - A padronização dos dados numéricos ajuda a acelerar a convergência de algoritmos baseados em gradiente.
          - A codificação one-hot é necessária para que modelos baseados em árvore possam lidar com dados categóricos.
        
        ### 5. **Detecção e Remoção de Outliers**
        
        ##### Função `remover_outliers(X, y, limiar=3)`
        
        - **Objetivo**: Identificar e remover outliers dos dados.
        - **Processo**:
          - Cálculo do Z-score para cada ponto de dados.
          - Remoção de pontos que estão além de um determinado limiar (por padrão, 3 desvios padrão da média).
        - **Justificativa Científica**:
          - Outliers podem distorcer o treinamento do modelo, especialmente em modelos sensíveis a valores extremos.
          - A remoção de outliers pode melhorar a qualidade do modelo e as previsões.
        
        ### 6. **Balanceamento de Classes**
        
        - **Ferramenta**: **SMOTE** (Synthetic Minority Over-sampling Technique).
        - **Aplicação**:
          - Utilizado opcionalmente em tarefas de classificação para balancear classes desbalanceadas.
          - Gera exemplos sintéticos da classe minoritária para equilibrar a distribuição.
        - **Justificativa Científica**:
          - Modelos de classificação podem ser tendenciosos em direção à classe majoritária em conjuntos de dados desbalanceados.
          - O uso do SMOTE ajuda a mitigar esse problema, permitindo que o modelo aprenda representações mais equilibradas.
        
        ### 7. **Modelos de Aprendizado de Máquina**
        
        O aplicativo suporta vários modelos:
        
        #### a) **XGBoost**
        
        - **Descrição**:
          - Implementação eficiente de gradient boosting.
          - Suporta regularização, que ajuda a evitar overfitting.
        - **Parâmetros Principais**:
          - `n_estimators`: Número de árvores.
          - `learning_rate`: Taxa de aprendizado.
          - `max_depth`: Profundidade máxima das árvores.
          - `reg_lambda`: Regularização L2.
        
        #### b) **Random Forest**
        
        - **Descrição**:
          - Conjunto de árvores de decisão construídas em subamostras do conjunto de dados.
          - Usa a média (regressão) ou votação majoritária (classificação) para melhorar a precisão e controlar o overfitting.
        - **Parâmetros Principais**:
          - `n_estimators`, `max_depth`.
        
        #### c) **CatBoost**
        
        - **Descrição**:
          - Modelo de gradient boosting que lida eficientemente com features categóricas sem necessidade de pré-processamento extensivo.
        - **Parâmetros Principais**:
          - Semelhantes ao XGBoost.
        
        #### d) **Stacking (Empilhamento)**
        
        ##### Função `stacking_model(tipo_problema)`
        
        - **Descrição**:
          - Combina previsões de vários modelos base (estimadores) usando um modelo meta (final_estimator).
        - **Estrutura**:
          - **Estimadores Base**:
            - Random Forest, XGBoost, CatBoost.
          - **Estimador Final**:
            - XGBoost (Classificação ou Regressão, conforme o tipo de problema).
        - **Justificativa Científica**:
          - O empilhamento pode melhorar o desempenho combinando diferentes modelos que capturam diferentes aspectos dos dados.
          - Ajuda a reduzir a variância e o viés, potencialmente levando a melhores previsões.
        
        ### 8. **Otimização de Hiperparâmetros**
        
        ##### Função `otimizar_modelo(modelo, X_train, y_train, param_distributions, tipo_problema)`
        
        - **Objetivo**: Encontrar a combinação ideal de hiperparâmetros para o modelo.
        - **Método**: **RandomizedSearchCV**
          - Permite a exploração de uma ampla gama de valores de hiperparâmetros de forma mais eficiente que a grid search.
        - **Processo**:
          - Define distribuições de parâmetros para serem testados.
          - Executa validação cruzada para avaliar o desempenho de cada combinação.
        - **Justificativa Científica**:
          - A otimização de hiperparâmetros é crucial para obter o melhor desempenho possível do modelo.
          - RandomizedSearchCV é eficiente em termos computacionais e pode encontrar boas soluções com menos iterações.
        
        ### 9. **Avaliação de Desempenho**
        
        #### a) **Métricas para Regressão**
        
        ##### Função `calcular_metricas_regressao(y_test, y_pred)`
        
        - **Métricas Calculadas**:
          - **MSE (Mean Squared Error)**: Erro quadrático médio.
          - **RMSE (Root Mean Squared Error)**: Raiz quadrada do MSE.
          - **MAE (Mean Absolute Error)**: Erro absoluto médio.
          - **MAPE (Mean Absolute Percentage Error)**: Erro percentual absoluto médio.
          - **R² (Coeficiente de Determinação)**: Medida da proporção da variância explicada pelo modelo.
          - **Erro Médio Absoluto**: Média das diferenças absolutas entre valores previstos e reais.
        - **Justificativa Científica**:
          - Essas métricas fornecem uma visão abrangente do desempenho do modelo em prever valores contínuos.
          - O MSE e o RMSE penalizam mais erros maiores, enquanto o MAE é mais robusto a outliers.
        
        #### b) **Métricas para Classificação**
        
        ##### Função `calcular_metricas_classificacao(y_test, y_pred, y_proba=None)`
        
        - **Métricas Calculadas**:
          - **Acurácia**: Proporção de previsões corretas.
          - **F1 Score**: Média harmônica de precisão e recall.
          - **Precisão**: Proporção de previsões positivas corretas.
          - **Recall**: Proporção de positivos reais identificados corretamente.
          - **AUC (Area Under the Curve)**: Para problemas binários, mede a capacidade do modelo em distinguir entre classes.
        - **Justificativa Científica**:
          - Essas métricas são essenciais para avaliar modelos de classificação, especialmente em conjuntos de dados desbalanceados.
        
        ### 10. **Visualização de Resultados**
        
        #### a) **Importância das Features**
        
        ##### Função `mostrar_importancia_features(modelo, X, preprocessor)`
        
        - **Objetivo**: Visualizar a importância relativa de cada feature no modelo.
        - **Processo**:
          - Extrai as importâncias das features do modelo (se disponível).
          - Combina os nomes das features numéricas e categóricas após o pré-processamento.
          - Plota um gráfico de barras horizontal mostrando as importâncias.
        - **Justificativa Científica**:
          - Compreender quais features influenciam mais o modelo pode fornecer insights valiosos sobre os fatores que afetam as variações nas águas subterrâneas.
          - Ajuda na interpretabilidade do modelo e na tomada de decisões informadas.
        
        #### b) **Gráficos de Dispersão e Resíduos**
        
        ##### Funções `plotar_dispersao_previsoes(y_test, y_pred)` e `plotar_residuos(y_test, y_pred)`
        
        - **Objetivo**: Visualizar a relação entre as previsões e os valores reais, além de analisar os resíduos.
        - **Processo**:
          - **Dispersão**: Plota as previsões versus os valores reais, com uma linha de referência.
          - **Resíduos**: Plota os resíduos (diferença entre real e previsto) em relação às previsões.
        - **Justificativa Científica**:
          - Gráficos de dispersão ajudam a identificar tendências gerais e possíveis desvios sistemáticos.
          - Análise de resíduos é fundamental para verificar a homocedasticidade e a presença de padrões não capturados pelo modelo.
        
        #### c) **Comparação de Previsões com Valores Reais**
        
        ##### Função `plotar_comparacao_previsoes(y_test, y_pred)`
        
        - **Objetivo**: Comparar diretamente as previsões do modelo com os valores reais ao longo das amostras.
        - **Processo**:
          - Plota duas linhas: uma representando os valores reais e outra as previsões do modelo.
        - **Justificativa Científica**:
          - Permite visualizar como o modelo está acompanhando as variações nos dados reais.
          - Útil para identificar períodos em que o modelo performa melhor ou pior.
        
        #### d) **Matriz de Confusão e Curva ROC** (Para Classificação)
        
        ##### Funções `plotar_matriz_confusao(y_test, y_pred)` e `plotar_curva_roc(y_test, y_proba)`
        
        - **Objetivo**: Avaliar o desempenho do modelo de classificação em termos de verdadeiros positivos, falsos positivos, etc.
        - **Processo**:
          - **Matriz de Confusão**: Mostra a distribuição das previsões corretas e incorretas.
          - **Curva ROC**: Avalia o desempenho do modelo em diferentes limiares de classificação.
        - **Justificativa Científica**:
          - A matriz de confusão ajuda a entender erros específicos do modelo.
          - A curva ROC e o AUC fornecem uma medida agregada de desempenho em todos os limiares.
        
        ### 11. **Comparação com Valores do Artigo**
        
        ##### Função `comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo)`
        
        - **Objetivo**: Comparar as métricas de desempenho do modelo atual com resultados apresentados em um artigo científico ou pesquisa prévia.
        - **Processo**:
          - Exibe as métricas lado a lado.
          - Fornece avisos se houver diferenças significativas.
        - **Justificativa Científica**:
          - A comparação com a literatura existente é essencial para validar o modelo e contextualizar os resultados.
          - Identifica áreas onde o modelo pode estar subperformando ou superando pesquisas anteriores.
        
        ## Fluxo Geral do Aplicativo
        
        1. **Carregamento dos Dados**:
           - O usuário carrega um arquivo CSV contendo os dados.
           - O aplicativo exibe uma pré-visualização dos dados.
        
        2. **Configurações Iniciais**:
           - O usuário seleciona a coluna alvo (variável a ser prevista).
           - Opções para indicar se os dados contêm colunas de data ou coordenadas geográficas.
           - Escolha do tipo de problema (Regressão ou Classificação) e do modelo a ser utilizado.
        
        3. **Pré-Processamento e Engenharia de Features**:
           - Extração de características temporais e codificação de coordenadas, se aplicável.
           - Remoção de outliers (opcional).
           - Aplicação do pipeline de pré-processamento.
        
        4. **Divisão dos Dados**:
           - Divisão em conjuntos de treinamento e teste.
           - Opção de utilizar validação cruzada temporal para dados sequenciais.
        
        5. **Treinamento do Modelo**:
           - Treinamento do modelo selecionado com os hiperparâmetros escolhidos.
           - Opção de otimizar os hiperparâmetros usando RandomizedSearchCV.
        
        6. **Avaliação do Modelo**:
           - Cálculo de métricas de desempenho.
           - Visualização de gráficos e importância das features.
           - Comparação com resultados de artigos, se os valores forem fornecidos.
        
        7. **Interatividade e Ajustes**:
           - O usuário pode ajustar hiperparâmetros, selecionar diferentes modelos e reexecutar o treinamento para melhorar os resultados.
        
        ## Considerações Científicas Adicionais
        
        - **Importância da Engenharia de Features**:
          - A incorporação de características temporais e espaciais é fundamental em problemas relacionados a recursos hídricos.
          - Eventos climáticos, padrões sazonais e localização geográfica influenciam diretamente as variações nas águas subterrâneas.
        
        - **Seleção de Modelos**:
          - Modelos baseados em árvores (como Random Forest e XGBoost) são adequados para lidar com dados não lineares e interações complexas entre features.
          - O uso de modelos ensemble (como o stacking) pode capturar diferentes aspectos dos dados e melhorar a generalização.
        
        - **Validação Cruzada Temporal**:
          - Em séries temporais, é crucial evitar o vazamento de informações do futuro para o passado.
          - O uso de `TimeSeriesSplit` respeita a ordem temporal dos dados, proporcionando uma avaliação mais realista.
        
        - **Interpretação dos Resultados**:
          - Além de métricas quantitativas, a análise visual dos resultados é essencial para entender o comportamento do modelo.
          - A identificação de padrões nos resíduos ou discrepâncias nas previsões pode levar a melhorias no modelo ou no pré-processamento.
        
        ## Conclusão
        
        O código e o aplicativo desenvolvidos proporcionam uma ferramenta abrangente para a previsão de variações nas águas subterrâneas, integrando técnicas avançadas de aprendizado de máquina com considerações específicas do domínio. A flexibilidade do aplicativo permite que usuários com diferentes níveis de conhecimento possam utilizá-lo para explorar seus dados, ajustar modelos e interpretar resultados.
        
        Ao abordar meticulosamente as etapas de preparação dos dados, seleção e treinamento de modelos, e avaliação de desempenho, o código serve não apenas como uma aplicação prática, mas também como um estudo detalhado das melhores práticas em aprendizado de máquina aplicado a problemas ambientais e hidrogeológicos.
        
        ## Recomendações para Uso e Melhoria
        
        - **Dados de Qualidade**: A precisão das previsões depende da qualidade e relevância dos dados. É recomendável utilizar conjuntos de dados atualizados e representativos.
        
        - **Exploração de Outros Modelos**: Embora o aplicativo ofereça vários modelos, a exploração de redes neurais ou outros algoritmos pode ser benéfica, dependendo do problema.
        
        - **Engenharia de Features Adicional**: A inclusão de outras variáveis, como uso do solo, precipitação, evapotranspiração, entre outras, pode melhorar o desempenho.
        
        - **Validação Externa**: Testar o modelo em dados de diferentes regiões ou períodos pode ajudar a avaliar sua capacidade de generalização.
        
        - **Colaboração Multidisciplinar**: Trabalhar em conjunto com especialistas em hidrologia e geologia pode fornecer insights valiosos para a interpretação dos resultados e orientação do modelo.
        """)
    
    #__________________________________________

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
            if tipo_problema == 'Classificação' and not time_series:
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
                
                if modelo is None:
                    st.stop()

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
                
                if modelo is None:
                    st.stop()

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
    #___________________________________________

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
    unique_classes = np.unique(y_test)
    ax.set_xticks(range(len(unique_classes)))
    ax.set_xticklabels(unique_classes, rotation=45)
    ax.set_yticks(range(len(unique_classes)))
    ax.set_yticklabels(unique_classes)
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

    # Executar a função principal
    if __name__ == "__main__":
        main()
