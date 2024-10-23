import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, mean_squared_error,
    mean_absolute_percentage_error, r2_score, confusion_matrix, roc_auc_score,
    classification_report, roc_curve, auc
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

# Função para preparar os dados (pré-processamento)
def preparar_dados(X, y, tipo_problema):
    try:
        # Identificar colunas numéricas e categóricas
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns

        # Pipelines para colunas numéricas e categóricas
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
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
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    erro_medio = np.mean(np.abs(y_test - y_pred))
    return mse, mape, r2, erro_medio

# Exibir métricas de comparação
def exibir_metricas_regressao(mse, mape, r2, erro_medio):
    st.write(f"**Erro Médio Quadrado (MSE):** {mse:.4f}")
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

# Função para exibir a importância das features
def mostrar_importancia_features(modelo, X, preprocessor):
    try:
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
            # Obter os nomes das features após o pré-processamento
            features = preprocessor.transformers_[0][2].tolist()  # Colunas numéricas
            cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out()
            features.extend(cat_features)
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

# Função principal
def main():
    st.title("Aplicativo de Aprendizado de Máquina")
    modelo_tipo, tipo_problema, n_estimators, learning_rate, max_depth, l2_reg, subsample, colsample_bytree, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo = configurar_sidebar()

    uploaded_file = st.sidebar.file_uploader("Carregue seus dados em CSV", type=["csv"])

    if uploaded_file:
        data = carregar_dados(uploaded_file)
        if data is not None:
            st.write("Pré-visualização dos Dados Carregados:")
            st.write(data.head())

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
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(modelo, X_train_full, y_train_full, cv=cv, scoring='neg_mean_squared_error')
                st.write(f"Validação Cruzada (MSE): {-np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

                # Ajustar o modelo nos dados completos de treinamento
                modelo.fit(X_train_full, y_train_full)

                # Fazer previsões no conjunto de teste
                y_pred = modelo.predict(X_test)

                # Calcular métricas de desempenho de regressão
                mse, mape, r2, erro_medio = calcular_metricas_regressao(y_test, y_pred)
                exibir_metricas_regressao(mse, mape, r2, erro_medio)

                # Comparar com os valores do artigo (se fornecidos)
                if mse_artigo > 0:
                    comparar_com_artigo(mse, mape, r2, erro_medio, mse_artigo, mape_artigo, r2_artigo, erro_medio_artigo)

                # Exibir a importância das features
                mostrar_importancia_features(modelo, X, preprocessor)

                # Exibir gráfico de dispersão de previsões vs valores reais
                plotar_dispersao_previsoes(y_test, y_pred)

                # Exibir gráfico de resíduos
                plotar_residuos(y_test, y_pred)

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

# Executar a função principal
if __name__ == "__main__":
    main()
