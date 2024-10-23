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

# Função para carregar e tratar os dados
def carregar_dados(file):
    data = pd.read_csv(file)

    # Exibir informações dos dados carregados
    st.write("Informações dos Dados:")
    st.write(data.describe())
    
    # Verificar e tratar valores nulos
    if data.isnull().sum().sum() > 0:
        st.warning("Os dados contêm valores nulos. Eles serão preenchidos com a média.")
        data.fillna(data.mean(), inplace=True)
    
    return data

# Função para normalizar os dados
def normalizar_dados(X_train, X_val, X_test, metodo='MinMax'):
    if metodo == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Ajustar a escala com os dados de treino e transformar
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled

# Função para treinar os modelos com parada precoce
def treinar_modelo(modelo_tipo, X_train, y_train, X_val, y_val, **kwargs):
    if modelo_tipo == 'XGBoost':
        modelo = XGBClassifier(**kwargs)
        modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=kwargs.get('early_stopping_rounds', 10), verbose=False)
    elif modelo_tipo == 'CatBoost':
        modelo = CatBoostClassifier(**kwargs, verbose=0)
        modelo.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=kwargs.get('early_stopping_rounds', 10))
    else:
        modelo = RandomForestClassifier(**kwargs)
        modelo.fit(X_train, y_train)
    return modelo

# Função para calcular as métricas de regressão
def calcular_metricas_regressao(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mape, r2

# Exibir métricas de comparação
def exibir_metricas_comparacao(mse, mape, r2):
    st.write(f"Erro Médio Quadrado (MSE): {mse:.4f}")
    st.write(f"Erro Percentual Absoluto Médio (MAPE): {mape:.4f}")
    st.write(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Função para validação cruzada e score médio
def validacao_cruzada(modelo, X_train, y_train, cv=5):
    scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')
    return np.mean(scores), np.std(scores)

# Exibir importância das features
def mostrar_importancia_features(modelo, X):
    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
        features = X.columns
        importancia_df = pd.DataFrame({'Features': features, 'Importância': importancias})
        importancia_df = importancia_df.sort_values(by='Importância', ascending=False)
        
        st.write("Importância das Variáveis (Features):")
        fig, ax = plt.subplots()
        importancia_df.plot(kind='bar', x='Features', y='Importância', legend=False, ax=ax)
        plt.title("Importância das Features")
        plt.ylabel("Importância")
        plt.tight_layout()
        st.pyplot(fig)

# Plotar curva ROC
def plotar_curva_roc(modelo, X_test_scaled, y_test):
    if hasattr(modelo, "predict_proba"):
        y_score = modelo.predict_proba(X_test_scaled)
        if y_score.shape[1] == 2:  # Classificação binária
            y_score = y_score[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Taxa de Falsos Positivos')
            ax.set_ylabel('Taxa de Verdadeiros Positivos')
            ax.set_title('Curva ROC')
            ax.legend(loc="lower right")
            st.pyplot(fig)
        else:  # Multi-classe
            for i in range(y_score.shape[1]):
                fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, lw=2, label=f'Classe {i} (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Taxa de Falsos Positivos')
                ax.set_ylabel('Taxa de Verdadeiros Positivos')
                ax.set_title(f'Curva ROC para Classe {i}')
                ax.legend(loc="lower right")
                st.pyplot(fig)
    else:
        st.warning("Este modelo não suporta a previsão de probabilidades para a curva ROC.")

# Plotar histogramas das previsões
def plotar_histograma_previsoes(y_test, y_pred):
    st.write("Histograma das Previsões vs Valores Reais:")
    fig, ax = plt.subplots()
    ax.hist(y_test, bins=10, alpha=0.5, label='Valores Reais', color='blue')
    ax.hist(y_pred, bins=10, alpha=0.5, label='Valores Previstos', color='red')
    ax.set_title('Histograma das Previsões')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Frequência')
    ax.legend()
    st.pyplot(fig)

# Configuração da barra lateral
st.sidebar.title("Configurações dos Modelos")

# Escolha do modelo
modelo_tipo = st.sidebar.selectbox('Escolha o Modelo', ['XGBoost', 'Random Forest', 'CatBoost'])

# Escolha de Classificação ou Regressão
tipo_problema = st.sidebar.selectbox('Escolha o Tipo de Problema', ['Classificação', 'Regressão'])

# Hiperparâmetros configuráveis para cada modelo
n_estimators = st.sidebar.slider('Número de Árvores (n_estimators)', 100, 1000, 300)
learning_rate = st.sidebar.slider('Taxa de Aprendizado (learning_rate)', 0.01, 0.3, 0.1)
max_depth = st.sidebar.slider('Profundidade Máxima (max_depth)', 3, 10, 6)
early_stopping_rounds = st.sidebar.slider('Parada Precoce (early_stopping_rounds)', 10, 100, 50)
l2_reg = st.sidebar.slider('Regularização L2 (Weight Decay)', 0.01, 1.0, 0.1)

# Hiperparâmetros específicos para o XGBoost
if modelo_tipo == 'XGBoost':
    subsample = st.sidebar.slider('Subsample (Taxa de Amostragem)', 0.5, 1.0, 0.8)

    colsample_bytree = st.sidebar.slider('ColSample ByTree (Taxa de Colunas por Árvore)', 0.5, 1.0, 0.8)

# Escolha do método de normalização
metodo_normalizacao = st.sidebar.selectbox('Método de Normalização', ['MinMax', 'Z-Score'])

# Upload do arquivo CSV
uploaded_file = st.sidebar.file_uploader("Carregue seus dados em CSV", type=["csv"])

if uploaded_file:
    # Carregar e tratar os dados
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

    # Dividir os dados em conjuntos de treino, validação e teste
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Normalizar os dados de entrada
    X_train_scaled, X_val_scaled, X_test_scaled = normalizar_dados(X_train, X_val, X_test, metodo=metodo_normalizacao)

    # Escolher modelo com base no tipo de problema
    if tipo_problema == 'Classificação':
        # Treinar o modelo selecionado com os parâmetros para classificação
        modelo_kwargs = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'reg_lambda': l2_reg
        }

        if modelo_tipo == 'XGBoost':
            modelo_kwargs['subsample'] = subsample
            modelo_kwargs['colsample_bytree'] = colsample_bytree
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

        st.write(f"Acurácia no Conjunto de Teste: {acc:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"Precisão: {precision:.4f}")
        st.write(f"Revocação: {recall:.4f}")

        # Realizar a validação cruzada
        cv_acc_mean, cv_acc_std = validacao_cruzada(modelo, X_train_scaled, y_train)
        st.write(f"Média da Acurácia na Validação Cruzada: {cv_acc_mean:.4f} ± {cv_acc_std:.4f}")

        # Plotar a matriz de confusão
        st.write("Matriz de Confusão:")
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

        # Plotar a curva ROC
        st.write("Curva ROC:")
        plotar_curva_roc(modelo, X_test_scaled, y_test)

        # Plotar histograma das previsões
        plotar_histograma_previsoes(y_test, y_pred)

    else:
        # Treinar o modelo selecionado com os parâmetros para regressão
        modelo_kwargs = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'reg_lambda': l2_reg
        }

        if modelo_tipo == 'XGBoost':
            modelo_kwargs['subsample'] = subsample
            modelo_kwargs['colsample_bytree'] = colsample_bytree
            modelo = XGBRegressor(**modelo_kwargs)
        elif modelo_tipo == 'CatBoost':
            modelo = CatBoostRegressor(**modelo_kwargs, verbose=0)
        else:
            modelo = RandomForestRegressor(**modelo_kwargs)

        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)

        # Calcular métricas de regressão
        mse, mape, r2 = calcular_metricas_regressao(y_test, y_pred)
        exibir_metricas_comparacao(mse, mape, r2)

        # Plotar histograma das previsões para regressão
        plotar_histograma_previsoes(y_test, y_pred)

    # Exibir a importância das features
    mostrar_importancia_features(modelo, X)
