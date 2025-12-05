import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Modelos Lineares ENEM P1 - Final")

# Variáveis Globais
Y_NAME = 'NOTA_MT_MATEMATICA'
DATA_FILE = "Enem_2024_Amostra_Perfeita.xlsx"

# Nomes das variáveis preditoras originais
X_NAMES_CANDIDATES = [
    'NOTA_CN_CIENCIAS_DA_NATUREZA',
    'NOTA_CH_CIENCIAS_HUMANAS',
    'NOTA_LC_LINGUAGENS_E_CODIGOS',
    'NOTA_REDACAO'
]
MODEL1_NAMES = ['NOTA_CN_CIENCIAS_DA_NATUREZA', 'NOTA_CH_CIENCIAS_HUMANAS', 'NOTA_REDACAO']
MODEL2_NAMES = ['NOTA_CN_CIENCIAS_DA_NATUREZA', 'NOTA_REDACAO']


# --- FUNÇÕES DE ANÁLISE REQUERIDAS PELO PIPELINE ---

def calculate_beta_hat_matricial(X, Y):
    """Calcula o vetor de coeficientes beta_hat usando álgebra matricial."""
    XTX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XTX_inv @ X.T @ Y
    return beta_hat

def calculate_metrics(Y_true, Y_scores, threshold):
    """Calcula todas as métricas de classificação."""
    Y_pred = (Y_scores >= threshold).astype(int)
    cm = confusion_matrix(Y_true, Y_pred)
    
    # Proteção contra erros de matriz de confusão incompleta
    if cm.size != 4:
        return None 
    tn, fp, fn, tp = cm.ravel()
    
    auc = roc_auc_score(Y_true, Y_scores)
    accuracy = accuracy_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred, zero_division=0)
    sensitivity = recall_score(Y_true, Y_pred, zero_division=0) 
    specificity = tn / (tn + fp)
    f1 = f1_score(Y_true, Y_pred, zero_division=0)
    
    return {
        'Curva ROC/AUC': auc, 
        'Acurácia': accuracy, 
        'Precisão': precision, 
        'Sensibilidade': sensitivity, 
        'Especificidade': specificity, 
        'F1 Score': f1
    }

class OLS_Wrapper(BaseEstimator, RegressorMixin):
    """Wrapper para usar OLS no K-Fold do Sklearn."""
    def __init__(self, add_constant=True):
        self.add_constant = add_constant
    def fit(self, X, y):
        if self.add_constant:
            X = sm.add_constant(X)
        self.model = sm.OLS(y, X).fit()
        return self
    def predict(self, X):
        if self.add_constant:
            X = sm.add_constant(X)
        return self.model.predict(X)

# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO (CACHED) ---
@st.cache_data
def load_and_process_data():
    """Carrega dados, executa split, fits, e calcula todas as métricas."""
    
    try:
        df = pd.read_excel(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo {DATA_FILE} não encontrado. Certifique-se de que ele está no mesmo diretório.")
        return None, None, None, None, None, None, None

    # 1. Limpeza e Split
    df_model = df[[Y_NAME] + X_NAMES_CANDIDATES].dropna()
    X1_data = df_model[MODEL1_NAMES]
    X2_data = df_model[MODEL2_NAMES]
    Y_data = df_model[Y_NAME]

    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_data, Y_data, test_size=0.3, random_state=42)
    X2_train, X2_test, _, _ = train_test_split(X2_data, Y_data, test_size=0.3, random_state=42)

    X1_train_const = sm.add_constant(X1_train)
    X2_train_const = sm.add_constant(X2_train)
    X1_test_const = sm.add_constant(X1_test)
    X2_test_const = sm.add_constant(X2_test)

    # 2. Fits dos Modelos
    model1_func = sm.OLS(Y_train, X1_train_const).fit()
    model2_func = sm.OLS(Y_train, X2_train_const).fit()
    model1_robust = sm.OLS(Y_train, X1_train_const).fit(cov_type='HC3') # Modelo de inferência final

    # 3. Métricas de Regressão e Parcimônia
    
    # AIC/BIC
    aic1, bic1 = model1_func.aic, model1_func.bic
    aic2, bic2 = model2_func.aic, model2_func.bic

    # RMSE (Teste - Função)
    Y1_pred_test = model1_func.predict(X1_test_const)
    Y2_pred_test = model2_func.predict(X2_test_const)
    rmse1 = np.sqrt(mean_squared_error(Y_test, Y1_pred_test))
    rmse2 = np.sqrt(mean_squared_error(Y_test, Y2_pred_test))

    # K-Fold RMSE
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_rmse1_kf = -cross_val_score(OLS_Wrapper(add_constant=True), X1_train, Y_train, scoring='neg_root_mean_squared_error', cv=cv).mean()
    mean_rmse2_kf = -cross_val_score(OLS_Wrapper(add_constant=True), X2_train, Y_train, scoring='neg_root_mean_squared_error', cv=cv).mean()
    
    # 4. Métricas de Classificação
    median_Y_train = Y_train.median()
    Y_test_class = (Y_test >= median_Y_train).astype(int)
    metrics1 = calculate_metrics(Y_test_class, Y1_pred_test, median_Y_train)
    metrics2 = calculate_metrics(Y_test_class, Y2_pred_test, median_Y_train)

    # 5. Multicolinearidade (VIF)
    def calculate_vif_train(df_X):
        df_X_no_const = df_X.drop(columns=['const'])
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_X_no_const.columns
        vif_data["VIF"] = [variance_inflation_factor(df_X_no_const.values, i) for i in range(df_X_no_const.shape[1])]
        return vif_data
    
    vif_m1 = calculate_vif_train(X1_train_const).set_index('feature').loc[MODEL1_NAMES[0]:].max().values[0]
    vif_m2 = calculate_vif_train(X2_train_const).set_index('feature').loc[MODEL2_NAMES[0]:].max().values[0]


    # 6. Comparação Final (Tabela)
    comparison_data = {
        'Métrica': ['RMSE (Teste)', 'AIC', 'BIC', 'RMSE K-fold', 'Curva ROC/AUC', 'F1 Score', 'Acurácia', 'VIF Máximo'],
        'Modelo 1 (Vencedor)': [rmse1, aic1, bic1, mean_rmse1_kf, metrics1['Curva ROC/AUC'], metrics1['F1 Score'], metrics1['Acurácia'], vif_m1],
        'Modelo 2 (Parcimonioso)': [rmse2, aic2, bic2, mean_rmse2_kf, metrics2['Curva ROC/AUC'], metrics2['F1 Score'], metrics2['Acurácia'], vif_m2]
    }
    df_comparison = pd.DataFrame(comparison_data).set_index('Métrica')

    # 7. Comparação Matricial vs. Função (Coeficientes)
    B1_hat_mat = calculate_beta_hat_matricial(X1_train_const.to_numpy(), Y_train.to_numpy().reshape(-1, 1))
    B2_hat_mat = calculate_beta_hat_matricial(X2_train_const.to_numpy(), Y_train.to_numpy().reshape(-1, 1))
    
    coefs_ols = {
        'Modelo': ['Modelo 1 (Função)', 'Modelo 1 (Matricial)', 'Modelo 2 (Função)', 'Modelo 2 (Matricial)'],
        'Intercepto': [model1_func.params['const'], B1_hat_mat[0, 0], model2_func.params['const'], B2_hat_mat[0, 0]],
        MODEL1_NAMES[0]: [model1_func.params[MODEL1_NAMES[0]], B1_hat_mat[1, 0], model2_func.params[MODEL2_NAMES[0]], B2_hat_mat[1, 0]],
        MODEL1_NAMES[2]: [model1_func.params[MODEL1_NAMES[2]], B1_hat_mat[3, 0], model2_func.params[MODEL2_NAMES[1]], B2_hat_mat[2, 0]],
    }
    df_coefs_ols = pd.DataFrame(coefs_ols).set_index('Modelo')
    
    
    return model1_func, model1_robust, X1_train_const, Y_train, Y_test_class, Y1_pred_test, Y2_pred_test, df_comparison, df_coefs_ols


# --- GERAÇÃO DE GRÁFICOS (CHAMADA APENAS NO STREAMLIT) ---

def generate_roc_curve(Y_true, Y1_scores, Y2_scores, auc1, auc2):
    """Gera o gráfico da Curva ROC."""
    fpr1, tpr1, _ = roc_curve(Y_true, Y1_scores)
    fpr2, tpr2, _ = roc_curve(Y_true, Y2_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr1, tpr1, label=f'Modelo 1 (AUC = {auc1:.4f})')
    ax.plot(fpr2, tpr2, label=f'Modelo 2 (AUC = {auc2:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    ax.set_title('Curva ROC para Comparação de Modelos')
    ax.legend()
    ax.grid(True)
    return fig

def generate_residual_plots(model_func, Y_train):
    """Gera os gráficos de Resíduos vs. Ajustados e Q-Q Plot."""
    resid = model_func.resid
    fitted = model_func.fittedvalues

    # Resíduos vs. Ajustados
    fig_resid, ax_resid = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=resid, alpha=0.3, ax=ax_resid)
    ax_resid.axhline(0, color='red', linestyle='--')
    ax_resid.set_title('Linearidade e Homocedasticidade')
    ax_resid.set_xlabel('Valores Ajustados')
    ax_resid.set_ylabel('Resíduos')

    # Q-Q Plot
    fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
    sm.qqplot(resid, line='s', fit=True, ax=ax_qq)
    ax_qq.set_title('Normalidade dos Resíduos (Q-Q Plot)')
    
    return fig_resid, fig_qq


# --- ESTRUTURA DO STREAMLIT ---

st.title("🎓 Modelos Lineares ENEM: Análise e Seleção de Modelos")
st.markdown("### Atividade Avaliativa P1 - Álgebra Matricial vs. Python")
st.markdown("---")

# Carregar os dados processados (usando cache)
(model1_func, model1_robust, X1_train_const, Y_train, Y_test_class, 
 Y1_pred_test, Y2_pred_test, df_comparison, df_coefs_ols) = load_and_process_data()

if df_comparison is None:
    st.stop()


# --------------------------------------------------------------------------
# --- SEÇÃO 1: SELEÇÃO FINAL E INFERÊNCIA ---
# --------------------------------------------------------------------------

st.header("1. Modelo Vencedor e Resultados Finais")
st.markdown(f"""
O modelo selecionado é o **Modelo 1** ($\text{{NOTA\_CN, NOTA\_CH, NOTA\_REDACAO}}$), que demonstrou a melhor performance preditiva e estabilidade ($\mathbf{{RMSE}}={df_comparison.loc['RMSE (Teste)']['Modelo 1 (Vencedor)']:.4f}$).
""")

st.subheader("Modelo Final de Inferência (Corrigido para Heterocedasticidade)")

st.latex(r'''
\mathbf{\hat{Y}_{\text{MT}}} = -0.7020 + 0.3535 \cdot \text{NOTA\_CN} + 0.3194 \cdot \text{NOTA\_CH} + 0.3313 \cdot \text{NOTA\_REDACAO}
''')

st.dataframe(pd.DataFrame({
    'Variável': model1_robust.params.index,
    'Coeficiente (β̂)': model1_robust.params.values,
    'Erro Padrão (HC3)': model1_robust.bse.values,
    'P-valor (Corrigido)': model1_robust.pvalues.values
}).style.format({
    'Coeficiente (β̂)': '{:.4f}',
    'Erro Padrão (HC3)': '{:.4f}',
    'P-valor (Corrigido)': '{:.4e}'
}), hide_index=True, use_container_width=True)

with st.expander("⚠️ Consequências da Multicolinearidade e Heterocedasticidade"):
    st.markdown("""
    - **Heterocedasticidade (Tratada):** Os Erros Padrão acima são **Robustos (HC3)**. Isso corrige a inconsistência dos $\text{p-valores}$ e garante a **validade estatística da Inferência** sobre a significância dos preditores.
    - **Multicolinearidade ($\mathbf{VIF \approx 300}$):** O modelo é excelente para **Previsão** ($\mathbf{\hat{Y}}$), mas o alto $\text{VIF}$ implica que a **Interpretabilidade** dos coeficientes individuais ($\mathbf{\hat{\beta}}$) é instável e não confiável.
    """)

st.markdown("---")

# --------------------------------------------------------------------------
# --- SEÇÃO 2: TABELA DE COMPARAÇÃO (PIPELINE) ---
# --------------------------------------------------------------------------

st.header("2. Comparação de Modelos (Pipeline - Roteiro 6)")
st.markdown("Validação da estabilidade e generalização no conjunto de **Teste**.")

# Customização da Tabela de Comparação
df_display = df_comparison.copy()
df_display = df_display.style.format('{:.4f}')

st.dataframe(df_display.highlight_min(
    axis=1, 
    subset=pd.IndexSlice[['RMSE (Teste)', 'AIC', 'BIC', 'RMSE K-fold'], :], 
    props='background-color: #d4edda;' # Verde claro para vencedor (menor)
).highlight_max(
    axis=1,
    subset=pd.IndexSlice[['Curva ROC/AUC', 'F1 Score', 'Acurácia'], :],
    props='background-color: #d4edda;'
), use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------
# --- SEÇÃO 3: DIAGNÓSTICO E VALIDAÇÃO MATRICIAL ---
# --------------------------------------------------------------------------

st.header("3. Dashboard de Diagnóstico e Validação")

col_diag1, col_diag2 = st.columns(2)

with col_diag1:
    st.subheader("Gráficos de Suposições (Modelo 1 - Treino)")
    
    # Geração dos gráficos de diagnóstico
    fig_resid, fig_qq = generate_residual_plots(model1_func, Y_train)
    
    st.pyplot(fig_resid)
    st.caption("Resíduos vs. Ajustados: Dispersão em funil indica Heterocedasticidade (tratada via HC3).")
    
    st.pyplot(fig_qq)
    st.caption("Q-Q Plot: Normalidade aproximada, com caudas pesadas (extremos).")

with col_diag2:
    st.subheader("Validação Matricial e Curva ROC")
    
    # Curva ROC
    auc1 = df_comparison.loc['Curva ROC/AUC', 'Modelo 1 (Vencedor)']
    auc2 = df_comparison.loc['Curva ROC/AUC', 'Modelo 2 (Parcimonioso)']
    fig_roc = generate_roc_curve(Y_test_class, Y1_pred_test, Y2_pred_test, auc1, auc2)
    st.pyplot(fig_roc)
    st.caption(f"Curva ROC: Modelo 1 (verde) demonstra maior poder de discriminação (AUC = {auc1:.4f}).")
    
    st.markdown("##### Coeficientes: Matricial vs. Função")
    st.markdown("A equivalência prova a correção do Estimador de Mínimos Quadrados ($\mathbf{\hat{\beta}}$).")
    st.dataframe(df_coefs_ols.T.style.format('{:.6f}'), use_container_width=True)

st.markdown("---")
st.info("✅ O projeto está concluído. O Modelo 1 é a solução mais robusta e preditiva, com validação e correção estatística.")
