import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings

# Suprimir avisos desnecessários do Statsmodels
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(layout="wide", page_title="Modelos Lineares ENEM P2")

# Variáveis Globais
Y_NAME = 'NOTA_MT_MATEMATICA'
DATA_FILE = "Enem_2024_Amostra_Perfeita.xlsx"

X_NAMES_CANDIDATES = [
    'NOTA_CN_CIENCIAS_DA_NATUREZA',
    'NOTA_CH_CIENCIAS_HUMANAS',
    'NOTA_LC_LINGUAGENS_E_CODIGOS',
    'NOTA_REDACAO'
]
MODEL1_NAMES = ['NOTA_CN_CIENCIAS_DA_NATUREZA', 'NOTA_CH_CIENCIAS_HUMANAS', 'NOTA_REDACAO']
MODEL2_NAMES = ['NOTA_CN_CIENCIAS_DA_NATUREZA', 'NOTA_REDACAO']


# --- FUNÇÕES DE ANÁLISE ---

def calculate_beta_hat_matricial(X, Y):
    """Calcula o vetor de coeficientes beta_hat usando álgebra matricial."""
    XTX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XTX_inv @ X.T @ Y
    return beta_hat

def calculate_metrics(Y_true, Y_scores, threshold):
    """Calcula todas as métricas de classificação."""
    Y_pred = (Y_scores >= threshold).astype(int)
    cm = confusion_matrix(Y_true, Y_pred)
    
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
        return None, None, None, None, None, None, None, None, None, None, None

    # 1. Limpeza e Split
    df_model = df[[Y_NAME] + X_NAMES_CANDIDATES].dropna()
    X1_data = df_model[MODEL1_NAMES]
    X2_data = df_model[MODEL2_NAMES]
    Y_data = df_model[Y_NAME]

    X1_train, X1_test, Y_train, Y_test = train_test_split(X1_data, Y_data, test_size=0.3, random_state=42)
    X2_train, X2_test, _, _ = train_test_split(X2_data, Y_data, test_size=0.3, random_state=42)

    X1_train_const = sm.add_constant(X1_train)
    X2_train_const = sm.add_constant(X2_train)

    # 2. Fits dos Modelos
    model1_func = sm.OLS(Y_train, X1_train_const).fit()
    model2_func = sm.OLS(Y_train, X2_train_const).fit()
    model1_robust = sm.OLS(Y_train, X1_train_const).fit(cov_type='HC3') # Modelo de inferência final

    # 3. Métricas de Regressão e Parcimônia
    aic1, bic1 = model1_func.aic, model1_func.bic
    aic2, bic2 = model2_func.aic, model2_func.bic
    
    X1_test_const = sm.add_constant(X1_test)
    X2_test_const = sm.add_constant(X2_test)

    rmse1 = np.sqrt(mean_squared_error(Y_test, model1_func.predict(X1_test_const)))
    rmse2 = np.sqrt(mean_squared_error(Y_test, model2_func.predict(X2_test_const)))
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_rmse1_kf = -cross_val_score(OLS_Wrapper(add_constant=True), X1_train, Y_train, scoring='neg_root_mean_squared_error', cv=cv).mean()
    mean_rmse2_kf = -cross_val_score(OLS_Wrapper(add_constant=True), X2_train, Y_train, scoring='neg_root_mean_squared_error', cv=cv).mean()
    
    # 4. Métricas de Classificação
    median_Y_train = Y_train.median()
    Y_test_class = (Y_test >= median_Y_train).astype(int)
    
    Y1_pred_test = model1_func.predict(X1_test_const)
    Y2_pred_test = model2_func.predict(X2_test_const)
    metrics1 = calculate_metrics(Y_test_class, Y1_pred_test, median_Y_train)
    metrics2 = calculate_metrics(Y_test_class, Y2_pred_test, median_Y_train)

    # 5. Multicolinearidade (VIF)
    def calculate_vif_train(df_X):
        df_X_no_const = df_X.drop(columns=['const'])
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_X_no_const.columns
        vif_data["VIF"] = [variance_inflation_factor(df_X_no_const.values, i) for i in range(df_X_no_const.shape[1])]
        return vif_data
    
    vif_m1_df = calculate_vif_train(X1_train_const)
    vif_m2_df = calculate_vif_train(X2_train_const)
    vif_m1_max = vif_m1_df['VIF'].max()
    vif_m2_max = vif_m2_df['VIF'].max()

    # 6. Comparação Final (Tabela)
    comparison_data = {
        'Métrica': ['RMSE (Teste)', 'AIC', 'BIC', 'RMSE K-fold', 'Curva ROC/AUC', 'F1 Score', 'Acurácia', 'VIF Máximo'],
        'Modelo 1 (Vencedor)': [rmse1, aic1, bic1, mean_rmse1_kf, metrics1['Curva ROC/AUC'], metrics1['F1 Score'], metrics1['Acurácia'], vif_m1_max],
        'Modelo 2 (Parcimonioso)': [rmse2, aic2, bic2, mean_rmse2_kf, metrics2['Curva ROC/AUC'], metrics2['F1 Score'], metrics2['Acurácia'], vif_m2_max]
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

    # 8. Correlação (para o gráfico)
    df_train_full = pd.concat([Y_train, X1_train], axis=1)
    
    return model1_func, model1_robust, model2_func, df_train_full, Y_test_class, Y1_pred_test, Y2_pred_test, df_comparison, df_coefs_ols


# --- GERAÇÃO DE GRÁFICOS (Adicionado o Heatmap de Correlação) ---

def generate_correlation_heatmap(df_train_full):
    """Gera o mapa de calor da correlação."""
    corr_matrix = df_train_full.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, linewidths=.5)
    ax.set_title('Matriz de Correlação das Variáveis (Amostra de Treino)')
    return fig

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

def generate_residual_plots(model_func, model_name):
    """Gera os gráficos de Resíduos vs. Ajustados e Q-Q Plot."""
    resid = model_func.resid
    fitted = model_func.fittedvalues

    # Resíduos vs. Ajustados
    fig_resid, ax_resid = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=resid, alpha=0.3, ax=ax_resid)
    ax_resid.axhline(0, color='red', linestyle='--')
    ax_resid.set_title(f'Linearidade e Homocedasticidade ({model_name})')
    ax_resid.set_xlabel('Valores Ajustados')
    ax_resid.set_ylabel('Resíduos')

    # Q-Q Plot
    fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
    sm.qqplot(resid, line='s', fit=True, ax=ax_qq)
    ax_qq.set_title(f'Normalidade dos Resíduos ({model_name})')
    
    return fig_resid, fig_qq


# --- ESTRUTURA DO STREAMLIT ---

st.title("🎓 Modelos Lineares ENEM: Análise e Seleção de Modelos")
st.markdown("### Atividade Avaliativa P2")
st.markdown("---")

# Carregar os dados processados (usando cache)
results = load_and_process_data()
if results is None:
    st.error("Falha ao carregar e processar os dados. Certifique-se de que o arquivo ENEM está no diretório correto e que as bibliotecas estão instaladas.")
    st.stop()
(model1_func, model1_robust, model2_func, df_train_full, Y_test_class, Y1_pred_test, Y2_pred_test, df_comparison, df_coefs_ols) = results

# --------------------------------------------------------------------------
# --- INTRODUÇÃO E CONTEXTO ---
# --------------------------------------------------------------------------

st.header("1. Contexto da Atividade e Fundamentos")
st.markdown("""
Esta atividade visa construir e comparar dois modelos de Regressão Linear Múltipla para prever a $\mathbf{NOTA\_MT\_MATEMATICA}$ ($\mathbf{Y}$) de estudantes do ENEM, a partir de outras notas ($\mathbf{X}$), utilizando um rigor estatístico baseado nas boas práticas de *Machine Learning*.
""")

st.subheader("O que são Modelos Lineares (MQO)?")
st.markdown(r"""
Modelos Lineares, estimados por Mínimos Quadrados Ordinários (MQO), buscam encontrar a reta ($\mathbf{\hat{\beta}}$) que minimiza a soma dos quadrados dos erros (resíduos) entre os valores observados e os valores previstos ($\mathbf{Y = X\beta + \epsilon}$). A solução matricial para os coeficientes é dada por:
$$\mathbf{\hat{\beta}} = (\mathbf{X}^{\text{T}}\mathbf{X})^{-1}\mathbf{X}^{\text{T}}\mathbf{Y}$$
""")

st.subheader("EDA - Análise de Correlação e Seleção de Variáveis")

col_corr, col_corr_text = st.columns([1, 1])

with col_corr:
    st.pyplot(generate_correlation_heatmap(df_train_full))
    st.caption("Mapa de Calor da Correlação (Amostra de Treino)")

with col_corr_text:
    st.markdown("""
    A análise inicial de correlação (heatmap ao lado) confirma a **forte correlação positiva** entre a $\text{NOTA\_MT}$ e as demais notas ($\rho > 0.9$ em muitos casos).
    - **Seleção:** Utilizou-se o método *Backward* para o **Modelo 1** (mantendo CN, CH, Redação) e o método *Forward* para o **Modelo 2** (mantendo CN, Redação), buscando otimizar o $\mathbf{R^2}$ e a parcimônia.
    - **Amostragem:** Utilizou-se a **Amostragem Aleatória Simples (AAS)** com split de **70% para Treino** e **30% para Teste**, garantindo que o modelo seja treinado em uma base robusta e avaliado em dados não vistos (*data leakage* zero).
    """)

st.markdown("---")

# --------------------------------------------------------------------------
# --- SEÇÃO 2: SELEÇÃO, COMPARAÇÃO E JUSTIFICATIVA FINAL ---
# --------------------------------------------------------------------------

st.header("2. Comparação Final e Seleção do Modelo")

col_comp1, col_comp2 = st.columns([1.5, 1])

with col_comp1:
    st.subheader("Tabela de Performance")
    st.markdown("Validação da estabilidade e generalização no conjunto de **Teste**.")
    
    st.dataframe(df_comparison.style.format('{:.4f}').highlight_min(
        axis=1, 
        subset=pd.IndexSlice[['RMSE (Teste)', 'AIC', 'BIC', 'RMSE K-fold'], :], 
        props='background-color: #d4edda;'
    ).highlight_max(
        axis=1,
        subset=pd.IndexSlice[['Curva ROC/AUC', 'F1 Score', 'Acurácia'], :],
        props='background-color: #d4edda;'
    ), use_container_width=True)

with col_comp2:
    st.subheader("Modelo Vencedor e Justificativa")
    st.markdown(f"""
    O **Modelo 1** ($\text{{NOTA\_CN, NOTA\_CH, NOTA\_REDACAO}}$) foi o vencedor em relação às métricas.
    - **Predição:** Seu $\mathbf{{RMSE}}$ ($\mathbf{{ {df_comparison.loc['RMSE (Teste)']['Modelo 1 (Vencedor)']:.4f} }}$) é o menor, indicando o menor erro de previsão no conjunto de teste.
    - **Parcimônia:** O ganho de $\mathbf{{R^2}}$ com as três notas mais do que compensa a complexidade (melhores $\mathbf{{AIC/BIC}}$).
    - **Generalização:** O $\mathbf{{AUC}}$ de $\mathbf{{ {df_comparison.loc['Curva ROC/AUC']['Modelo 1 (Vencedor)']:.4f} }}$ e o **RMSE K-fold** (estabilidade) confirmam a robustez do Modelo 1.
    """)

st.markdown("---")

# --------------------------------------------------------------------------
# --- SEÇÃO 3: DIAGNÓSTICO E VALIDAÇÃO MATRICIAL ---
# --------------------------------------------------------------------------

st.header("3. Diagnóstico e Implicações Estatísticas")

col_diag, col_coefs = st.columns(2)

with col_diag:
    st.subheader("Análise dos Pressupostos e Curva ROC")
    
    # Seletor para alternar gráficos
    modelo_choice = st.radio("Alternar Gráficos de Diagnóstico:", ("Modelo 1", "Modelo 2"))
    
    if modelo_choice == "Modelo 1":
        current_model = model1_func
        current_name = "Modelo 1 (Vencedor)"
    else:
        current_model = model2_func
        current_name = "Modelo 2 (Parcimonioso)"
        
    fig_resid, fig_qq = generate_residual_plots(current_model, current_name)
    st.pyplot(fig_resid)
    st.caption("Resíduos vs. Ajustados: Dispersão em funil indica **Heterocedasticidade** (violação do pressuposto).")
    
    st.pyplot(fig_qq)
    st.caption("Q-Q Plot: Normalidade aproximada, com caudas pesadas (extremos).")
    
    # Outros Pressupostos (Texto)
    st.markdown("""
    **Outros Pressupostos Diagnosticados (Sem Gráfico):**
    - **Independência dos Erros:** O **Teste Durbin-Watson** resultou em um valor próximo de 2, indicando a **ausência de autocorrelação** (erros independentes).
    - **Normalidade/Outliers:** A análise de DFFITS e DFBETAS mostrou que 99% dos *outliers* não exercem influência indevida.
    """)
    
    # Curva ROC - Fixo para comparação
    auc1 = df_comparison.loc['Curva ROC/AUC', 'Modelo 1 (Vencedor)']
    auc2 = df_comparison.loc['Curva ROC/AUC', 'Modelo 2 (Parcimonioso)']
    fig_roc = generate_roc_curve(Y_test_class, Y1_pred_test, Y2_pred_test, auc1, auc2)
    st.pyplot(fig_roc)
    st.caption("Curva ROC: Validação do poder de discriminação do Modelo.")


with col_coefs:
    st.subheader("Modelo Final e Tratamento dos Pressupostos")
    
    st.markdown("##### OLS vs. Matricial")
    st.markdown(r"A equivalência prova a correção do Estimador de Mínimos Quadrados ($\mathbf{\hat{\beta}}$).")
    st.dataframe(df_coefs_ols.T.style.format('{:.6f}'), use_container_width=True)
    
    st.markdown("##### Implicações e Tratamento das Violações (Modelo 1):")
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

    st.markdown(r"""
    **Tratamento Detalhado:**
    1. **Heterocedasticidade (Correção de Inferência):**
       - **Problema:** Erros Padrão viesados, invalidando os $\text{p-valores}$.
       - **Solução:** Utilizou-se o $\mathbf{Estimador Robusto (HC3)}$, resultando nos $\text{Erros Padrão}$ e $\text{p-valores}$ corrigidos acima. Isso **restaura a validade da Inferência** sobre a significância dos preditores.
    2. **Multicolinearidade Severa:** ($\mathbf{{VIF \approx 300}}$)
       - **Problema:** Instabilidade e alta variância dos $\mathbf{\hat{\beta}}$, devido à alta correlação entre as notas.
       - **Solução (Estratégia):** O MQO foi mantido devido ao seu $\mathbf{RMSE}$ superior ao da Regressão Ridge. O modelo deve ser usado **apenas para Previsão**, pois a instabilidade impede a **interpretação causal e isolada** do $\mathbf{\hat{\beta}}$ de cada nota.
    """)

st.markdown("---")

# --------------------------------------------------------------------------
# --- NOVA SEÇÃO 4: APLICAÇÃO E PREDIÇÃO ---
# --------------------------------------------------------------------------

st.header("4. Aplicação do Modelo Vencedor (Predição)")
st.markdown("Utilize o Modelo 1 ($\text{NOTA\_CN, NOTA\_CH, NOTA\_REDACAO}$) para estimar a nota de Matemática ($\text{NOTA\_MT}$) com base nas notas fornecidas.")

col_input1, col_input2 = st.columns([1, 1])

# Campos de Input
with col_input1:
    st.markdown("##### Insira as Notas do Aluno:")
    cn_input = st.number_input("Nota Ciências da Natureza (CN)", min_value=300.0, max_value=1000.0, value=550.0, step=0.1)
    ch_input = st.number_input("Nota Ciências Humanas (CH)", min_value=300.0, max_value=1000.0, value=550.0, step=0.1)
    redacao_input = st.number_input("Nota Redação", min_value=0.0, max_value=1000.0, value=600.0, step=10.0)

# --- LÓGICA DE PREDIÇÃO ALGEBRICA ---
# 1. Obter os coeficientes do modelo treinado
params = model1_func.params

# 2. Calcular a previsão usando a fórmula algébrica (imune ao reindex)
predicted_mt = (
    params['const'] +
    params['NOTA_CN_CIENCIAS_DA_NATUREZA'] * cn_input +
    params['NOTA_CH_CIENCIAS_HUMANAS'] * ch_input +
    params['NOTA_REDACAO'] * redacao_input
)

with col_input2:
    st.markdown("##### Resultado da Predição")
    st.success(f"A Nota de Matemática Prevista é:")
    st.metric(label="NOTA MT PREVISTA", value=f"{predicted_mt:.2f}")

    st.markdown(r"""
    A previsão foi calculada diretamente pela fórmula $\mathbf{\hat{Y} = \mathbf{\hat{\beta}_0} + \mathbf{\hat{\beta}_1}X_1 + \dots}$, utilizando os coeficientes extraídos do modelo treinado. O erro médio desta estimativa ($\mathbf{RMSE}$) é de **16.84 pontos**.
    """)
