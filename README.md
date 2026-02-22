# modelos_lineares_p2

# 🎓 Análise de Modelos Lineares: Predição de Desempenho no ENEM 2024

Este repositório contém uma aplicação desenvolvida em **Streamlit** para análise, comparação e aplicação de modelos de **Regressão Linear Múltipla (MQO)**, utilizando uma amostra de 5% dos microdados do ENEM 2024.

O objetivo principal é prever a nota de **Matemática (MT)** com base no desempenho do candidato em outras áreas do conhecimento (Ciências da Natureza, Ciências Humanas e Redação).

## 🚀 Funcionalidades

* **Processamento de Dados:** Limpeza automática e divisão treino/teste (70/30).
* **Análise Exploratória (EDA):** Heatmap de correlação para identificação de multicolinearidade.
* **Comparação de Modelos:**
* **Modelo 1 (Vencedor):** Utiliza CN, CH e Redação (Seleção via *Backward*).
* **Modelo 2 (Parcimonioso):** Utiliza apenas CN e Redação (Seleção via *Forward*).


* **Diagnóstico Estatístico:** * Análise de Resíduos (Homocedasticidade e Normalidade).
* Cálculo de VIF (Variance Inflation Factor).
* Validação Matricial ().


* **Métricas de Performance:** RMSE, AIC, BIC, Cross-Validation (K-Fold) e Curva ROC/AUC.
* **Calculadora de Predição:** Interface interativa para estimar a nota de Matemática em tempo real.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Bibliotecas de Dados:** `pandas`, `numpy`, `scikit-learn`
* **Modelagem Estatística:** `statsmodels`
* **Visualização:** `matplotlib`, `seaborn`
* **Interface Web:** `streamlit`

## 📊 Metodologia e Insights

### Diagnóstico de Pressupostos

O projeto aborda criticamente as violações dos pressupostos clássicos de Gauss-Markov:

1. **Heterocedasticidade:** Identificada via análise visual de resíduos e tratada com o uso de **Erros Padrão Robustos (HC3)** para garantir a validade da inferência.
2. **Multicolinearidade:** Detectada através de um VIF elevado, indicando que as notas são altamente correlacionadas. Por conta disso, o modelo é recomendado para **previsão** e não para interpretação causal isolada.
3. **Performance:** O Modelo 1 apresentou o menor RMSE e melhores índices de AIC/BIC, sendo escolhido para o deploy na ferramenta de predição.
