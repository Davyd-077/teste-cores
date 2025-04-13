import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

# Arquivo onde o histórico será salvo
ARQUIVO = 'historico_cores.csv'

# Cores e pesos
peso_cor = {'preto': 2, 'vermelho': 2, 'branco': 14}
cores_validas = list(peso_cor.keys())

# Inicializa ou carrega o histórico
if not os.path.exists(ARQUIVO):
    pd.DataFrame(columns=["cor"]).to_csv(ARQUIVO, index=False)

# Carrega o histórico
df = pd.read_csv(ARQUIVO)

# Título da interface
st.title("IA: Previsão de Cor com Aprendizado Contínuo")
st.write("Insira cores separadas por vírgula (ex: preto,vermelho,branco). A IA vai prever a próxima com base nas últimas 10.")

# Caixa de entrada
entrada = st.text_input("Insira cores:")

if st.button("Adicionar e Prever"):
    cores_input = [c.strip().lower() for c in entrada.split(",") if c.strip().lower() in cores_validas]

    if not cores_input:
        st.warning("Nenhuma cor válida foi inserida.")
    else:
        # Adiciona ao histórico
        novas_cores = pd.DataFrame(cores_input, columns=["cor"])
        df = pd.concat([df, novas_cores], ignore_index=True)
        df.to_csv(ARQUIVO, index=False)
        st.success("Cores adicionadas com sucesso!")

        # Limpa o campo de texto (simulando com um placeholder)
        st.experimental_rerun()

# Exibe o histórico
st.subheader("Histórico de Cores")
st.dataframe(df.tail(50), use_container_width=True)

# Previsão
if len(df) >= 11:
    # Codifica as cores
    le = LabelEncoder()
    df["cor_codificada"] = le.fit_transform(df["cor"])

    X, y = [], []

    for i in range(len(df) - 10):
        entrada_seq = df["cor_codificada"].iloc[i:i+10].tolist()
        saida = df["cor_codificada"].iloc[i+10]
        X.append(entrada_seq)
        y.append(saida)

    # Calcula pesos
    y_cores = le.inverse_transform(y)
    pesos = [peso_cor[c] for c in y_cores]

    # Treinamento
    modelo = SGDClassifier(max_iter=1000, tol=1e-3)
    modelo.fit(X, y, sample_weight=pesos)

    ultimos_10 = df["cor_codificada"].iloc[-10:].tolist()
    pred = modelo.predict([ultimos_10])[0]
    cor_prevista = le.inverse_transform([pred])[0]

    st.subheader("Próxima Cor Prevista:")
    st.success(f"**{cor_prevista.upper()}**")

else:
    st.info("Insira pelo menos 11 cores para que a IA possa começar a prever.")
