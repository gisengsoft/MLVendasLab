import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configurações da página
st.set_page_config(
    page_title="🍦 Gelato Mágico - Previsão de Vendas",
    page_icon="🍦",
    layout="wide"
)

# Definir caminho absoluto para o modelo
MODELO_PATH = r"C:\MLVendasLab\outputs\modelo_final.joblib"

# Carregar o modelo
@st.cache_resource
def carregar_modelo():
    try:
        if os.path.exists(MODELO_PATH):
            modelo = joblib.load(MODELO_PATH)
            st.success("Modelo carregado com sucesso!")
            return modelo
        else:
            st.error(f"Arquivo não encontrado: {MODELO_PATH}")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Tentar carregar o modelo
try:
    modelo = carregar_modelo()
    modelo_carregado = modelo is not None
except Exception as e:
    st.error(f"Falha ao carregar o modelo: {e}")
    modelo_carregado = False

# Título e descrição
st.title('🍦 Previsão de Vendas - Gelato Mágico')
st.subheader('Sistema de previsão baseado em temperatura')

# Sidebar com informações
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/ice-cream-cone.png")
    st.subheader("Sobre")
    st.info("""
    Este sistema utiliza Machine Learning para prever vendas de sorvete com base na temperatura.
    
    Desenvolvido por: Gelato Mágico
    """)
    
    # Informação sobre o modelo
    st.subheader("Informações do Modelo")
    if modelo_carregado:
        st.success("✅ Modelo carregado")
    else:
        st.error("❌ Modelo não disponível")
        st.info(f"Procurando em: {MODELO_PATH}")

# Layout principal
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Faça uma previsão")
    temperatura = st.slider(
        'Selecione a temperatura (°C):', 
        min_value=20.0, 
        max_value=37.0, 
        value=30.0, 
        step=0.5
    )
    
    if st.button('Fazer Previsão', type="primary"):
        if modelo_carregado:
            try:
                previsao = int(modelo.predict([[temperatura]])[0])
                
                st.success(f'Para uma temperatura de {temperatura}°C, '
                          f'a previsão de vendas é de **{previsao} sorvetes**.')
                
                # Mostrar recomendações
                if previsao > 150:
                    st.info("📊 Recomendação: Considere aumentar o estoque e a equipe.")
                elif previsao > 100:
                    st.info("📊 Recomendação: Estoque e equipe normais.")
                else:
                    st.info("📊 Recomendação: Considere promoções para atrair clientes.")
            except Exception as e:
                st.error(f"Erro ao fazer previsão: {e}")
        else:
            st.error("Modelo não carregado. Não é possível fazer previsões.")

with col2:
    st.subheader("Relação Temperatura x Vendas")
    
    if modelo_carregado:
        try:
            # Gerar dados para o gráfico
            temperaturas = np.arange(20, 38, 0.5).reshape(-1, 1)
            previsoes = modelo.predict(temperaturas)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(temperaturas, previsoes, 'r-', linewidth=2)
            
            if 'previsao' in locals():
                ax.scatter([temperatura], [previsao], color='blue', s=100, label='Previsão atual')
                
            ax.set_title('Relação entre Temperatura e Vendas Previstas')
            ax.set_xlabel('Temperatura (°C)')
            ax.set_ylabel('Vendas Previstas (unidades)')
            ax.grid(True, alpha=0.3)
            if 'previsao' in locals():
                ax.legend()
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")
    else:
        st.warning("Gráfico não disponível: modelo não carregado.")

# Tabela de previsões
if modelo_carregado:
    try:
        st.subheader("Tabela de Referência")
        temperaturas_ref = list(range(20, 38, 2))
        previsoes_ref = [int(modelo.predict([[t]])[0]) for t in temperaturas_ref]

        df_ref = pd.DataFrame({
            "Temperatura (°C)": temperaturas_ref,
            "Previsão de Vendas": previsoes_ref
        })

        st.table(df_ref)
    except Exception as e:
        st.error(f"Erro ao gerar tabela de referência: {e}")
else:
    st.warning("Tabela de referência não disponível: modelo não carregado.")

# Instruções adicionais
st.markdown("---")
st.subheader("Como usar:")
st.markdown("""
1. Use o controle deslizante para selecionar a temperatura desejada
2. Clique no botão "Fazer Previsão"
3. Veja o resultado e as recomendações baseadas na previsão
""")

# Debug info (apenas em desenvolvimento)
if st.checkbox("Mostrar informações de debug"):
    st.subheader("Informações de Debug")
    st.write(f"Caminho do modelo: {MODELO_PATH}")
    st.write(f"Arquivo existe? {'Sim' if os.path.exists(MODELO_PATH) else 'Não'}")
    st.write(f"Diretório atual: {os.getcwd()}")