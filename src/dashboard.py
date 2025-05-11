import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import datetime

# Configurações da página com tema aprimorado
st.set_page_config(
    page_title="🍦 Gelato Mágico - Previsão de Vendas",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/gisengsoft/MLVendasLab',
        'About': 'Sistema de previsão de vendas baseado em Machine Learning'
    }
)

# CSS personalizado para melhorar a aparência
st.markdown("""
<style>
    /* Estilo geral */
    .main {
        background-color: #F5F7F9;
    }
    /* Botões mais atrativos */
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    /* Cards para métricas */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    /* Cores para títulos */
    h1 {
        color: #1E3A8A;
    }
    h3 {
        color: #2563EB;
    }
    /* Tabelas mais bonitas */
    div[data-testid="stTable"] {
        border-radius: 5px;
        overflow: hidden;
    }
    /* Container estilizado */
    .css-12w0qpk {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    /* Status do modelo */
    .status-box {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .status-ok {
        background-color: #d1fae5;
        color: #059669;
    }
    .status-error {
        background-color: #fee2e2;
        color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado da sessão para histórico
if 'historico' not in st.session_state:
    st.session_state.historico = []

# Definir caminho absoluto para o modelo
MODELO_PATH = r"C:\MLVendasLab\outputs\modelo_final.joblib"

# Função para criar um termômetro visual
def criar_termometro(temperatura, min_temp=20, max_temp=37):
    pct = (temperatura - min_temp) / (max_temp - min_temp) * 100
    cor = '#5DA5DA' if temperatura < 25 else '#FAA43A' if temperatura < 32 else '#F15854'
    
    html = f"""
    <div style="margin:10px 0;">
        <div style="margin-bottom:5px;"><b>Termômetro</b></div>
        <div style="width:100%; background-color:#ddd; border-radius:5px;">
            <div style="width:{pct}%; height:24px; background-color:{cor}; 
                border-radius:5px; text-align:center; line-height:24px; color:white; font-weight:bold;">
                {temperatura}°C
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:3px;">
            <span>Frio (20°C)</span>
            <span>Moderado</span>
            <span>Quente (37°C)</span>
        </div>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)

# Função para converter DataFrame para CSV (para download)
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Carregar o modelo
@st.cache_resource
def carregar_modelo():
    try:
        if os.path.exists(MODELO_PATH):
            modelo = joblib.load(MODELO_PATH)
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
st.markdown("### Sistema de previsão baseado em temperatura")

# Sidebar com informações
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/ice-cream-cone.png")
    
    st.markdown("### Sobre")
    st.info("""
    Este sistema utiliza Machine Learning para prever vendas de sorvete com base na temperatura.
    
    Desenvolvido por: Gelato Mágico
    """)
    
    # Informação sobre o modelo
    st.markdown("### Informações do Modelo")
    if modelo_carregado:
        st.markdown('<div class="status-box status-ok">✅ Modelo carregado</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-error">❌ Modelo não disponível</div>', unsafe_allow_html=True)
        st.info(f"Procurando em: {MODELO_PATH}")
    
    # Data e hora atual
    now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    st.markdown(f"**Última atualização:** {now}")

# Layout principal com duas colunas
col1, col2 = st.columns([1, 2])

# Coluna 1 - Área de previsão
with col1:
    with st.container():
        st.markdown("### Faça uma previsão")
        
        # Controle de temperatura com termômetro visual
        temperatura = st.slider(
            'Selecione a temperatura (°C):', 
            min_value=20.0, 
            max_value=37.0, 
            value=27.0, 
            step=0.5
        )
        
        # Mostrar termômetro visual
        criar_termometro(temperatura)
        
        # Botão de previsão
        prever_clicked = st.button('Fazer Previsão', type="primary")
        
        # Processar a previsão quando o botão for clicado
        if prever_clicked and modelo_carregado:
            try:
                previsao = int(modelo.predict([[temperatura]])[0])
                
                # Adicionar ao histórico
                st.session_state.historico.append({
                    'temperatura': temperatura,
                    'previsao': previsao,
                    'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
                })
                
                # Mostrar cards com métricas
                st.success(f'Previsão calculada com sucesso!')
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Temperatura", f"{temperatura}°C")
                with col_b:
                    st.metric("Vendas Previstas", f"{previsao}")
                
                # Recomendações baseadas na previsão
                st.subheader("Recomendação:")
                if previsao > 150:
                    st.info("📈 **Alta demanda esperada!** Aumente o estoque e a equipe.")
                elif previsao > 100:
                    st.info("🔄 **Demanda normal.** Mantenha estoque e equipe regulares.")
                else:
                    st.info("📉 **Baixa demanda esperada.** Considere promoções para atrair clientes.")
                
            except Exception as e:
                st.error(f"Erro ao fazer previsão: {e}")
        elif prever_clicked:
            st.error("Modelo não carregado. Não é possível fazer previsões.")

# Coluna 2 - Gráfico e visualização
with col2:
    st.markdown("### Relação Temperatura x Vendas")
    
    if modelo_carregado:
        try:
            # Gerar dados para o gráfico
            temperaturas = np.arange(20, 38, 0.5).reshape(-1, 1)
            previsoes = modelo.predict(temperaturas)
            
            # Criar figura
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sombreamento para faixas de temperatura
            ax.axvspan(20, 25, alpha=0.1, color='blue', label='Frio')
            ax.axvspan(25, 32, alpha=0.1, color='orange', label='Moderado')
            ax.axvspan(32, 38, alpha=0.1, color='red', label='Quente')
            
            # Linha de previsão
            ax.plot(temperaturas, previsoes, 'r-', linewidth=2.5, label='Modelo de previsão')
            
            # Marcar previsão atual
            if prever_clicked and 'previsao' in locals():
                ax.scatter([temperatura], [previsao], color='blue', s=120, zorder=5, label='Previsão atual')
                
            # Estilizar o gráfico
            ax.set_title('Relação entre Temperatura e Vendas Previstas', fontsize=14)
            ax.set_xlabel('Temperatura (°C)', fontsize=12)
            ax.set_ylabel('Vendas Previstas (unidades)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend()
            
            # Adicionar anotações para faixas
            ax.text(22.5, np.max(previsoes)*0.9, "Frio", fontsize=9, 
                   ha='center', va='center', alpha=0.7, color='blue')
            ax.text(28.5, np.max(previsoes)*0.9, "Moderado", fontsize=9,
                   ha='center', va='center', alpha=0.7, color='darkorange')
            ax.text(35, np.max(previsoes)*0.9, "Quente", fontsize=9, 
                   ha='center', va='center', alpha=0.7, color='red')
            
            # Exibir o gráfico no Streamlit
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")
    else:
        st.warning("Gráfico não disponível: modelo não carregado.")

# Tabela de previsões e histórico
st.markdown("---")
col_tab1, col_tab2 = st.columns(2)

# Tabela de referência
with col_tab1:
    if modelo_carregado:
        try:
            st.markdown("### Tabela de Referência")
            temperaturas_ref = list(range(20, 38, 2))
            previsoes_ref = [int(modelo.predict([[t]])[0]) for t in temperaturas_ref]

            df_ref = pd.DataFrame({
                "Temperatura (°C)": temperaturas_ref,
                "Previsão de Vendas": previsoes_ref
            })

            # Tabela interativa
            st.dataframe(df_ref, use_container_width=True)
            
            # Botão para download
            csv = convert_df(df_ref)
            st.download_button(
                "📥 Baixar Tabela CSV",
                csv,
                "previsao_vendas_sorvete.csv",
                "text/csv",
                key='download-csv'
            )
        except Exception as e:
            st.error(f"Erro ao gerar tabela de referência: {e}")
    else:
        st.warning("Tabela de referência não disponível: modelo não carregado.")

# Histórico de previsões
with col_tab2:
    st.markdown("### Histórico de Previsões")
    if st.session_state.historico:
        hist_df = pd.DataFrame(st.session_state.historico)
        st.dataframe(hist_df.style.format({"temperatura": "{:.1f}°C"}), use_container_width=True)
        
        if st.button("Limpar Histórico"):
            st.session_state.historico = []
            st.experimental_rerun()
    else:
        st.info("Nenhuma previsão realizada nesta sessão.")

# Instruções de uso
with st.expander("Como usar este dashboard", expanded=False):
    st.markdown("""
    ### Instruções de Uso:
    
    1. **Selecione a temperatura** usando o controle deslizante
    2. Clique no botão **Fazer Previsão** para calcular a previsão de vendas
    3. Veja o resultado e as recomendações baseadas na previsão
    4. Consulte o gráfico para entender a relação entre temperatura e vendas
    5. Use a tabela de referência para valores comuns
    6. Visualize seu histórico de previsões nesta sessão
    
    Para melhores resultados, use temperaturas entre 20°C e 37°C.
    """)

# Informações de debug (opcional)
with st.expander("Informações de Debug", expanded=False):
    st.subheader("Informações de Debug")
    st.write(f"Caminho do modelo: {MODELO_PATH}")
    st.write(f"Arquivo existe? {'Sim' if os.path.exists(MODELO_PATH) else 'Não'}")
    st.write(f"Diretório atual: {os.getcwd()}")
    
    if st.checkbox("Mostrar informações do sistema"):
        st.write(f"Python: {sys.version}")
        st.write(f"Streamlit: {st.__version__}")
        if 'numpy' in sys.modules:
            st.write(f"NumPy: {np.__version__}")
        if 'pandas' in sys.modules:
            st.write(f"Pandas: {pd.__version__}")