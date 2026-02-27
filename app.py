import streamlit as st
import pandas as pd
import os
import tempfile
from core.logic import process_project

# Configuração da página
st.set_page_config(
    page_title="Validador Estrutural DXF",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS REFORÇADO (MANTIDO) ---
st.markdown("""
    <style>
    /* Estilo Base do Card */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        color: #000000;
    }

    /* Força a cor do texto */
    div[data-testid="stMetric"] label { color: #555 !important; font-weight: 600; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #000 !important; }

    /* Lógica de Cores para as Colunas */
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(1) div[data-testid="stMetric"] {
        border-left: 10px solid #3b82f6 !important; /* Azul */
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) div[data-testid="stMetric"] {
        border-left: 10px solid #22c55e !important; /* Verde */
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(3) div[data-testid="stMetric"] {
        border-left: 10px solid #ef4444 !important; /* Vermelho */
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÃO DE PROCESSAMENTO COM CACHE ---
@st.cache_data(show_spinner=False)
def processar_arquivo_cached(uploaded_file_content, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_file:
        tmp_file.write(uploaded_file_content)
        tmp_path = tmp_file.name
    
    try:
        return process_project(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Entrada de Dados")
    # ALTERAÇÃO 1: accept_multiple_files=True
    uploaded_files = st.file_uploader(
        "Upload de arquivos DXF (v2010/2013)", 
        type=["dxf"], 
        accept_multiple_files=True
    )
    
    st.divider()
    
    st.header("ℹ️ Instruções")
    st.markdown("""
    **Multi-arquivos:**
    Você pode selecionar vários arquivos DXF de uma vez. Cada um será aberto em uma aba separada.
    
    **Passo a passo:**
    1. Salve seus projetos em versão **DXF 2010 ou 2013**.
    2. Garanta que as camadas de armadura estejam visíveis.
    3. Arraste todos os arquivos para a área de upload.
    """)
    st.caption("Versão 1.4.0 (Multi-File)")

# --- ÁREA PRINCIPAL ---
st.title("🏗️ Validador Estrutural Automatizado")

if not uploaded_files:
    st.warning("👈 Por favor, faça o upload de um ou mais arquivos DXF na barra lateral.")

else:
    # Cria uma lista com os nomes para as abas
    tab_names = [f.name for f in uploaded_files]
    
    # ALTERAÇÃO 2: Criação dinâmica das abas
    tabs = st.tabs(tab_names)

    # Itera sobre as abas e os arquivos simultaneamente
    for tab, uploaded_file in zip(tabs, uploaded_files):
        with tab:
            st.subheader(f"Arquivo: {uploaded_file.name}")
            
            with st.spinner(f"Processando {uploaded_file.name}..."):
                try:
                    df_resultado, _, msg, _ = processar_arquivo_cached(uploaded_file.getvalue(), uploaded_file.name)
                    
                    if "Erro" in msg and not isinstance(df_resultado, pd.DataFrame):
                        st.error(msg)
                    else:
                        # --- LÓGICA DE EXIBIÇÃO POR ARQUIVO ---
                        if isinstance(df_resultado, pd.DataFrame) and not df_resultado.empty:
                            
                            if 'status' not in df_resultado.columns:
                                df_resultado['status'] = 'Desconhecido'

                            erros = df_resultado[df_resultado['status'] == '❌ ERRO']
                            qtd_erros = len(erros)
                            
                            # KPIs
                            kpi1, kpi2, kpi3 = st.columns(3)
                            kpi1.metric("Total Analisado", len(df_resultado))
                            kpi2.metric("Itens Corretos", len(df_resultado) - qtd_erros)
                            kpi3.metric("Divergências", qtd_erros)

                            st.divider()

                            col_filtros, col_download = st.columns([3, 1])
                            
                            with col_filtros:
                                st.write("📊 **Relatório de Validação**")
                                # ALTERAÇÃO 3: key única usando o nome do arquivo
                                filtro = st.radio(
                                    "Filtro:", 
                                    ["Mostrar Apenas Divergências", "Mostrar Tudo"], 
                                    horizontal=True, 
                                    label_visibility="collapsed",
                                    key=f"radio_{uploaded_file.name}" 
                                )
                            
                            df_show = erros if filtro == "Mostrar Apenas Divergências" else df_resultado

                            def highlight_status(val):
                                color = '#ffcdd2' if val == '❌ ERRO' else '#c8e6c9'
                                return f'background-color: {color}'

                            st.data_editor(
                                df_show.style.map(highlight_status, subset=['status']),
                                use_container_width=True,
                                column_config={
                                    "status": st.column_config.TextColumn("Status"),
                                },
                                disabled=True,
                                height=400,
                                key=f"editor_{uploaded_file.name}" # Key única
                            )

                            with col_download:
                                st.write("") 
                                st.write("") 
                                csv = df_show.to_csv(index=False).encode('utf-8-sig')
                                st.download_button(
                                    label="📥 Baixar CSV",
                                    data=csv,
                                    file_name=f"relatorio_{uploaded_file.name}.csv",
                                    mime="text/csv",
                                    type="primary",
                                    use_container_width=True,
                                    key=f"btn_{uploaded_file.name}" # Key única
                                )
                        else:
                            st.warning("Arquivo processado, mas nenhum dado encontrado.")

                except Exception as e:
                    st.error(f"Erro ao processar este arquivo: {e}")