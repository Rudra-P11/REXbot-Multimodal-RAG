import streamlit as st

st.set_page_config(layout="wide", page_title="REXchat RAG System")

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""
if "qdrant_api_key" not in st.session_state:
    st.session_state.qdrant_api_key = ""
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

pages = [
    st.Page("pages/1_Configuration.py", title="Configuration", icon="⚙️"),
    st.Page("pages/2_Knowledge_Base.py", title="Knowledge Base", icon="📚"),
    st.Page("pages/3_AI_Chatbot.py", title="AI Chatbot", icon="🤖"),
]

pg = st.navigation(pages)

st.sidebar.title("Navigation")
st.sidebar.info("Select a page to proceed.")

pg.run()
