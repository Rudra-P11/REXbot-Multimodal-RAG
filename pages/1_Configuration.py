import streamlit as st

st.title("🪄 System Configuration")

st.markdown("""
Please enter your API keys below. These are required for the system to function.
- **Gemini API Key**: Get it from [Google AI Studio](https://aistudio.google.com/).
- **Qdrant Credentials**: Get them from [Qdrant Cloud](https://cloud.qdrant.io/).
""")

with st.form("config_form"):
    gemini_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password")
    qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url, placeholder="https://xxx.qdrant.tech")
    qdrant_key = st.text_input("Qdrant API Key", value=st.session_state.qdrant_api_key, type="password")
    
    submitted = st.form_submit_button("Save Configuration")
    
    if submitted:
        st.session_state.gemini_api_key = gemini_key
        st.session_state.qdrant_url = qdrant_url
        st.session_state.qdrant_api_key = qdrant_key
        st.success("Configuration saved successfully!")
        st.info("Navigate to the 'Knowledge Base' page to upload documents.")
