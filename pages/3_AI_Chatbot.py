import streamlit as st
from utils.gemini_handler import GeminiHandler
from utils.qdrant_handler import QdrantHandler
import time

st.title("🌟 AI Chatbot")

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""
if "qdrant_api_key" not in st.session_state:
    st.session_state.qdrant_api_key = ""

if not st.session_state.get("gemini_api_key") or not st.session_state.get("qdrant_api_key"):
    st.warning("⚠️ Please configure your API keys in the Configuration page first.")
    st.stop()

gemini = GeminiHandler(st.session_state.gemini_api_key)
qdrant = QdrantHandler(st.session_state.qdrant_url, st.session_state.qdrant_api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("⚙️ RAG Settings")
    advanced_mode = st.toggle("Advanced RAG (HyDE & Anti-Hallucination)", value=True, help="Uses Hypothetical Document Embeddings and strict context filtering to maximize precision.")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            
            search_query_text = prompt
            
            # **ADVANCED RAG: HyDE (Hypothetical Document Embeddings)**
            if advanced_mode:
                hyde_prompt = f"Please write a passage to answer the question: {prompt}. Write it as if it's found in a factual document."
                hypothetical_answer = gemini.generate_response(
                    system_instruction="You are a helpful expert. Generate a plausible, factual-sounding document snippet that answers the user's question.",
                    messages=[{"role": "user", "content": hyde_prompt}]
                )
                if not "Error" in hypothetical_answer:
                    search_query_text = hypothetical_answer
                    st.toast("🧠 Using HyDE for semantic retrieval...")
            
            query_vector = gemini.get_query_embedding(search_query_text)
            
            top_k_fetch = 8 if advanced_mode else 5
            search_results = qdrant.hybrid_search(query_vector, search_query_text, top_k=top_k_fetch)
            
            context_text = ""
            citations = []
            for res in search_results:
                payload = res.payload
                text = payload.get('text', '')
                source = payload.get('source', 'Unknown')
                page = payload.get('page', 'N/A')
                score = getattr(res, 'score', 0.0)
                
                context_text += f"---\nSource: {source} (Page {page})\nContent: {text}\n"
                citations.append(f"📄 {source} (p.{page}) - Score: {score:.2f}")

            if advanced_mode:
                system_prompt = f"""You are a strict, factual AI assistant. 
                Using ONLY the provided Context, answer the user's question comprehensively.
                If the answer cannot be determined directly from the Context, you MUST declare: "I don't have enough information to answer that based on the provided documents." Do not guess or hallucinate.
                Always cite the source name and page number for EVERY claim you make.
                
                Context:
                {context_text}
                """
            else:
                system_prompt = f"""You are REX, a helpful AI assistant. Use the following context to answer the user's question. 
                If the answer is not in the context, say you don't know. 
                Always cite the source page number if available.
                example: "The capital of France is Paris. [Source: Document1, p.5]"
                Context:
                {context_text}
                """
            
            # Prepare messages for Gemini (History + Current Prompt with Context)
            # We append context to the system prompt or the last message.
            # Using system prompt is cleaner for RAG.
            
            # API Call
            try:
                # The new GeminiHandler.generate_response handles the message format robustly
                # We pass the history (st.session_state.messages) and the prompt with context
                
                # We want to send history but the last message should have the context
                # So we pass history[:-1] and a custom last message
                history = st.session_state.messages[:-1]
                current_message = {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"}
                
                # We can call with full history list if we want, or just current. 
                # For RAG, history + context is better.
                api_messages = history + [current_message]

                response_text = gemini.generate_response(system_prompt, api_messages)
                
                full_response = response_text
                message_placeholder.markdown(full_response)
                
                with st.expander("📚 Sources & context"):
                    for c in citations:
                        st.write(c)
                    st.text_area("Raw Context", context_text, height=200)

            except Exception as e:
                full_response = f"Error: {str(e)}"
                message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
