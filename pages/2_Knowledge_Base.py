import streamlit as st
from utils.pdf_processor import PDFProcessor
from utils.gemini_handler import GeminiHandler
from utils.qdrant_handler import QdrantHandler
from qdrant_client.models import PointStruct
import uuid
import time
from PIL import Image
import io

st.title("📔 Knowledge Base Builder")

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""
if "qdrant_api_key" not in st.session_state:
    st.session_state.qdrant_api_key = ""
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if not st.session_state.get("gemini_api_key") or not st.session_state.get("qdrant_api_key"):
    st.warning("🎡 Please configure your API keys in the Configuration page first.")
    st.stop()

# Initialize Handlers
gemini = GeminiHandler(st.session_state.gemini_api_key)
qdrant = QdrantHandler(st.session_state.qdrant_url, st.session_state.qdrant_api_key)

# Initialize Collection
try:
    qdrant.create_collection()
    st.success("☑️ Connected to Qdrant Vector Store")
except Exception as e:
    st.error(f"🥲 Failed to connect to Qdrant: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

if uploaded_file:
    with st.expander("📄 PDF Processing Status", expanded=True):
        process_images = st.checkbox("Enable Multimodal Processing (Extract & Describe Images)", value=True)
        st.info("💡 Standardized High-Quality Chunking is enabled (1500 chars / 10% overlap).")

    if st.button("🪄 Process and Index Document"):
        with st.status("Processing Document...", expanded=True) as status:
            
            status.write("Extracting content from PDF...")
            file_bytes = uploaded_file.read()
            processor = PDFProcessor()
            text_chunks, images = processor.process_pdf(file_bytes)
            st.write(f"extracted {len(text_chunks)} text chunks and {len(images)} images.")
            
            image_descriptions = []
            if process_images and images:
                status.write(f"Analyzing {len(images)} images with Gemini Vision...")
                progress_bar = st.progress(0)
                for i, img_data in enumerate(images):
                    try:
                        image = Image.open(io.BytesIO(img_data["bytes"]))
                        
                        status.write(f"Describing image {i+1} of {len(images)}...")
                        desc = gemini.describe_image(img_data["bytes"])
                        
                        if not desc or desc.strip() == "":
                            desc = f"Image on page {img_data['page']}, index {img_data['index']}. [Image Description Failed]" 
                            
                        image_descriptions.append(desc)
                    except Exception as e:
                        print(f"Error processing image {i}: {e}")
                    progress_bar.progress((i + 1) / len(images))
            
            status.write("✨Generating Embeddings and Indexing...")
            points = []
            
            full_content_for_context = []
            
            for i, chunk in enumerate(text_chunks):
                emb = gemini.get_embedding(chunk)
                if emb:
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={"text": chunk, "type": "text", "source": uploaded_file.name}
                    ))
                    full_content_for_context.append(chunk)

            for i, desc in enumerate(image_descriptions):
                emb = gemini.get_embedding(desc)
                if emb:
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={"text": desc, "type": "image", "source": uploaded_file.name, "page": images[i]['page']}
                    ))

            if points:
                qdrant.upsert_points(points)
                
                st.session_state.indexed_files.append(uploaded_file.name)
                
                status.update(label="✅ Indexing Complete!", state="complete", expanded=False)
                st.success(f"Successfully indexed {len(points)} items from '{uploaded_file.name}'.")
                
                if len(full_content_for_context) > 20:
                    st.toast("💡 Large document detected. Context Caching can be enabled in Chat.")
            else:
                st.error("No content could be indexed.")

if st.session_state.indexed_files:
    st.sidebar.markdown("### 📄 Indexed Documents")
    for f in st.session_state.indexed_files:
        st.sidebar.write(f"- {f}")


