import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Tuple
import io
from PIL import Image

class PDFProcessor:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, file_bytes: bytes) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extracts text and images from a PDF file.
        Returns:
            text_chunks: List of text strings ready for embedding.
            images: List of dicts containing image metadata and bytes.
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""
        images = []

        for page_num, page in enumerate(doc):
            full_text += page.get_text()
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "bytes": image_bytes,
                    "ext": base_image["ext"]
                })

        text_chunks = self.text_splitter.split_text(full_text)
        return text_chunks, images
