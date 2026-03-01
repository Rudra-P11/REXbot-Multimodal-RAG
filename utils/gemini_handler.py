from google import genai
from google.genai import types
import time
from typing import List, Dict, Any, Optional
import datetime

class GeminiHandler:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash" 
        self.embedding_model = "models/gemini-embedding-001"
        
    def get_embedding(self, text: str) -> List[float]:
        """Generates embeddings for the given text."""
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def get_query_embedding(self, text: str) -> List[float]:
        """Generates embeddings for a query."""
        try:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

    def generate_response(
        self, 
        system_instruction: str, 
        messages: List[Dict[str, Any]], 
        context: Optional[str] = None
    ) -> str:
        """Generates a response using the new google-genai SDK."""
        try:
            contents = []
            for msg in messages:
                role = "user" if msg['role'] == "user" else "model"
                parts = msg.get('parts', msg.get('content', ''))
                if isinstance(parts, list):
                    parts = parts[0] 
                contents.append(types.Content(role=role, parts=[types.Part(text=parts)]))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini (SDK v2): {e}"

    def describe_image(self, image_bytes: bytes) -> str:
        """Processes an image using Gemini Vision to generate a description."""
        try:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_bytes))
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[image, "Describe this image, chart, or table in high detail. Focus on extracting all text, numbers, and structural information."],
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return response.text
        except Exception as e:
            print(f"Error describing image: {e}")
            return ""

    def create_context_cache(self, content: str, display_name: str):
        """Creates a context cache using the new SDK."""
        try:
            cache = self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    display_name=display_name,
                    system_instruction=f"Answer based on: {display_name}",
                    contents=[content],
                    ttl="600s"
                )
            )
            return cache
        except Exception as e:
            print(f"Error creating cache: {e}")
            return None
