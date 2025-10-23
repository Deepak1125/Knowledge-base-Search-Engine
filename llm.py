from langchain_core.language_models.llms import LLM
from typing import List, Optional
from pydantic import Field
import requests
import json

# -------------------- Gemini LLM --------------------
class GeminiLLM(LLM):
    api_key: str = Field(...)
    model_name: str = Field(default="gemini-2.5-pro")
    url: str = None

    def __init__(self, **data):
        super().__init__(**data)
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }

        response = requests.post(self.url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            try:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                return "⚠️ Unexpected Gemini response format."
        else:
            return f"❌ Error: {response.status_code} - {response.text}"

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"


# -------------------- Helper Function --------------------
# This is the one you'll call in your FastAPI route
gemini_model = None

def init_llm(api_key: str, model_name: str = "gemini-2.5-pro"):
    """Initialize the global Gemini LLM instance (call once at startup)."""
    global gemini_model
    gemini_model = GeminiLLM(api_key=api_key, model_name=model_name)

def ask_llm(prompt: str) -> str:
    """Call Gemini LLM with a prompt and return the response text."""
    global gemini_model
    if gemini_model is None:
        return "❌ LLM not initialized. Call init_llm(api_key) first."
    return gemini_model(prompt)
