from local_audio_notes.config import Settings
from local_audio_notes.llm.lmstudio import LMStudioClient
from local_audio_notes.llm.ollama import OllamaClient

LLMClient = OllamaClient | LMStudioClient


def llm_client(settings: Settings) -> LLMClient:
    """The configured local LLM backend. Both talk only to `settings.ollama_url`."""
    if settings.llm_backend == "lmstudio":
        return LMStudioClient(settings)
    return OllamaClient(settings)
