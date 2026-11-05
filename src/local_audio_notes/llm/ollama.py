from typing import Any

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.llm.http import local_json, valid_vectors


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
        return local_json(self.settings, "Ollama", method, endpoint, payload)

    def models(self) -> list[dict[str, Any]]:
        value = self.request("GET", "/api/tags")
        if not isinstance(value, dict) or not isinstance(value.get("models"), list):
            raise NotesError("Invalid Ollama model inventory")
        return value["models"]

    def require_local_model(self, name: str | None = None) -> str:
        """Verify a model (default: the notes model) is local; return a name+digest identity."""
        name = self.settings.ollama_model if name is None else name
        if not name or "cloud" in name.lower():
            raise NotesError("Select an installed local Ollama model in config.toml")
        models = self.models()
        matches = [m for m in models if m.get("name") == name]
        if not matches or any(m.get("remote_host") or m.get("remote_model") for m in matches):
            raise NotesError("Configured Ollama model is not an installed local model")
        info = self.request("POST", "/api/show", {"model": name})
        if (
            not isinstance(info, dict)
            or info.get("remote_host")
            or info.get("remote_model")
            or not info.get("model_info")
        ):
            raise NotesError("Ollama model has no verifiable local model metadata")
        return f"{name}@{matches[0].get('digest') or 'unknown-digest'}"

    def generate(self, system: str, content: str, schema: dict[str, Any]) -> str:
        value = self.request(
            "POST",
            "/api/chat",
            {
                "model": self.settings.ollama_model,
                "stream": False,
                "think": False,
                "keep_alive": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": self.settings.ollama_context,
                    "num_predict": self.settings.ollama_predict,
                },
            },
        )
        if (
            not isinstance(value, dict)
            or value.get("done") is not True
            or value.get("done_reason") == "length"
        ):
            raise NotesError(
                "Ollama response incomplete; reduce chunk size or increase output budget"
            )
        try:
            return value["message"]["content"]
        except (KeyError, TypeError):
            raise NotesError("Ollama response has no content") from None

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Local embeddings for a small batch; keep batches small (2 MiB response cap)."""
        value = self.request(
            "POST",
            "/api/embed",
            {"model": model, "input": texts, "keep_alive": 0, "truncate": True},
        )
        vectors = value.get("embeddings") if isinstance(value, dict) else None
        if not valid_vectors(vectors, len(texts)):
            raise NotesError("Ollama returned invalid embeddings")
        return vectors
