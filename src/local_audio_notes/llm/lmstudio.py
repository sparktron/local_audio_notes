from typing import Any

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError
from local_audio_notes.llm.http import local_json, valid_vectors

# Weight formats LM Studio runs from local files. Entries without one (cloud or otherwise
# unknown models) are treated as remote and never used.
LOCAL_FORMATS = {"gguf", "mlx"}
# Idle seconds before LM Studio unloads a model it loaded on demand for these requests, so GPU
# memory is free for the next transcription (the counterpart of Ollama's keep_alive 0). Long
# enough to stay loaded across one meeting's consecutive requests.
IDLE_TTL_SECONDS = 60


class LMStudioClient:
    """LM Studio or Bionic on loopback: OpenAI-compatible chat/embeddings, native model list.

    LM Studio cannot set a context length per request; the model must be loaded with at
    least `ollama_context` tokens or the server may silently truncate the transcript. The
    `ttl` sent with each request unloads on-demand models soon after the last one; a model
    loaded by hand in the app stays loaded.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
        return local_json(self.settings, "LM Studio", method, endpoint, payload)

    def models(self) -> list[dict[str, Any]]:
        """Downloaded and loaded models, in the shape the Ollama inventory uses elsewhere."""
        value = self.request("GET", "/api/v0/models")
        data = value.get("data") if isinstance(value, dict) else None
        if not isinstance(data, list) or not all(
            isinstance(m, dict) and isinstance(m.get("id"), str) for m in data
        ):
            raise NotesError("Invalid LM Studio model inventory")
        return [
            {**m, "name": m["id"], "remote_model": m.get("compatibility_type") not in LOCAL_FORMATS}
            for m in data
        ]

    def require_local_model(self, name: str | None = None) -> str:
        """Verify a model (default: the notes model) is local; return an identity string."""
        notes_model = name is None
        name = self.settings.ollama_model if notes_model else name
        if not name or "cloud" in name.lower():
            raise NotesError("Select a downloaded local LM Studio model in config.toml")
        matches = [m for m in self.models() if m["name"] == name]
        if len(matches) != 1 or matches[0]["remote_model"]:
            raise NotesError("Configured LM Studio model is not a downloaded local model")
        model = matches[0]
        loaded = model.get("loaded_context_length")
        if notes_model and isinstance(loaded, int) and loaded < self.settings.ollama_context:
            raise NotesError("LM Studio model is loaded with less context than ollama_context")
        quantization = model.get("quantization") or "unknown-quantization"
        return f"lmstudio:{name}@{model['compatibility_type']}-{quantization}"

    def generate(self, system: str, content: str, schema: dict[str, Any]) -> str:
        value = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.settings.ollama_model,
                "stream": False,
                "ttl": IDLE_TTL_SECONDS,
                "temperature": 0,
                "max_tokens": self.settings.ollama_predict,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True, "schema": schema},
                },
            },
        )
        try:
            choice = value["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise NotesError("LM Studio response has no content") from None
        if choice.get("finish_reason") != "stop" or not isinstance(text, str):
            raise NotesError(
                "LM Studio response incomplete; reduce chunk size or increase output budget"
            )
        return text

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """Local embeddings for a small batch; keep batches small (2 MiB response cap)."""
        value = self.request(
            "POST", "/v1/embeddings", {"model": model, "input": texts, "ttl": IDLE_TTL_SECONDS}
        )
        data = value.get("data") if isinstance(value, dict) else None
        try:
            vectors = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        except (KeyError, TypeError):
            vectors = None
        if not valid_vectors(vectors, len(texts)):
            raise NotesError("LM Studio returned invalid embeddings")
        return vectors
