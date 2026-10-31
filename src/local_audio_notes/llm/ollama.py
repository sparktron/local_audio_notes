import json
from typing import Any

import httpx

from local_audio_notes.config import Settings
from local_audio_notes.errors import NotesError


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            with (
                httpx.Client(
                    trust_env=False, follow_redirects=False, timeout=self.settings.ollama_timeout
                ) as client,
                client.stream(
                    method, self.settings.ollama_url + endpoint, json=payload
                ) as response,
            ):
                response.raise_for_status()
                chunks = bytearray()
                for block in response.iter_bytes():
                    chunks.extend(block)
                    if len(chunks) > 2 * 1024**2:
                        raise NotesError("Ollama response exceeds limit")
                return json.loads(chunks)
        except (httpx.HTTPError, ValueError):
            raise NotesError("Local Ollama API unavailable or returned invalid data") from None

    def models(self) -> list[dict[str, Any]]:
        value = self.request("GET", "/api/tags")
        if not isinstance(value, dict) or not isinstance(value.get("models"), list):
            raise NotesError("Invalid Ollama model inventory")
        return value["models"]

    def require_local_model(self) -> str:
        """Verify the configured model is local; return a name+digest identity for caching."""
        name = self.settings.ollama_model
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
