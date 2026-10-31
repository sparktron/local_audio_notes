import json

import httpx
import pytest

from local_audio_notes.errors import NotesError
from local_audio_notes.llm.ollama import OllamaClient


def test_http_request_schema_no_proxy_redirects(monkeypatch, settings):
    real_client = httpx.Client
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"done": True, "message": {"content": '{"items": []}'}})

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", client)
    assert OllamaClient(settings).generate("system", "data", {"type": "object"}) == '{"items": []}'
    assert captured[0]["format"] == {"type": "object"}
    assert captured[0]["keep_alive"] == 0


@pytest.mark.parametrize("remote", ["remote_host", "remote_model"])
def test_cloud_alias_rejected(monkeypatch, settings, remote):
    client = OllamaClient(settings)
    monkeypatch.setattr(client, "models", lambda: [{"name": settings.ollama_model}])
    monkeypatch.setattr(client, "request", lambda *args: {remote: "remote", "model_info": {"x": 1}})
    with pytest.raises(NotesError, match="local"):
        client.require_local_model()


def test_truncation_rejected(monkeypatch, settings):
    client = OllamaClient(settings)
    monkeypatch.setattr(client, "request", lambda *args: {"done": True, "done_reason": "length"})
    with pytest.raises(NotesError, match="incomplete"):
        client.generate("system", "data", {})
