import json

import httpx
import pytest

from local_audio_notes.errors import NotesError
from local_audio_notes.llm import llm_client
from local_audio_notes.llm.lmstudio import IDLE_TTL_SECONDS, LMStudioClient
from local_audio_notes.llm.ollama import OllamaClient

MODEL = "qwen3-14b-abliterated"
LOCAL = {"id": MODEL, "type": "llm", "compatibility_type": "gguf", "quantization": "Q6_K"}


@pytest.fixture
def lmstudio(settings):
    settings.llm_backend = "lmstudio"
    settings.ollama_url = "http://127.0.0.1:1234"
    settings.ollama_model = MODEL
    return settings


def serve(monkeypatch, respond):
    """Route httpx through `respond(request)`, recording every request."""
    real_client, seen = httpx.Client, []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=respond(request))

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "Client", client)
    return seen


def reply(content='{"items": []}', finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def test_backend_selection(settings, lmstudio):
    assert isinstance(llm_client(lmstudio), LMStudioClient)
    lmstudio.llm_backend = "ollama"
    assert isinstance(llm_client(lmstudio), OllamaClient)


def test_chat_request_uses_json_schema_on_configured_port(monkeypatch, lmstudio):
    seen = serve(monkeypatch, lambda request: reply())
    schema = {"type": "object", "properties": {"items": {"type": "array"}}}
    assert LMStudioClient(lmstudio).generate("system", "data", schema) == '{"items": []}'
    (request,) = seen
    assert str(request.url) == "http://127.0.0.1:1234/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == MODEL and body["stream"] is False and body["temperature"] == 0
    assert body["ttl"] == IDLE_TTL_SECONDS  # on-demand models unload after the notes
    assert body["max_tokens"] == lmstudio.ollama_predict
    assert body["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "data"},
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": schema},
    }


@pytest.mark.parametrize(
    "value",
    [reply(finish="length"), reply(finish=None), reply(content=None), {"choices": []}, {}],
)
def test_incomplete_or_empty_response_rejected(monkeypatch, lmstudio, value):
    client = LMStudioClient(lmstudio)
    monkeypatch.setattr(client, "request", lambda *args: value)
    with pytest.raises(NotesError, match="incomplete|no content"):
        client.generate("system", "data", {})


def test_server_errors_are_redacted(monkeypatch, lmstudio):
    real_client = httpx.Client

    def client(**kwargs):
        transport = httpx.MockTransport(lambda r: httpx.Response(500, text="secret transcript"))
        return real_client(**kwargs, transport=transport)

    monkeypatch.setattr(httpx, "Client", client)
    with pytest.raises(NotesError, match="^Local LM Studio API unavailable") as error:
        LMStudioClient(lmstudio).models()
    assert "secret" not in str(error.value)


def test_local_model_accepted_with_identity(monkeypatch, lmstudio):
    seen = serve(monkeypatch, lambda request: {"data": [LOCAL]})
    assert LMStudioClient(lmstudio).require_local_model() == f"lmstudio:{MODEL}@gguf-Q6_K"
    assert str(seen[0].url) == "http://127.0.0.1:1234/api/v0/models"


@pytest.mark.parametrize(
    "entries",
    [
        [],  # not downloaded
        [{**LOCAL, "compatibility_type": None}],  # no local weight format: cloud or unknown
        [{"id": MODEL, "type": "llm"}],
        [LOCAL, LOCAL],  # ambiguous
    ],
)
def test_non_local_models_rejected(monkeypatch, lmstudio, entries):
    serve(monkeypatch, lambda request: {"data": entries})
    with pytest.raises(NotesError, match="local"):
        LMStudioClient(lmstudio).require_local_model()


def test_cloud_name_rejected_before_any_request(monkeypatch, lmstudio):
    seen = serve(monkeypatch, lambda request: {"data": [{**LOCAL, "id": "x-cloud"}]})
    with pytest.raises(NotesError, match="local"):
        LMStudioClient(lmstudio).require_local_model("x-cloud")
    assert seen == []


def test_short_loaded_context_rejected(monkeypatch, lmstudio):
    loaded = {**LOCAL, "loaded_context_length": lmstudio.ollama_context - 1}
    serve(monkeypatch, lambda request: {"data": [loaded]})
    with pytest.raises(NotesError, match="context"):
        LMStudioClient(lmstudio).require_local_model()


def test_embedding_model_does_not_require_notes_context(monkeypatch, lmstudio):
    embedding = {
        **LOCAL,
        "id": "nomic-embed-text",
        "type": "embeddings",
        "loaded_context_length": 2048,
    }
    serve(monkeypatch, lambda request: {"data": [embedding]})
    assert LMStudioClient(lmstudio).require_local_model("nomic-embed-text") == (
        "lmstudio:nomic-embed-text@gguf-Q6_K"
    )


def test_inventory_marks_remote_entries(monkeypatch, lmstudio):
    cloud = {"id": "deepseek-ai/DeepSeek-V4-Pro", "type": "llm"}
    serve(monkeypatch, lambda request: {"data": [LOCAL, cloud]})
    models = LMStudioClient(lmstudio).models()
    assert [(m["name"], m["remote_model"]) for m in models] == [
        (MODEL, False),
        (cloud["id"], True),
    ]


@pytest.mark.parametrize("value", [{"data": "x"}, {"data": [{"name": MODEL}]}, []])
def test_invalid_inventory_rejected(monkeypatch, lmstudio, value):
    serve(monkeypatch, lambda request: value)
    with pytest.raises(NotesError, match="inventory"):
        LMStudioClient(lmstudio).models()


def test_embeddings_ordered_by_index(monkeypatch, lmstudio):
    data = [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
    seen = serve(monkeypatch, lambda request: {"data": data})
    assert LMStudioClient(lmstudio).embed("nomic", ["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert str(seen[0].url) == "http://127.0.0.1:1234/v1/embeddings"
    assert json.loads(seen[0].content) == {
        "model": "nomic",
        "input": ["a", "b"],
        "ttl": IDLE_TTL_SECONDS,
    }


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [1.0]}],  # wrong count
        [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}],
        [{"index": 0, "embedding": [float("nan")]}, {"index": 1, "embedding": [1.0]}],
        [{"embedding": [1.0]}, {"embedding": [1.0]}],
        None,
    ],
)
def test_invalid_embeddings_rejected(monkeypatch, lmstudio, data):
    client = LMStudioClient(lmstudio)
    monkeypatch.setattr(client, "request", lambda *args: {"data": data})
    with pytest.raises(NotesError, match="invalid embeddings"):
        client.embed("nomic", ["a", "b"])
