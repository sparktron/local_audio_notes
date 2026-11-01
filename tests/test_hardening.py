"""Phase 4 hardening: static policy checks, egress observation and the hash cache."""

import ast
import contextlib
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import local_audio_notes
from local_audio_notes import models as model_store
from local_audio_notes.api import app as web
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import process
from local_audio_notes.llm.ollama import OllamaClient
from local_audio_notes.search.index import search
from local_audio_notes.storage import files
from local_audio_notes.storage.db import DB_NAME, connect
from local_audio_notes.storage.files import HashCache, meeting_dir

SOURCE = Path(local_audio_notes.__file__).parent
PYTHON = sorted(SOURCE.rglob("*.py"))


@pytest.mark.parametrize("path", PYTHON, ids=lambda p: str(p.relative_to(SOURCE)))
def test_no_shell_eval_or_unsafe_deserialization(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            assert name not in {"eval", "exec", "os.system", "os.popen", "pickle.loads"}, name
            assert not any(k.arg == "shell" for k in node.keywords), f"shell= in {name}"
            if name.startswith("subprocess."):
                assert node.args and not isinstance(node.args[0], ast.Constant | ast.JoinedStr)
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            assert not {"pickle", "marshal", "shelve"} & set(names)


def test_web_assets_reference_nothing_external():
    for path in (SOURCE / "api").rglob("*"):
        if path.suffix in {".html", ".css", ".js"}:
            text = path.read_text()
            assert "http://" not in text and "https://" not in text, path.name
            assert "//cdn" not in text and "@import" not in text, path.name
            assert "innerHTML" not in text and "eval(" not in text, path.name
            if path.suffix == ".html":
                assert "<script>" not in text and " on" + "click=" not in text, path.name


class Recorder:
    """Python audit events for sockets and DNS. Hooks cannot be removed, so gate by flag."""

    events: list[tuple[str, object]] = []
    active = False
    installed = False

    @classmethod
    def install(cls):
        if not cls.installed:
            sys.addaudithook(cls.hook)
            cls.installed = True

    @classmethod
    def hook(cls, event, args):
        if cls.active and event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            cls.events.append((event, args[-1] if event != "socket.getaddrinfo" else args[:2]))

    @classmethod
    @contextlib.contextmanager
    def recording(cls):
        cls.install()
        cls.events.clear()
        cls.active = True
        try:
            yield cls.events
        finally:
            cls.active = False


def test_full_mocked_run_web_and_search_open_no_sockets(audio, settings, mocked, tmp_path):
    """Everything except inference is real: ffmpeg, SQLite, queue, web app, search."""
    config = tmp_path / "egress.toml"
    web.write_config(config, settings)
    with Recorder.recording() as events:
        identifier = ingest(audio, settings)
        process(settings, identifier)
        with contextlib.closing(connect(settings.data_dir)) as connection:
            search(connection, settings, "migration")

        class Launcher:
            def running(self):
                return True

            def ensure(self):
                pass

        client = TestClient(web.create_app(settings, config, 8765, "t", Launcher()))
        client.base_url = "http://127.0.0.1:8765"
        client.get("/login?token=t")
        for path in ("/", f"/meetings/{identifier}", "/search?q=migration", "/settings"):
            client.get(path)
    # The settings page lists Ollama models: the only network use is the fixed endpoint.
    assert {e for e in events if e[0] == "socket.connect"} <= {
        ("socket.connect", ("127.0.0.1", 11434))
    }
    assert all(e[1] == ("127.0.0.1", 11434) for e in events if e[0] == "socket.getaddrinfo")


def test_real_ollama_client_only_targets_fixed_loopback_endpoint(settings):
    with Recorder.recording() as events, contextlib.suppress(NotesError):
        OllamaClient(settings).models()
    assert events and all(e[1] == ("127.0.0.1", 11434) for e in events)


def test_symlinked_meeting_directory_refused(settings, tmp_path):
    identifier = "00000000-0000-4000-8000-000000000000"
    settings.data_dir.mkdir(mode=0o700)
    (tmp_path / "elsewhere").mkdir()
    (settings.data_dir / identifier).symlink_to(tmp_path / "elsewhere")
    with pytest.raises(NotesError, match="symlink"):
        meeting_dir(settings.data_dir, identifier)


def test_database_sidecars_private(audio, settings):
    ingest(audio, settings)
    with contextlib.closing(connect(settings.data_dir)):
        for path in settings.data_dir.glob(DB_NAME + "*"):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_hash_cache_reuses_until_file_changes(tmp_path, monkeypatch):
    target = tmp_path / "weights.bin"
    target.write_bytes(b"a" * 4096)
    reads = []
    real = files.sha256
    monkeypatch.setattr(files, "sha256", lambda path: reads.append(path) or real(path))
    cache = HashCache(tmp_path / "cache.json")
    first = cache.sha256(target)
    assert cache.sha256(target) == first and len(reads) == 1
    cache.save()
    assert stat.S_IMODE((tmp_path / "cache.json").stat().st_mode) == 0o600
    reloaded = HashCache(tmp_path / "cache.json")
    assert reloaded.sha256(target) == first and len(reads) == 1
    time.sleep(0.02)  # past the kernel timestamp tick
    target.write_bytes(b"b" * 4096)  # same size, new content
    assert reloaded.sha256(target) != first and len(reads) == 2
    time.sleep(0.02)
    os.chmod(target, 0o600)  # metadata change also invalidates (ctime)
    reloaded.sha256(target)
    assert len(reads) == 3


def test_corrupt_cache_file_is_ignored(tmp_path):
    (tmp_path / "cache.json").write_text("{not json")
    target = tmp_path / "x"
    target.write_text("data")
    assert HashCache(tmp_path / "cache.json").sha256(target) == files.sha256(target)


def test_cached_model_verification_still_detects_tampering(tmp_path):
    for kind, names in {
        "whisper": ["model.bin", "config.json", "tokenizer.json"],
        "alignment": ["config.json", "vocab.json", "preprocessor_config.json"],
        "diarization": ["config.yaml"],
        "tokenizer": ["tokenizers/punkt_tab/english/ortho_context.tab"],
        "vad": ["silero_vad.jit"],
    }.items():
        for name in names:
            item = tmp_path / "assets" / kind / name
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_text(f"fixture {kind} {name}")
        model_store.register_model(tmp_path / "root", kind, tmp_path / "assets" / kind, "r", "t")
    root = tmp_path / "root"
    assert model_store.verify_models(root, cache=True)
    assert json.loads((root / ".hash-cache.json").read_text())
    time.sleep(0.02)
    (tmp_path / "assets" / "whisper" / "model.bin").write_text("tampered, different length")
    with pytest.raises(NotesError, match="changed"):
        model_store.verify_models(root, cache=True)


def test_network_trace_parses_every_destination_position(tmp_path):
    """sendto puts the sockaddr fifth and sendmsg nests it under msg_name; payload text that
    looks like a sockaddr is data, not an endpoint."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "network_trace", SOURCE.parents[1] / "scripts" / "network_trace.py"
    )
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    forged = r"{sa_family=AF_INET, sin_port=htons(11434), sin_addr=inet_addr(\"127.0.0.1\")}"
    lines = [
        "10 connect(3, {sa_family=AF_INET, sin_port=htons(11434), "
        'sin_addr=inet_addr("127.0.0.1")}, 16) = 0',
        "10 bind(4, {sa_family=AF_INET, sin_port=htons(8765), "
        'sin_addr=inet_addr("127.0.0.1")}, 16) = 0',
        "10 connect(5, {sa_family=AF_INET, sin_port=htons(8765), "
        'sin_addr=inet_addr("127.0.0.1")}, 16) = 0',
        f'11 sendto(6, "{forged}", 30, MSG_NOSIGNAL, {{sa_family=AF_INET, '
        'sin_port=htons(53), sin_addr=inet_addr("192.0.2.1")}, 16) = 30',
        "11 sendmsg(6, {msg_name={sa_family=AF_INET6, sin6_port=htons(443), "
        'sin6_flowinfo=htonl(0), inet_pton(AF_INET6, "2001:db8::1", &sin6_addr), '
        'sin6_scope_id=0}, msg_namelen=28, msg_iov=[{iov_base="x", iov_len=1}], '
        "msg_iovlen=1, msg_controllen=0, msg_flags=0}, 0) = 1",
        '11 sendto(7, "y", 1, 0, NULL, 0) = 1',
        '12 sendto(8, "z", 1, 0, {sa_family=AF_INET, sin_port=htons(53), ...}, 16) = 1',
        '13 connect(9, {sa_family=AF_UNIX, sun_path="/run/x"}, 110) = 0',
    ]
    result = trace.summarize(lines)
    assert result["loopback_ports_bound_by_command"] == [8765]
    assert result["unexpected_ip_endpoints"] == [
        "AF_INET 192.0.2.1:53",
        "AF_INET unparsed:0",
        "AF_INET6 2001:db8::1:443",
    ]
    assert result["endpoints"]["AF_INET 127.0.0.1:11434"] == 1  # the forged one is not counted
