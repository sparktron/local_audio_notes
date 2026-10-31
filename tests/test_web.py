import contextlib
import json
import re
import stat

import pytest
from fastapi.testclient import TestClient

from local_audio_notes.api import app as web
from local_audio_notes.config import load_settings
from local_audio_notes.jobs import queue
from local_audio_notes.storage.db import connect

BASE = "http://127.0.0.1:8765"
TOKEN = "fixed-test-token"


class FakeLauncher:
    def __init__(self):
        self.started = 0

    def running(self):
        return self.started > 0  # like a real worker that has not drained the queue yet

    def ensure(self):
        self.started += 1


@pytest.fixture
def config(settings, tmp_path):
    path = tmp_path / "config.toml"
    web.write_config(path, settings)
    return path


@pytest.fixture
def launcher():
    return FakeLauncher()


@pytest.fixture
def anonymous(settings, config, launcher):
    app = web.create_app(load_settings(config), config, 8765, TOKEN, launcher)
    return TestClient(app, base_url=BASE)


@pytest.fixture
def client(anonymous):
    assert anonymous.get(f"/login?token={TOKEN}").status_code == 200
    return anonymous


def csrf(client, path="/settings"):
    return re.search(r'name="csrf" value="([^"]+)"', client.get(path).text).group(1)


def post(client, path, data, token=None, **kwargs):
    return client.post(path, data={"csrf": token or csrf(client), **data}, **kwargs)


def upload(client, path, kind="audio"):
    return client.post(
        f"/api/upload?kind={kind}",
        content=path.read_bytes(),
        headers={
            "X-CSRF-Token": csrf(client, "/import"),
            "Content-Type": "application/octet-stream",
        },
    )


def imported(client, settings, audio, process=False):
    upload_id = upload(client, audio).json()["upload_id"]
    data = {"audio_upload": upload_id, "mode": "transcribe"}
    if process:
        data["process"] = "on"
    response = post(client, "/import", data, csrf(client, "/import"))
    assert response.status_code == 200, response.text
    return str(response.url).rsplit("/", 1)[1]


def run_queue(settings):
    with contextlib.closing(connect(settings.data_dir)) as connection:
        return queue.work(settings, connection)


def test_sign_in_required_and_security_headers(anonymous):
    response = anonymous.get("/")
    assert response.status_code == 401 and "Sign in required" in response.text
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert anonymous.get("/static/app.css").status_code == 200
    assert anonymous.get("/login?token=wrong").status_code == 403
    assert anonymous.get("/meetings/x/audio").status_code == 401
    response = anonymous.get(f"/login?token={TOKEN}", follow_redirects=False)
    assert response.status_code == 303
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert anonymous.get("/").status_code == 200


def test_no_external_resources_or_api_docs(client):
    page = client.get("/").text
    assert not re.search(r'(src|href)="(https?:)?//', page)
    assert "<script>" not in page  # only the same-origin external file
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_framework_telemetry_disabled(client):
    config = getattr(client.app, "_telemetry", None)
    if config is not None:  # FastAPI >= 0.162
        assert not any(config[k] for k in web.TELEMETRY_SWITCHES)


def test_dns_rebinding_host_rejected(client):
    response = client.get("/", headers={"Host": "attacker.example:8765"})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "headers",
    [{"Origin": "http://attacker.example"}, {"Sec-Fetch-Site": "cross-site"}],
)
def test_cross_site_posts_refused(client, headers):
    response = client.post("/settings", data={"csrf": csrf(client)}, headers=headers)
    assert response.status_code == 403


def test_csrf_token_required(client, audio):
    assert client.post("/settings", data={"batch_size": "4"}).status_code == 400
    assert post(client, "/settings", {"batch_size": "4"}, token="forged").status_code == 400
    response = client.post(
        "/api/upload?kind=audio", content=audio.read_bytes(), headers={"X-CSRF-Token": "x"}
    )
    assert response.status_code == 400


def test_upload_bounds_and_import(client, settings, audio, launcher):
    token = csrf(client, "/import")
    too_big = client.post(
        "/api/upload?kind=transcript",
        content=b"x" * (settings.max_transcript_bytes + 1),
        headers={"X-CSRF-Token": token},
    )
    assert too_big.status_code == 400
    assert (
        client.post(
            "/api/upload?kind=../../x", content=b"x", headers={"X-CSRF-Token": token}
        ).status_code
        == 400
    )
    identifier = imported(client, settings, audio, process=True)
    assert launcher.started == 1
    assert list((settings.data_dir / ".uploads").iterdir()) == []
    with contextlib.closing(connect(settings.data_dir)) as connection:
        assert queue.describe(connection, identifier)["state"] == "queued"
    # Importing the same bytes again resolves to the same meeting.
    assert imported(client, settings, audio) == identifier


def test_import_rejects_non_audio_and_forged_ids(client, settings, tmp_path):
    fake = tmp_path / "fake.wav"
    fake.write_text("not audio")
    upload_id = upload(client, fake).json()["upload_id"]
    response = post(client, "/import", {"audio_upload": upload_id}, csrf(client, "/import"))
    assert response.status_code == 400 and "Unsupported audio content" in response.text
    assert list((settings.data_dir / ".uploads").iterdir()) == []
    response = post(client, "/import", {"audio_upload": "../../etc/passwd"})
    assert response.status_code == 400


@pytest.mark.parametrize(
    "path",
    ["/meetings/..%2F..%2Fetc", "/meetings/not-a-uuid", "/meetings/not-a-uuid/export.md"],
)
def test_path_traversal_and_unknown_meetings(client, path):
    response = client.get(path)
    assert response.status_code in {400, 404}
    assert "Invalid meeting ID" in response.text or response.status_code == 404
    unknown = client.get("/meetings/00000000-0000-4000-8000-000000000000")
    assert unknown.status_code == 400 and "Meeting not found" in unknown.text


def test_full_flow_notes_speakers_and_exports(client, settings, audio, mocked):
    identifier = imported(client, settings, audio, process=True)
    assert run_queue(settings)[0][1] == "succeeded"
    page = client.get(f"/meetings/{identifier}").text
    assert "Executive summary" in page and "Decided to use SQLite" in page
    assert "<td>Alice</td>" in page and "<td>November 6</td>" in page
    assert 'class="seek" data-start="6.0"' in page

    post(client, f"/meetings/{identifier}/details", {"title": "<script>alert(1)</script>"})
    post(
        client,
        f"/meetings/{identifier}/speakers",
        {"name:Speaker 3": "Alice", "name:Speaker 1": ""},
    )
    post(
        client,
        f"/meetings/{identifier}/corrections",
        {"seg:s000006": "Speaker 1", "seg:s000000": "Speaker 1", "seg:bogus": "Speaker 1"},
    )
    page = client.get(f"/meetings/{identifier}").text
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert 'class="corrected"' in page and 'value="Alice"' in page
    dashboard = client.get("/?q=script").text
    assert "&lt;script&gt;" in dashboard and "<script>alert" not in dashboard

    markdown = client.get(f"/meetings/{identifier}/export.md")
    assert markdown.status_code == 200
    assert "attachment" in markdown.headers["content-disposition"]
    assert "Alice \\[Speaker 3\\] (confirmed)" in markdown.text
    assert "s000006 [00:00:18–00:00:20] Speaker 1" in markdown.text
    exported = client.get(f"/meetings/{identifier}/export.json").json()
    assert exported["meeting"]["title"] == "<script>alert(1)</script>"

    audio_response = client.get(f"/meetings/{identifier}/audio", headers={"Range": "bytes=0-99"})
    assert audio_response.status_code == 206 and len(audio_response.content) == 100
    assert audio_response.headers["content-type"] == "audio/wav"


def test_invalid_correction_target_rejected(client, settings, audio, mocked):
    identifier = imported(client, settings, audio, process=True)
    run_queue(settings)
    response = post(client, f"/meetings/{identifier}/corrections", {"seg:s000001": "Mallory"})
    assert response.status_code == 400 and "existing speaker label" in response.text


def test_process_and_cancel_from_browser(client, settings, audio, launcher):
    identifier = imported(client, settings, audio)
    post(client, f"/meetings/{identifier}/process", {"mode": "transcribe", "notes_only": "on"})
    assert launcher.started == 1
    with contextlib.closing(connect(settings.data_dir)) as connection:
        job = queue.describe(connection, identifier)["job"]
    assert job["kind"] == "notes"
    page = client.get(f"/meetings/{identifier}").text
    assert f"/jobs/{job['id']}/cancel" in page
    post(client, f"/jobs/{job['id']}/cancel", {})
    with contextlib.closing(connect(settings.data_dir)) as connection:
        assert queue.describe(connection, identifier)["state"] == "cancelled"
    assert post(client, f"/meetings/{identifier}/process", {"mode": "rm -rf"}).status_code == 400


def test_settings_saved_validated_and_paths_fixed(client, config, settings):
    response = post(
        client,
        "/settings",
        {
            "batch_size": "4",
            "device": "cpu",
            "min_speakers": "2",
            "max_speakers": "3",
            "ollama_model": "qwen2.5:14b",
            "data_dir": "/tmp/elsewhere",
            "ollama_url": "http://attacker.example:11434",
        },
    )
    assert response.status_code == 200 and "Saved" in response.text
    saved = load_settings(config)
    assert (saved.batch_size, saved.device, saved.min_speakers) == (4, "cpu", 2)
    assert saved.data_dir == settings.data_dir and saved.ollama_url == "http://127.0.0.1:11434"
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    response = post(client, "/settings", {"min_speakers": "9", "max_speakers": "2"})
    assert response.status_code == 400 and "Invalid settings" in response.text
    assert post(client, "/settings", {"batch_size": "lots"}).status_code == 400
    assert load_settings(config).batch_size == 4


def test_settings_read_only_without_config(settings, launcher):
    app = web.create_app(settings, None, 8765, TOKEN, launcher)
    client = TestClient(app, base_url=BASE)
    client.get(f"/login?token={TOKEN}")
    assert "read-only" in client.get("/settings").text
    assert post(client, "/settings", {"batch_size": "4"}).status_code == 400


def test_unexpected_errors_do_not_leak_details(client, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("PRIVATE transcript excerpt")

    monkeypatch.setattr(web, "describe", broken)
    client_raises = TestClient(client.app, base_url=BASE, raise_server_exceptions=False)
    client_raises.cookies = client.cookies
    response = client_raises.get("/meetings/00000000-0000-4000-8000-000000000000")
    assert "PRIVATE" not in response.text


def test_stale_uploads_removed_at_start(settings, config, launcher):
    staging = settings.data_dir / ".uploads"
    staging.mkdir(parents=True)
    (staging / ("0" * 32)).write_bytes(b"left over")
    web.create_app(load_settings(config), config, 8765, TOKEN, launcher)
    assert list(staging.iterdir()) == []


def test_worker_launcher_uses_fixed_arguments(monkeypatch, config):
    seen = {}

    class Popen:
        def __init__(self, args, **kwargs):
            seen.update(args=args, **kwargs)

        def poll(self):
            return None

    monkeypatch.setattr(web.subprocess, "Popen", Popen)
    launcher = web.WorkerLauncher(config)
    launcher.ensure()
    launcher.ensure()
    assert seen["args"][1:] == ["-m", "local_audio_notes", "--config", str(config), "worker"]
    assert "shell" not in seen and seen["stdin"] == web.subprocess.DEVNULL


@pytest.mark.parametrize("port", [80, 70000])
def test_serve_refuses_privileged_or_invalid_ports(settings, port):
    with pytest.raises(web.NotesError, match="Port"):
        web.serve(settings, None, port)


def test_health(client):
    assert json.loads(client.get("/health").text)["ok"] is True
