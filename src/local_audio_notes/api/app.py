"""Loopback-only FastAPI interface over the same storage, queue and pipeline as the CLI."""

import asyncio
import contextlib
import inspect
import json
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from local_audio_notes.api.security import COOKIE, HEADERS, LOOPBACK_HOSTS, Guard, read_form
from local_audio_notes.config import Settings, load_settings
from local_audio_notes.errors import NotesError
from local_audio_notes.extraction.grounding import normalize
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import cancel, describe, enqueue, lock_path
from local_audio_notes.llm.ollama import OllamaClient
from local_audio_notes.notes.render import timestamp
from local_audio_notes.pipeline import export_notes, get_meeting, verified_checkpoint
from local_audio_notes.schemas import CATEGORIES, Transcript
from local_audio_notes.search.index import MARK_END, MARK_START, search
from local_audio_notes.speakers import mapping
from local_audio_notes.storage.db import connect, transaction
from local_audio_notes.storage.files import meeting_dir, private_dir
from local_audio_notes.storage.locking import exclusive_lock

HERE = Path(__file__).parent
TELEMETRY_SWITCHES = ("tracing", "metrics", "logs", "operation_spans", "auto_configure")
UPLOAD_ID = re.compile(r"[0-9a-f]{32}")
TRANSCRIPT_KINDS = {".txt", ".srt", ".json"}
MEDIA_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg"}
# Fields editable from the browser. Storage paths, network and offline enforcement are not:
# changing them from a web page would widen what a forged request could do.
EDITABLE: dict[str, type] = {
    "whisper_model": str,
    "language": str,
    "alignment_language": str,
    "device": str,
    "compute_type": str,
    "batch_size": int,
    "cpu_threads": int,
    "min_speakers": int,
    "max_speakers": int,
    "ollama_model": str,
    "chunk_chars": int,
    "chunk_overlap_segments": int,
    "oom_cpu_fallback": bool,
    "embedding_model": str,
}


class WorkerLauncher:
    """Runs queued jobs in a separate `worker` process (fixed argument array, no shell), so
    model memory is released when the queue drains and the web server stays responsive."""

    def __init__(self, config_path: Path | None):
        self.config_path = config_path
        self.process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ensure(self) -> None:
        with self._lock:
            if self.running():
                return
            config = ["--config", str(self.config_path)] if self.config_path else []
            self.process = subprocess.Popen(
                [sys.executable, "-m", "local_audio_notes", *config, "worker"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)  # JSON strings are valid TOML strings


def write_config(path: Path, settings: Settings) -> None:
    from local_audio_notes.storage.files import atomic_write

    lines = ["# Written by the local web interface; comments are not preserved."]
    lines += [f"{k} = {toml_value(v)}" for k, v in settings.model_dump(mode="json").items()]
    atomic_write(path, "\n".join(lines) + "\n")
    path.chmod(0o600)


@contextlib.contextmanager
def database(settings: Settings) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(settings.data_dir)) as connection:
        yield connection


def create_app(
    settings: Settings,
    config_path: Path | None = None,
    port: int = 8765,
    login_token: str | None = None,
    launcher: WorkerLauncher | None = None,
) -> FastAPI:
    guard = Guard(port, login_token)
    launcher = launcher or WorkerLauncher(config_path)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["timestamp"] = timestamp

    def highlight(snippet: str) -> Markup:
        # Escape first, then turn the private-use match markers into <mark> tags.
        escaped = str(escape(snippet))
        return Markup(escaped.replace(MARK_START, "<mark>").replace(MARK_END, "</mark>"))

    templates.env.filters["highlight"] = highlight
    state: dict[str, Any] = {"settings": settings}
    uploads = private_dir(private_dir(settings.data_dir) / ".uploads")
    # Staged uploads are this server's own scratch copies; originals live in meeting dirs.
    for stale in uploads.iterdir():
        if stale.is_file() and not stale.is_symlink():
            stale.unlink()

    def queued_jobs() -> bool:
        with database(state["settings"]) as connection:
            return bool(
                connection.execute("SELECT 1 FROM jobs WHERE state = 'queued' LIMIT 1").fetchone()
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        async def watch() -> None:
            # A job queued while a worker is exiting would otherwise wait for the next request.
            while True:
                await asyncio.sleep(10)
                with contextlib.suppress(Exception):
                    if not launcher.running() and await run_in_threadpool(queued_jobs):
                        launcher.ensure()

        task = asyncio.create_task(watch())
        yield
        task.cancel()

    options: dict[str, Any] = {}
    if "telemetry" in inspect.signature(FastAPI).parameters:
        # FastAPI >= 0.162 instruments requests (exception text included) and can export
        # OTLP from environment variables. Disable all of it: no telemetry leaves the host.
        options["telemetry"] = dict.fromkeys(TELEMETRY_SWITCHES, False)
    # No /docs, /redoc or OpenAPI: the docs pages load assets from a CDN.
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan, **options)
    app.state.guard = guard
    app.state.launcher = launcher
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def protect(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        if request.method not in {"GET", "HEAD"} and guard.cross_site(request):
            response: Response = Response("Cross-site request refused", status_code=403)
        elif path != "/login" and not path.startswith("/static/") and not guard.session(request):
            response = templates.TemplateResponse(
                request, "signin.html", {"csrf": ""}, status_code=401
            )
        else:
            response = await call_next(request)
        response.headers.update(HEADERS)
        return response

    # Added last so it runs first: reject DNS-rebinding hostnames before anything else.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOOPBACK_HOSTS))

    @app.exception_handler(NotesError)
    async def notes_error(request: Request, error: NotesError) -> Response:
        return page(request, "error.html", {"message": str(error)}, status=400)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception) -> Response:
        # Library exception text can contain transcript content or paths: class name only.
        message = f"{type(error).__name__}; details suppressed for privacy"
        return page(request, "error.html", {"message": message}, status=500)

    def page(request: Request, name: str, context: dict[str, Any], status: int = 200) -> Response:
        return templates.TemplateResponse(
            request, name, {"csrf": guard.csrf(request), **context}, status_code=status
        )

    def back(url: str) -> RedirectResponse:
        return RedirectResponse(url, status_code=303)

    @app.get("/login")
    def login(token: str = "") -> Response:
        session = guard.login(token)
        if session is None:
            return Response("Invalid or expired sign-in link", status_code=403)
        response = back("/")
        response.set_cookie(COOKIE, session, httponly=True, samesite="strict", path="/")
        return response

    @app.get("/")
    def dashboard(request: Request, q: str = "") -> Response:
        current = state["settings"]
        with database(current) as connection:
            query = normalize(q)[:200]
            pattern = "%" + re.sub(r"([\\%_])", r"\\\1", query) + "%"
            meetings = connection.execute(
                "SELECT m.id, m.title, m.meeting_date, m.duration, m.imported_at, "
                "(SELECT state FROM jobs j WHERE j.meeting_id = m.id ORDER BY j.id DESC LIMIT 1)"
                " AS state, (SELECT stage FROM jobs j WHERE j.meeting_id = m.id "
                "ORDER BY j.id DESC LIMIT 1) AS stage FROM meetings m "
                "WHERE ? = '' OR ifnull(m.title, '') LIKE ? ESCAPE '\\' "
                "OR ifnull(m.meeting_date, '') LIKE ? ESCAPE '\\' "
                "ORDER BY m.imported_at DESC LIMIT 100",
                (query, pattern, pattern),
            ).fetchall()
            stats = {
                "meetings": connection.execute("SELECT count(*) FROM meetings").fetchone()[0],
                "hours": connection.execute(
                    "SELECT ifnull(sum(duration), 0) / 3600.0 FROM meetings"
                ).fetchone()[0],
                "jobs": dict(
                    connection.execute("SELECT state, count(*) FROM jobs GROUP BY state").fetchall()
                ),
                "stages": connection.execute(
                    "SELECT stage, count(*) AS runs, avg(seconds) AS mean, max(seconds) AS longest "
                    "FROM stage_runs WHERE state = 'succeeded' GROUP BY stage ORDER BY stage"
                ).fetchall(),
            }
        if stats["jobs"].get("queued") and not launcher.running():
            launcher.ensure()
        return page(
            request,
            "dashboard.html",
            {"meetings": meetings, "stats": stats, "q": query, "worker": launcher.running()},
        )

    @app.get("/search")
    def search_page(
        request: Request,
        q: str = "",
        participant: str = "",
        category: str = "",
        date_from: str = "",
        date_to: str = "",
        semantic: str = "",
    ) -> Response:
        current = state["settings"]
        hits, error = [], None
        if category and category not in CATEGORIES:
            raise NotesError("Unknown category")
        for value in (date_from, date_to):
            if value:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    raise NotesError("Dates must be YYYY-MM-DD") from None
        if q.strip():
            try:
                with database(current) as connection:
                    hits = search(
                        connection,
                        current,
                        q[:500],
                        participant=participant[:100],
                        category=category,
                        date_from=date_from,
                        date_to=date_to,
                        semantic=semantic == "on",
                        embedder=OllamaClient(current) if semantic == "on" else None,
                    )
            except NotesError as found:
                error = str(found)
        return page(
            request,
            "search.html",
            {
                "q": q,
                "participant": participant,
                "category": category,
                "date_from": date_from,
                "date_to": date_to,
                "semantic": semantic == "on",
                "semantic_available": bool(current.embedding_model),
                "categories": CATEGORIES,
                "hits": hits,
                "error": error,
            },
        )

    @app.get("/import")
    def import_page(request: Request) -> Response:
        current = state["settings"]
        return page(
            request,
            "import.html",
            {
                "audio_limit": current.max_audio_bytes,
                "transcript_limit": current.max_transcript_bytes,
            },
        )

    @app.post("/api/upload")
    async def upload(request: Request, kind: str) -> Response:
        guard.check_csrf(request, request.headers.get("x-csrf-token"))
        current = state["settings"]
        limits = {"audio": current.max_audio_bytes, "transcript": current.max_transcript_bytes}
        if kind not in limits:
            raise NotesError("Unknown upload kind")
        declared = request.headers.get("content-length")
        if declared is None or not declared.isdigit() or not 0 < int(declared) <= limits[kind]:
            raise NotesError("Upload is empty or exceeds the size limit")
        identifier = secrets.token_hex(16)
        target = uploads / identifier
        size = 0
        try:
            with open(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
                async for block in request.stream():
                    size += len(block)
                    if size > limits[kind]:
                        raise NotesError("Upload exceeds the size limit")
                    out.write(block)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return JSONResponse({"upload_id": identifier})

    @app.post("/import")
    async def import_submit(request: Request) -> Response:
        form = await read_form(request, guard)
        audio_id, transcript_id = form.get("audio_upload", ""), form.get("transcript_upload", "")
        kind = form.get("transcript_kind", "")
        mode = form.get("mode", "transcribe")
        if not UPLOAD_ID.fullmatch(audio_id) or (
            transcript_id
            and (not UPLOAD_ID.fullmatch(transcript_id) or kind not in TRANSCRIPT_KINDS)
        ):
            raise NotesError("Upload the audio file (and optional transcript) first")
        if mode not in {"transcribe", "align", "compare"}:
            raise NotesError("Unknown processing mode")

        def run() -> str:
            current = state["settings"]
            staged = [uploads / audio_id]
            transcript = None
            try:
                if transcript_id:
                    # ingest() reads the format from the suffix; the name is ours, not the user's.
                    transcript = uploads / f"{transcript_id}{kind}"
                    (uploads / transcript_id).rename(transcript)
                    staged.append(transcript)
                identifier = ingest(staged[0], current, transcript)
            finally:
                for path in staged:
                    path.unlink(missing_ok=True)
            if form.get("process") == "on":
                with database(current) as connection:
                    enqueue(connection, current, identifier, "full", mode)
                launcher.ensure()
            return identifier

        identifier = await run_in_threadpool(run)
        return back(f"/meetings/{identifier}")

    def meeting_view(identifier: str) -> dict[str, Any]:
        current = state["settings"]
        directory = meeting_dir(current.data_dir, identifier)
        with database(current) as connection:
            meeting = get_meeting(connection, identifier)
            status = describe(connection, identifier)
            found = verified_checkpoint(connection, identifier, directory, "attributing")
            transcript, view, labels = None, mapping.SpeakerView(), []
            if found:
                transcript = Transcript.model_validate_json(
                    (directory / "transcript.json").read_text()
                )
                view = mapping.load_view(connection, identifier, found["transcript.json"])
                labels = sorted(
                    mapping.labels(connection, identifier), key=lambda x: int(x.split()[1])
                )
        notes = None
        if (directory / "notes.json").is_file():
            notes = json.loads((directory / "notes.json").read_text())
        return {
            "meeting": meeting,
            "status": status,
            "transcript": transcript,
            "view": view,
            "labels": labels,
            "choices": labels + list(mapping.UNRESOLVED),
            "notes": notes,
            "active": status["state"] in {"queued", "running"},
        }

    @app.get("/meetings/{identifier}")
    def meeting_page(request: Request, identifier: str) -> Response:
        context = meeting_view(identifier)
        if context["status"]["state"] == "queued" and not launcher.running():
            launcher.ensure()
        return page(request, "meeting.html", {**context, "worker": launcher.running()})

    @app.get("/meetings/{identifier}/audio")
    def audio(identifier: str) -> Response:
        current = state["settings"]
        directory = meeting_dir(current.data_dir, identifier)
        with database(current) as connection:
            formats = json.loads(get_meeting(connection, identifier)["container"])
        media = next((MEDIA_TYPES[f] for f in formats if f in MEDIA_TYPES), "audio/mp4")
        return FileResponse(directory / "original.audio", media_type=media)

    @app.get("/meetings/{identifier}/export.{fmt}")
    def export(identifier: str, fmt: str) -> Response:
        if fmt not in {"md", "json"}:
            raise NotesError("Unknown export format")
        current = state["settings"]
        directory = meeting_dir(current.data_dir, identifier)
        with database(current) as connection, exclusive_lock(lock_path(current, identifier)):
            export_notes(current, connection, identifier)
        media = "text/markdown" if fmt == "md" else "application/json"
        return FileResponse(
            directory / f"notes.{fmt}",
            media_type=media + "; charset=utf-8",
            filename=f"meeting-{identifier}.{fmt}",
        )

    def rerender(connection: sqlite3.Connection, identifier: str) -> None:
        current = state["settings"]
        directory = meeting_dir(current.data_dir, identifier)
        if verified_checkpoint(connection, identifier, directory, "summarizing"):
            with exclusive_lock(lock_path(current, identifier)):
                export_notes(current, connection, identifier)

    @app.post("/meetings/{identifier}/details")
    async def details(request: Request, identifier: str) -> Response:
        form = await read_form(request, guard)

        def run() -> None:
            title = normalize(form.get("title", ""))
            meeting_date = form.get("meeting_date", "").strip()
            if len(title) > 200 or any(not c.isprintable() for c in title):
                raise NotesError("Title must be at most 200 printable characters")
            if meeting_date:
                try:
                    meeting_date = date.fromisoformat(meeting_date).isoformat()
                except ValueError:
                    raise NotesError("Date must be YYYY-MM-DD") from None
            with database(state["settings"]) as connection:
                get_meeting(connection, identifier)
                with transaction(connection):
                    connection.execute(
                        "UPDATE meetings SET title = ?, meeting_date = ? WHERE id = ?",
                        (title or None, meeting_date or None, identifier),
                    )
                rerender(connection, identifier)

        await run_in_threadpool(run)
        return back(f"/meetings/{identifier}")

    def transcript_sha(connection: sqlite3.Connection, identifier: str) -> str:
        directory = meeting_dir(state["settings"].data_dir, identifier)
        found = verified_checkpoint(connection, identifier, directory, "attributing")
        if found is None:
            raise NotesError("Speaker labels exist only after processing")
        return found["transcript.json"]

    @app.post("/meetings/{identifier}/speakers")
    async def speakers(request: Request, identifier: str) -> Response:
        form = await read_form(request, guard)

        def run() -> None:
            with database(state["settings"]) as connection:
                sha = transcript_sha(connection, identifier)
                for label in mapping.labels(connection, identifier):
                    name = normalize(form.get(f"name:{label}", ""))
                    if name:
                        mapping.set_name(connection, identifier, sha, label, name)
                    elif f"name:{label}" in form:
                        mapping.clear_name(connection, identifier, sha, label)
                rerender(connection, identifier)

        await run_in_threadpool(run)
        return back(f"/meetings/{identifier}#speakers")

    @app.post("/meetings/{identifier}/corrections")
    async def corrections(request: Request, identifier: str) -> Response:
        form = await read_form(request, guard)

        def run() -> None:
            with database(state["settings"]) as connection:
                sha = transcript_sha(connection, identifier)
                original = dict(
                    connection.execute(
                        "SELECT segment_id, speaker FROM segments WHERE meeting_id = ?",
                        (identifier,),
                    ).fetchall()
                )
                view = mapping.load_view(connection, identifier, sha)
                for key, chosen in form.items():
                    if not key.startswith("seg:") or key[4:] not in original:
                        continue
                    segment = key[4:]
                    if chosen == original[segment]:
                        if segment in view.overrides:
                            mapping.clear_correction(connection, identifier, sha, segment)
                    elif chosen != view.overrides.get(segment):
                        mapping.correct_segment(connection, identifier, sha, segment, chosen)
                rerender(connection, identifier)

        await run_in_threadpool(run)
        return back(f"/meetings/{identifier}#transcript")

    @app.post("/meetings/{identifier}/process")
    async def process_meeting(request: Request, identifier: str) -> Response:
        form = await read_form(request, guard)
        mode = form.get("mode", "transcribe")
        if mode not in {"transcribe", "align", "compare"}:
            raise NotesError("Unknown processing mode")

        def run() -> None:
            current = state["settings"]
            meeting_dir(current.data_dir, identifier)
            kind = "notes" if form.get("notes_only") == "on" else "full"
            with database(current) as connection:
                enqueue(connection, current, identifier, kind, mode, form.get("force") == "on")
            launcher.ensure()

        await run_in_threadpool(run)
        return back(f"/meetings/{identifier}")

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(request: Request, job_id: int) -> Response:
        form = await read_form(request, guard)

        def run() -> str:
            with database(state["settings"]) as connection:
                meeting_id = connection.execute(
                    "SELECT meeting_id FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                cancel(connection, job_id)
            return meeting_id[0]

        meeting_id = await run_in_threadpool(run)
        target = form.get("next", "")
        return back(f"/meetings/{meeting_id}" if target != "/" else "/")

    @app.get("/settings")
    def settings_page(request: Request, saved: str = "") -> Response:
        current = state["settings"]
        installed: list[str] = []
        with contextlib.suppress(NotesError):
            installed = sorted(
                str(m.get("name"))
                for m in OllamaClient(current).models()
                if not (m.get("remote_model") or m.get("remote_host"))
            )
        return page(
            request,
            "settings.html",
            {
                "values": current.model_dump(mode="json"),
                "editable": EDITABLE,
                "installed": installed,
                "config_path": config_path,
                "saved": saved == "1",
            },
        )

    @app.post("/settings")
    async def settings_submit(request: Request) -> Response:
        form = await read_form(request, guard)
        if config_path is None:
            raise NotesError("Start the server with --config to save settings")
        current = state["settings"]
        values = current.model_dump()
        for name, kind in EDITABLE.items():
            raw = form.get(name)
            if kind is bool:
                values[name] = raw == "on"
            elif raw is not None:
                try:
                    values[name] = kind(raw.strip())
                except ValueError:
                    raise NotesError(f"Invalid value for {name}") from None
        try:
            updated = Settings.model_validate(values)
        except ValidationError as error:
            fields = sorted({str(e["loc"][0]) for e in error.errors() if e["loc"]})
            raise NotesError("Invalid settings: " + (", ".join(fields) or "values")) from None
        await run_in_threadpool(write_config, config_path, updated)
        # Re-read through the normal loader so the saved file is what the server now uses.
        state["settings"] = await run_in_threadpool(load_settings, config_path)
        return back("/settings?saved=1")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "time": time.time(), "worker": launcher.running()}

    return app


def serve(settings: Settings, config_path: Path | None, port: int) -> None:
    """Bind loopback ourselves (no hostname resolution) and run until interrupted."""
    import socket

    import uvicorn

    if not 1024 <= port <= 65535:
        raise NotesError("Port must be between 1024 and 65535")
    if shutil.which("ffprobe") is None:
        raise NotesError("ffprobe is required for imports")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", port))
    except OSError:
        listener.close()
        raise NotesError("Port is in use or unavailable on 127.0.0.1") from None
    app = create_app(settings, config_path, port)
    print(f"Open http://127.0.0.1:{port}/login?token={app.state.guard.login_token}", flush=True)
    print("Listening on loopback only. Ctrl-C stops the server; queued jobs keep running.")
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="critical", access_log=False, server_header=False)
    )
    server.run(sockets=[listener])
