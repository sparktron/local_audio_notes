import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from local_audio_notes.config import Settings, load_settings
from local_audio_notes.errors import Interrupted, NotesError
from local_audio_notes.extraction.grounding import normalize
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.jobs.queue import (
    cancel,
    describe,
    enqueue,
    lock_path,
    process,
    recover,
    work,
)
from local_audio_notes.llm.ollama import OllamaClient
from local_audio_notes.models import (
    KINDS,
    install_assets,
    install_model,
    register_model,
    verify_models,
)
from local_audio_notes.pipeline import export_notes, get_meeting, reindex, verified_checkpoint
from local_audio_notes.schemas import CATEGORIES
from local_audio_notes.search.index import MARK_END, MARK_START, search
from local_audio_notes.security.runtime import enable_runtime, isolated_network
from local_audio_notes.speakers import mapping
from local_audio_notes.storage.db import connect, transaction
from local_audio_notes.storage.files import meeting_dir
from local_audio_notes.storage.locking import exclusive_lock


def _pair(value: str) -> tuple[str, str]:
    key, separator, other = value.partition("=")
    if not separator or not key.strip() or not other.strip():
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    return key.strip(), other.strip()


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Local audio notes CLI")
    cli.add_argument("--config", type=Path, help="TOML configuration (no automatic .env loading)")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect installed tools without changing the machine")
    check = sub.add_parser("models-check", help="Verify required local model files and hashes")
    check.add_argument(
        "--full", action="store_true", help="Rehash every file even if the hash cache matches"
    )
    sub.add_parser("ollama-models", help="List installed models from localhost only")
    offline = sub.add_parser("offline-check", help="Enable application guards; check OS isolation")
    offline.add_argument("--enforced", action="store_true")
    importer = sub.add_parser("ingest", help="Copy and validate audio; prints meeting ID")
    importer.add_argument("audio", type=Path)
    importer.add_argument("--transcript", type=Path)
    for name, text in (
        ("run", "Queue and process a meeting now, resuming from verified checkpoints"),
        ("enqueue", "Queue processing for a later worker run; prints the job ID"),
    ):
        runner = sub.add_parser(name, help=text)
        runner.add_argument("meeting_id")
        runner.add_argument(
            "--mode", choices=["transcribe", "align", "compare"], default="transcribe"
        )
        runner.add_argument(
            "--notes-only", action="store_true", help="Reuse the attributed transcript"
        )
        runner.add_argument("--force", action="store_true", help="Ignore stage checkpoints")
    worker = sub.add_parser("worker", help="Recover abandoned jobs, then run queued jobs")
    worker.add_argument("--once", action="store_true", help="Run at most one job")
    jobs = sub.add_parser("jobs", help="List recent jobs (no transcript content)")
    jobs.add_argument("--limit", type=int, default=20)
    stop = sub.add_parser("cancel", help="Cancel a queued job or stop a running one safely")
    stop.add_argument("job_id", type=int)
    sub.add_parser("recover", help="Requeue jobs whose worker died; adopt orphan imports")
    status = sub.add_parser("status", help="Show stage and metrics, never transcript text")
    status.add_argument("meeting_id")
    meta = sub.add_parser("meeting-set", help="Set or clear the meeting title and date")
    meta.add_argument("meeting_id")
    meta.add_argument("--title")
    meta.add_argument("--date", help="YYYY-MM-DD")
    meta.add_argument("--clear-title", action="store_true")
    meta.add_argument("--clear-date", action="store_true")
    speakers = sub.add_parser(
        "speakers", help="Show or confirm speaker names and segment corrections for one meeting"
    )
    speakers.add_argument("meeting_id")
    speakers.add_argument(
        "--name", type=_pair, action="append", default=[], metavar="'Speaker N=Name'"
    )
    speakers.add_argument("--unname", action="append", default=[], metavar="'Speaker N'")
    speakers.add_argument(
        "--assign", type=_pair, action="append", default=[], metavar="'s000123=Speaker N'"
    )
    speakers.add_argument("--unassign", action="append", default=[], metavar="s000123")
    finder = sub.add_parser("search", help="Search titles, transcripts and extracted items")
    finder.add_argument("query")
    finder.add_argument("--participant", default="", help="Speaker label or confirmed name")
    finder.add_argument(
        "--category", default="", choices=["", *CATEGORIES], help="Only extracted items"
    )
    finder.add_argument("--from", dest="date_from", default="", help="Meeting date YYYY-MM-DD")
    finder.add_argument("--to", dest="date_to", default="", help="Meeting date YYYY-MM-DD")
    finder.add_argument("--semantic", action="store_true", help="Local embedding similarity")
    finder.add_argument("--limit", type=int, default=20)
    rebuild = sub.add_parser("reindex", help="Rebuild search indexes from verified artifacts")
    rebuild.add_argument(
        "--embeddings", action="store_true", help="Also recompute local embeddings"
    )
    web = sub.add_parser("serve", help="Run the web interface on 127.0.0.1 only")
    web.add_argument("--port", type=int, default=8765)
    export = sub.add_parser("export", help="Re-render notes; optionally copy them to a new file")
    export.add_argument("meeting_id")
    export.add_argument("--format", choices=["md", "json"], default="md")
    export.add_argument("--output", type=Path, help="New file to create (never overwritten)")
    install = sub.add_parser("models-install", help="Explicit network-enabled model installation")
    install.add_argument("--kind", required=True, choices=KINDS[:3])
    install.add_argument("--repo", required=True)
    install.add_argument("--revision", required=True, help="Full immutable 40-character commit SHA")
    install.add_argument("--allow-download", action="store_true")
    assets = sub.add_parser(
        "assets-install", help="Explicit tokenizer download / bundled VAD inventory"
    )
    assets.add_argument("--allow-download", action="store_true")
    register = sub.add_parser(
        "models-register", help="Inventory trusted, materialized local assets"
    )
    register.add_argument("--kind", required=True, choices=KINDS)
    register.add_argument("--path", required=True, type=Path)
    register.add_argument("--revision", required=True)
    return cli


def doctor() -> dict:
    tools = {
        name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "nvidia-smi", "nvcc", "ollama")
    }
    packages = {}
    for name in ("torch", "whisperx", "pyannote.audio", "silero-vad", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = "not available"
    if tools["nvidia-smi"]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
        )
        gpu = result.stdout.strip() if result.returncode == 0 else "driver query failed"
    return {
        "python": platform.python_version(),
        "tools": tools,
        "packages": packages,
        "gpu": gpu,
        "note": "Driver detection is not a GPU inference test",
    }


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    # Third-party ML libraries occasionally print file paths or transcript snippets.
    # Keep stdout/stderr quiet while they run; safe errors are reported afterwards.
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        yield


@contextlib.contextmanager
def requeue_on_sigterm() -> Iterator[None]:
    """A service stop returns the running job to the queue instead of failing it."""

    def terminate(signum: int, frame: Any) -> None:
        raise Interrupted

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _transcript_sha(settings: Settings, connection: sqlite3.Connection, meeting_id: str) -> str:
    found = verified_checkpoint(
        connection, meeting_id, meeting_dir(settings.data_dir, meeting_id), "attributing"
    )
    if found is None:
        raise NotesError("Speaker labels exist only after processing; run the meeting first")
    return found["transcript.json"]


def _rerender(settings: Settings, connection: sqlite3.Connection, meeting_id: str) -> bool:
    """Refresh existing notes after metadata/speaker edits; no model calls."""
    directory = meeting_dir(settings.data_dir, meeting_id)
    if verified_checkpoint(connection, meeting_id, directory, "summarizing") is None:
        return False
    with exclusive_lock(lock_path(settings, meeting_id)):
        export_notes(settings, connection, meeting_id)
    return True


def _iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise NotesError("Date must be YYYY-MM-DD") from None


def _meeting_set(args: argparse.Namespace, settings: Settings, connection: Any) -> None:
    get_meeting(connection, args.meeting_id)
    updates: dict[str, str | None] = {}
    if args.title is not None:
        title = normalize(args.title)
        if not 1 <= len(title) <= 200 or any(not c.isprintable() for c in title):
            raise NotesError("Title must be 1-200 printable characters")
        updates["title"] = title
    if args.date is not None:
        try:
            updates["meeting_date"] = date.fromisoformat(args.date).isoformat()
        except ValueError:
            raise NotesError("Date must be YYYY-MM-DD") from None
    if args.clear_title:
        updates["title"] = None
    if args.clear_date:
        updates["meeting_date"] = None
    with transaction(connection):
        for column, value in updates.items():
            connection.execute(
                f"UPDATE meetings SET {column} = ? WHERE id = ?", (value, args.meeting_id)
            )
    print("Notes re-rendered" if _rerender(settings, connection, args.meeting_id) else "Saved")


def _speakers(args: argparse.Namespace, settings: Settings, connection: Any) -> None:
    meeting_id = args.meeting_id
    get_meeting(connection, meeting_id)
    transcript_sha = _transcript_sha(settings, connection, meeting_id)
    for label, name in args.name:
        mapping.set_name(connection, meeting_id, transcript_sha, label, name)
    for label in args.unname:
        mapping.clear_name(connection, meeting_id, transcript_sha, label)
    for segment_id, label in args.assign:
        mapping.correct_segment(connection, meeting_id, transcript_sha, segment_id, label)
    for segment_id in args.unassign:
        mapping.clear_correction(connection, meeting_id, transcript_sha, segment_id)
    view = mapping.load_view(connection, meeting_id, transcript_sha)
    changed = args.name or args.unname or args.assign or args.unassign
    print(
        json.dumps(
            {
                "labels": sorted(mapping.labels(connection, meeting_id)),
                "confirmed_names": view.names,
                "segment_corrections": view.overrides,
                "stale_entries_not_applied": view.stale_entries,
                "notes_rerendered": bool(changed) and _rerender(settings, connection, meeting_id),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _export(args: argparse.Namespace, settings: Settings, connection: Any) -> None:
    get_meeting(connection, args.meeting_id)
    with exclusive_lock(lock_path(settings, args.meeting_id)):
        export_notes(settings, connection, args.meeting_id)
    source = meeting_dir(settings.data_dir, args.meeting_id) / f"notes.{args.format}"
    if args.output is None:
        print(source)
        return
    try:
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
    except OSError:
        raise NotesError(
            "Output must be a new file path; existing files are never overwritten"
        ) from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(source.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    print("Exported")


def _queue_command(args: argparse.Namespace, settings: Settings) -> int:
    with contextlib.closing(connect(settings.data_dir)) as connection:
        if args.command == "enqueue":
            kind = "notes" if args.notes_only else "full"
            print(enqueue(connection, settings, args.meeting_id, kind, args.mode, args.force))
            return 0
        if args.command == "worker":
            with requeue_on_sigterm(), quiet():
                results = work(
                    settings, connection, once=args.once, reload=lambda: load_settings(args.config)
                )
            for job_id, outcome in results:
                print(json.dumps({"job": job_id, "outcome": outcome}))
            return 1 if any(outcome != "succeeded" for _, outcome in results) else 0
        if args.command == "jobs":
            rows = connection.execute(
                "SELECT id, meeting_id, kind, mode, state, stage, attempts, error_stage, "
                "error_class, created_at, finished_at FROM jobs ORDER BY id DESC LIMIT ?",
                (max(1, min(args.limit, 1000)),),
            )
            for row in rows:
                print(json.dumps(dict(row)))
        elif args.command == "cancel":
            print(cancel(connection, args.job_id))
        elif args.command == "recover":
            print(json.dumps(recover(connection, settings)))
        elif args.command == "status":
            meeting_dir(settings.data_dir, args.meeting_id)
            print(json.dumps(describe(connection, args.meeting_id), indent=2))
        elif args.command == "meeting-set":
            _meeting_set(args, settings, connection)
        elif args.command == "speakers":
            _speakers(args, settings, connection)
        elif args.command == "export":
            _export(args, settings, connection)
        elif args.command == "search":
            for date_value in (args.date_from, args.date_to):
                if date_value:
                    _iso_date(date_value)
            hits = search(
                connection,
                settings,
                args.query,
                participant=args.participant,
                category=args.category,
                date_from=args.date_from,
                date_to=args.date_to,
                semantic=args.semantic,
                embedder=OllamaClient(settings) if args.semantic else None,
                limit=args.limit,
            )
            for hit in hits:
                value = hit.as_dict()
                value["snippet"] = value["snippet"].replace(MARK_START, "[").replace(MARK_END, "]")
                print(json.dumps(value, ensure_ascii=False))
        elif args.command == "reindex":
            embedder = OllamaClient(settings) if args.embeddings else None
            if args.embeddings and not settings.embedding_model:
                raise NotesError("Set embedding_model to an installed local model first")
            print(json.dumps(reindex(settings, connection, embedder)))
    return 0


QUEUE_COMMANDS = {
    "search",
    "reindex",
    "enqueue",
    "worker",
    "jobs",
    "cancel",
    "recover",
    "status",
    "meeting-set",
    "speakers",
    "export",
}


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        settings = load_settings(args.config)
        setup = args.command in {"models-install", "assets-install"}
        if setup:
            if os.geteuid() == 0:
                raise NotesError("Run setup as an unprivileged user")
            os.umask(0o077)
            os.environ.update(
                {
                    "PYANNOTE_METRICS_ENABLED": "0",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "OTEL_SDK_DISABLED": "true",
                    "DO_NOT_TRACK": "1",
                }
            )
        else:
            enable_runtime(
                enforced=settings.offline_enforced
                or (args.command == "offline-check" and args.enforced)
            )
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
        elif args.command == "ingest":
            print(ingest(args.audio, settings, args.transcript))
        elif args.command == "run":
            with requeue_on_sigterm(), quiet():
                process(settings, args.meeting_id, args.mode, args.notes_only, args.force)
            print("Complete: notes.md, notes.json and transcript.json in the meeting directory")
        elif args.command == "serve":
            try:
                from local_audio_notes.api.app import serve
            except ImportError:
                raise NotesError(
                    "Web dependencies missing; install with: .venv/bin/pip install -e '.[web]'"
                ) from None
            serve(settings, args.config.resolve() if args.config else None, args.port)
        elif args.command in QUEUE_COMMANDS:
            return _queue_command(args, settings)
        elif args.command == "models-check":
            cache = settings.hash_cache and not args.full
            print(json.dumps({"verified": list(verify_models(settings.models_dir, cache=cache))}))
        elif args.command == "ollama-models":
            for model in OllamaClient(settings).models():
                print(
                    json.dumps(
                        {
                            "name": model.get("name"),
                            "digest": model.get("digest"),
                            "size": model.get("size"),
                            "remote": bool(model.get("remote_model") or model.get("remote_host")),
                        }
                    )
                )
        elif args.command == "offline-check":
            print(
                json.dumps(
                    {
                        "application_guard": True,
                        "isolated_namespace": isolated_network(),
                        "native_egress_tested": False,
                    }
                )
            )
        elif args.command == "models-register":
            register_model(settings.models_dir, args.kind, args.path, args.revision, "manual-local")
            print(
                "Local asset hashes recorded; trust and license review remain your responsibility"
            )
        elif args.command == "models-install":
            install_model(
                settings.models_dir, args.kind, args.repo, args.revision, args.allow_download
            )
            print("Model installed and artifact hashes recorded")
        elif args.command == "assets-install":
            install_assets(settings.models_dir, args.allow_download)
            print("Tokenizer and VAD assets inventoried")
        return 0
    except NotesError as error:
        print(f"Error: {error}", file=sys.stderr)
    except ValidationError:
        print(
            "Error: configuration or artifact validation failed (content suppressed)",
            file=sys.stderr,
        )
    except KeyboardInterrupt:
        print("Interrupted; original recordings and completed artifacts retained", file=sys.stderr)
        return 130
    except Interrupted:
        print("Stopped by signal; the job was returned to the queue", file=sys.stderr)
        return 143
    except Exception as error:
        # Native library exceptions may contain transcript excerpts, paths or tokens.
        print(
            f"Error: {type(error).__name__}; details suppressed for privacy. "
            "Check status and docs/TROUBLESHOOTING.md.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
