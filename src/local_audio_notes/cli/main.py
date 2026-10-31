import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from local_audio_notes.config import load_settings
from local_audio_notes.errors import NotesError
from local_audio_notes.ingestion.importer import ingest
from local_audio_notes.llm.ollama import OllamaClient
from local_audio_notes.models import (
    KINDS,
    install_assets,
    install_model,
    register_model,
    verify_models,
)
from local_audio_notes.pipeline import load_meeting, process
from local_audio_notes.security.runtime import enable_runtime, isolated_network


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Local audio notes — Phase 1 CLI")
    cli.add_argument("--config", type=Path, help="TOML configuration (no automatic .env loading)")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect installed tools without changing the machine")
    sub.add_parser("models-check", help="Verify required local model files and hashes")
    sub.add_parser("ollama-models", help="List installed models from localhost only")
    offline = sub.add_parser("offline-check", help="Enable application guards; check OS isolation")
    offline.add_argument("--enforced", action="store_true")
    importer = sub.add_parser("ingest", help="Copy and validate audio; prints meeting ID")
    importer.add_argument("audio", type=Path)
    importer.add_argument("--transcript", type=Path)
    runner = sub.add_parser("run", help="Process imported audio or retry only notes")
    runner.add_argument("meeting_id")
    runner.add_argument("--mode", choices=["transcribe", "align", "compare"], default="transcribe")
    runner.add_argument("--notes-only", action="store_true")
    status = sub.add_parser("status", help="Show stage and metrics, never transcript text")
    status.add_argument("meeting_id")
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
            # Third-party ML libraries occasionally print file paths or transcript snippets.
            # Keep stdout/stderr quiet while they run; safe errors are handled below.
            with (
                open(os.devnull, "w") as sink,
                contextlib.redirect_stdout(sink),
                contextlib.redirect_stderr(sink),
            ):
                process(settings, args.meeting_id, args.mode, args.notes_only)
            print("Complete: notes.md, notes.json and transcript.json in the meeting directory")
        elif args.command == "status":
            _, metadata = load_meeting(settings, args.meeting_id)
            print(
                json.dumps(
                    {
                        k: metadata[k]
                        for k in ("id", "state", "failed_stage", "error", "timings")
                        if k in metadata
                    },
                    indent=2,
                )
            )
        elif args.command == "models-check":
            print(json.dumps({"verified": list(verify_models(settings.models_dir))}))
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
