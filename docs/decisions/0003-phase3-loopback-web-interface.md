# ADR-0003: Loopback web interface security model

- **Date:** 2026-10-31
- **Status:** accepted for Phase 3

## Context

Phase 3 adds a browser interface (dashboard, import, transcript viewer, speaker correction,
notes and export, settings). The default deployment must not expose an unauthenticated
interface beyond loopback. A loopback port is still reachable by every local user and,
indirectly, by any web page the user visits (cross-site requests, DNS rebinding).

## Decisions

1. **Bind loopback ourselves and fail closed.** `serve` binds `127.0.0.1` with its own
   socket (no hostname resolution, so the Python network guard stays active) and has no
   option to listen elsewhere. Remote access is out of scope; use SSH port forwarding.
2. **Per-launch sign-in link.** A random token printed at startup is exchanged at `/login`
   for an HttpOnly, SameSite=Strict session cookie held in server memory. Other local users
   cannot use the port without the token. A restart invalidates all sessions.
3. **Cross-site defences on every request.** Host must be `127.0.0.1` or `localhost`
   (DNS rebinding); unsafe methods are refused when `Origin` or `Sec-Fetch-Site` shows another
   site; every form and upload carries a per-session CSRF token. No CORS headers are sent.
4. **No external or inline content.** Strict CSP (`default-src 'none'`, same-origin scripts,
   styles and media, `frame-ancestors 'none'`), one same-origin JavaScript file, Jinja2
   autoescaping, FastAPI's CDN-backed `/docs` and OpenAPI disabled, FastAPI telemetry disabled.
5. **Uploads stream raw bytes** (`application/octet-stream`, not multipart) into a private
   `.uploads` directory with a hard byte limit, then go through the same content-validated
   `ingest`. Staged copies are deleted after import and at startup. No multipart parser or
   system temp directory is involved.
6. **Processing runs in a separate `worker` process**, started with a fixed argument array
   when jobs are queued. Model memory is released when the queue drains, the server stays
   responsive, and the CLI's durable queue and recovery apply unchanged.
7. **Settings editable from the browser are limited** to transcription, GPU, diarization and
   LLM-selection fields. Storage paths, the Ollama URL, offline enforcement and size limits
   are startup-only, so a forged request could not redirect data or disable protections.

## Alternatives considered

- Unauthenticated loopback (common for local tools): any local user or rebinding attack
  could read transcripts.
- Running the pipeline in a server thread: ML libraries print to process-wide streams and
  hold GPU memory for the server's lifetime.
- Multipart forms: Starlette spools large parts to the system temp directory.

## Consequences

Uploading needs JavaScript (one small file); the CLI remains for scripted or no-JS use.
The sign-in token stays in browser history for that launch. Same-user processes can still
read the token from the terminal or the database directly; they are outside the threat model.
Saving settings rewrites the configuration file without comments.
