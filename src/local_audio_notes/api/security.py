"""Request guard for the loopback web interface.

Threats addressed: other local users or processes reaching the loopback port (per-launch
login token -> session cookie), malicious web pages the user visits (Host checks against
DNS rebinding, Origin/Sec-Fetch-Site checks and CSRF tokens on every state change,
SameSite=Strict cookies, no CORS) and content injection (strict CSP, no inline scripts,
autoescaped templates). Not addressed: a malicious process running as the same user.
"""

import hmac
import secrets
from urllib.parse import parse_qs

from starlette.requests import Request

from local_audio_notes.errors import NotesError

COOKIE = "lan_session"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
FORM_LIMIT = 2 * 1024**2
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
        "media-src 'self'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; "
        "base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class Guard:
    def __init__(self, port: int, login_token: str | None = None):
        self.port = port
        self.login_token = login_token or secrets.token_urlsafe(32)
        self.origins = {f"http://{host}:{port}" for host in LOOPBACK_HOSTS}
        # session id -> CSRF token. In memory: a restart signs everyone out.
        self.sessions: dict[str, str] = {}

    def login(self, token: str) -> str | None:
        if not hmac.compare_digest(token.encode(), self.login_token.encode()):
            return None
        session = secrets.token_urlsafe(32)
        self.sessions[session] = secrets.token_urlsafe(32)
        return session

    def session(self, request: Request) -> str | None:
        session = request.cookies.get(COOKIE)
        return session if session in self.sessions else None

    def csrf(self, request: Request) -> str:
        session = self.session(request)
        return self.sessions[session] if session else ""

    def cross_site(self, request: Request) -> bool:
        origin = request.headers.get("origin")
        site = request.headers.get("sec-fetch-site")
        return bool(
            (origin is not None and origin not in self.origins)
            or (site is not None and site not in {"same-origin", "none"})
        )

    def check_csrf(self, request: Request, supplied: str | None) -> None:
        expected = self.csrf(request)
        if not expected or not supplied or not hmac.compare_digest(supplied, expected):
            raise NotesError("Request verification failed; reload the page and try again")


async def read_form(request: Request, guard: Guard) -> dict[str, str]:
    """Bounded application/x-www-form-urlencoded parsing with CSRF verification."""
    if request.headers.get("content-type", "").split(";")[0] != (
        "application/x-www-form-urlencoded"
    ):
        raise NotesError("Unsupported form encoding")
    body = bytearray()
    async for block in request.stream():
        body.extend(block)
        if len(body) > FORM_LIMIT:
            raise NotesError("Form too large")
    try:
        parsed = parse_qs(
            body.decode("utf-8"), keep_blank_values=True, strict_parsing=False, max_num_fields=20000
        )
    except (UnicodeDecodeError, ValueError):
        raise NotesError("Invalid form data") from None
    form = {key: values[-1] for key, values in parsed.items()}
    guard.check_csrf(request, form.get("csrf"))
    return form
