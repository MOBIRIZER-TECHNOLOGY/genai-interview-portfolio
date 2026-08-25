"""
The OAuth 2.1 flow, ported from a demo script into assertions.

`05_mcp_server/auth_client_demo.py` walks the whole flow and prints it, which is
excellent for showing a human what happens and useless as a regression test:
nothing fails. The security properties it demonstrates are exactly the ones you
never want to discover are broken -- a PKCE check that stopped verifying, a
refresh token that stayed valid after rotation, a revoked token that still works.

So each printed "must not be 200" becomes an assertion here.

Cost control: `05_mcp_server/server.py` lazy-loads the RAG pipeline and the LoRA
adapter, so the auth surface -- registration, authorize, token, refresh,
revocation, scope enforcement -- can be exercised without loading a model or
touching the GPU. The two tool calls used are `whoami` (pure token
introspection) and a scope denial that is rejected *before* any model would
load. That keeps this file in the default CI subset where it belongs: an auth
regression should fail in seconds, on any machine.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

httpx = pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "05_mcp_server" / "auth_server.py"

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base_url():
    """Spawn the OAuth-protected MCP server on a free port; tear it down after."""
    port = _free_port()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), "--port", str(port)],
        cwd=str(SERVER.parent), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 90
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"auth server exited early:\n{out[-2000:]}")
            try:
                r = httpx.get(f"{url}/.well-known/oauth-authorization-server", timeout=2)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            pytest.fail("auth server did not become ready within 90s")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def metadata(base_url):
    return httpx.get(f"{base_url}/.well-known/oauth-authorization-server", timeout=10).json()


def _register(base_url, scopes: str = "atlas:read") -> dict:
    """RFC 7591 dynamic client registration.

    The `scope` field matters: a client may only later *request* scopes it
    registered for. Omitting it here made /authorize refuse `atlas:ask` with no
    code in the redirect -- which is the registry doing its job, and worth a
    comment because the failure surfaces two steps later as a missing code.
    """
    r = httpx.post(f"{base_url}/register", timeout=10, json={
        "client_name": "pytest-client",
        "redirect_uris": ["http://localhost:9999/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "client_secret_post",
        "scope": scopes,
    })
    assert r.status_code in (200, 201), r.text
    return r.json()


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    return verifier, challenge


def _authorize(base_url, client: dict, challenge: str, scopes: str) -> str:
    """Drive /authorize and pull the one-time code out of the redirect."""
    r = httpx.get(f"{base_url}/authorize", timeout=10, follow_redirects=False, params={
        "response_type": "code",
        "client_id": client["client_id"],
        "redirect_uri": "http://localhost:9999/callback",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": scopes,
        "state": "pytest-state",
    })
    assert r.status_code in (302, 307), f"expected a redirect, got {r.status_code}"
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q.get("state") == ["pytest-state"], "state must be echoed back unchanged"
    return q["code"][0]


def _token(base_url, client: dict, **form) -> httpx.Response:
    return httpx.post(f"{base_url}/token", timeout=10, data={
        "client_id": client["client_id"],
        "client_secret": client["client_secret"],
        **form,
    })


@pytest.fixture(scope="module")
def granted(base_url):
    """A live access/refresh pair with two of the three scopes."""
    client = _register(base_url, "atlas:read atlas:ask")
    verifier, challenge = _pkce()
    code = _authorize(base_url, client, challenge, "atlas:read atlas:ask")
    r = _token(base_url, client, grant_type="authorization_code", code=code,
               redirect_uri="http://localhost:9999/callback", code_verifier=verifier)
    assert r.status_code == 200, r.text
    return {"client": client, **r.json()}


# ----------------------------------------------------------------- discovery


def test_metadata_advertises_s256_only(metadata):
    """OAuth 2.1 removes `plain`. Advertising it would invite a downgrade."""
    assert metadata["code_challenge_methods_supported"] == ["S256"]


def test_metadata_advertises_the_three_scopes(metadata):
    assert set(metadata["scopes_supported"]) == {"atlas:read", "atlas:ask", "atlas:triage"}


def test_metadata_has_no_implicit_or_password_grant(metadata):
    """Both grants are forbidden in OAuth 2.1 -- their absence is the point."""
    grants = set(metadata.get("grant_types_supported", []))
    assert grants == {"authorization_code", "refresh_token"}
    assert "token" not in metadata.get("response_types_supported", [])


# ------------------------------------------------------------------- PKCE


def test_wrong_verifier_is_rejected(base_url):
    """THE PKCE property: a stolen code is useless without the verifier.

    This is the assertion the demo script prints as "must not be 200". If this
    ever returns tokens, an intercepted authorization code becomes a full
    account takeover and nothing else in the flow would notice.
    """
    client = _register(base_url)
    _verifier, challenge = _pkce()
    code = _authorize(base_url, client, challenge, "atlas:read")
    wrong, _ = _pkce()                      # a valid verifier, for a different challenge
    r = _token(base_url, client, grant_type="authorization_code", code=code,
               redirect_uri="http://localhost:9999/callback", code_verifier=wrong)
    assert r.status_code == 400, f"PKCE did not reject a bad verifier: {r.status_code} {r.text}"


def test_authorization_code_is_single_use(base_url):
    """A replayed code must fail even with the CORRECT verifier."""
    client = _register(base_url)
    verifier, challenge = _pkce()
    code = _authorize(base_url, client, challenge, "atlas:read")
    first = _token(base_url, client, grant_type="authorization_code", code=code,
                   redirect_uri="http://localhost:9999/callback", code_verifier=verifier)
    assert first.status_code == 200, first.text
    second = _token(base_url, client, grant_type="authorization_code", code=code,
                    redirect_uri="http://localhost:9999/callback", code_verifier=verifier)
    assert second.status_code == 400, "an authorization code must not be reusable"


def test_token_carries_only_the_requested_scopes(granted):
    assert set(granted["scope"].split()) == {"atlas:read", "atlas:ask"}
    assert "atlas:triage" not in granted["scope"]


# --------------------------------------------------------- refresh rotation


def test_refresh_rotates_and_invalidates_the_old_token(base_url):
    """Rotation is what makes refresh-token theft detectable."""
    client = _register(base_url)
    verifier, challenge = _pkce()
    code = _authorize(base_url, client, challenge, "atlas:read")
    tok = _token(base_url, client, grant_type="authorization_code", code=code,
                 redirect_uri="http://localhost:9999/callback", code_verifier=verifier).json()

    rotated = _token(base_url, client, grant_type="refresh_token",
                     refresh_token=tok["refresh_token"])
    assert rotated.status_code == 200, rotated.text
    new = rotated.json()
    assert new["refresh_token"] != tok["refresh_token"], "refresh token must rotate"
    assert new["access_token"] != tok["access_token"]

    replay = _token(base_url, client, grant_type="refresh_token",
                    refresh_token=tok["refresh_token"])
    assert replay.status_code == 400, "the OLD refresh token must stop working after rotation"


# --------------------------------------------------------------- the resource


def _mcp_call(base_url, token: str | None, tool: str, args: dict | None = None):
    """Minimal MCP streamable-HTTP call: initialize, then tools/call."""
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    init = httpx.post(f"{base_url}/mcp", timeout=30, headers=headers, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "pytest", "version": "1"}}})
    if init.status_code != 200:
        return init, None
    sid = init.headers.get("Mcp-Session-Id")
    headers["Mcp-Session-Id"] = sid
    httpx.post(f"{base_url}/mcp", timeout=30, headers=headers, json={
        "jsonrpc": "2.0", "method": "notifications/initialized"})
    call = httpx.post(f"{base_url}/mcp", timeout=60, headers=headers, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": args or {}}})
    return call, sid


def test_tool_call_without_a_token_is_unauthorized(base_url):
    r, _ = _mcp_call(base_url, None, "whoami")
    assert r.status_code == 401, "an unauthenticated tool call must be refused"


def test_revoked_token_stops_working(base_url):
    """Revocation must take effect immediately -- the cost of opaque tokens."""
    client = _register(base_url)
    verifier, challenge = _pkce()
    code = _authorize(base_url, client, challenge, "atlas:read")
    tok = _token(base_url, client, grant_type="authorization_code", code=code,
                 redirect_uri="http://localhost:9999/callback", code_verifier=verifier).json()

    ok, _ = _mcp_call(base_url, tok["access_token"], "whoami")
    assert ok.status_code == 200, "token should work before revocation"

    rev = httpx.post(f"{base_url}/revoke", timeout=10, data={
        "token": tok["access_token"], "client_id": client["client_id"],
        "client_secret": client["client_secret"]})
    assert rev.status_code == 200, rev.text

    after, _ = _mcp_call(base_url, tok["access_token"], "whoami")
    assert after.status_code == 401, "a revoked token must not be accepted"


def test_out_of_scope_tool_is_denied_and_names_the_scope(granted, base_url):
    """Least privilege at the TOOL boundary, not just the server boundary.

    The denial is also checked before any model loads, which is why this test
    costs milliseconds instead of a GPU warm-up.
    """
    r, _ = _mcp_call(base_url, granted["access_token"], "triage_incident",
                     {"report": "atlas-vision is down"})
    body = r.text
    assert "atlas:triage" in body, f"denial should name the missing scope: {body[:400]}"
    assert "requires" in body.lower()
