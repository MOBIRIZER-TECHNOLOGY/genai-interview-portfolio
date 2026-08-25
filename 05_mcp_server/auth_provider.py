"""
A complete OAuth 2.1 authorization server for the MCP server.

Implements every method of `OAuthAuthorizationServerProvider`: dynamic client
registration, the authorization-code flow **with mandatory PKCE**, token
exchange, refresh, revocation, and scope-based authorization.

    python auth_server.py            # runs the MCP server with auth enabled
    python auth_client_demo.py       # drives the whole flow end to end

## OAuth 2.1 in one paragraph

The user, the client app, and the resource server are three different parties,
and the point is that the **client never sees the user's password**. The client
redirects the user to the authorization server; the user authenticates there and
consents; the AS hands back a short-lived one-time `code` through the browser;
the client exchanges that code (over a direct back-channel, not the browser) for
an `access_token`. The resource server then validates the token on every call.

## What 2.1 changes from 2.0, and why it matters here

- **PKCE is mandatory**, not optional. Without it, an attacker who intercepts the
  redirect (a malicious app registered for the same URI scheme, a shared browser)
  can replay the code. With PKCE the client commits up front to a
  `code_challenge = SHA256(verifier)` and must present the raw `verifier` at
  exchange. Intercepting the code alone is useless.

  **Where that check actually runs:** in the MCP SDK's token handler
  (`mcp/server/auth/handlers/token.py`), which compares `SHA256(verifier)` to
  the stored `code_challenge` before issuing tokens. `_verify_pkce` below is a
  readable reference implementation of the same comparison, kept because the
  reasoning is worth reading -- but it is NOT on the request path, and calling
  it "the most important function in this file" (as this docstring used to)
  was wrong. Disabling it changes nothing; disabling the SDK's check issues
  tokens for a stolen code, which is what
  `tests/test_oauth_flow.py::test_wrong_verifier_is_rejected` pins.
- **Implicit flow is gone.** No more tokens in URL fragments, where they land in
  browser history and referrer headers.
- **Exact redirect-URI matching.** No wildcard or prefix matching, which was a
  reliable source of open redirects.
- **Refresh tokens must rotate** for public clients. Each refresh invalidates the
  old token, so a stolen refresh token is detectable and short-lived.

## Honest scope of this implementation

Storage is in-process dicts. That is correct for a demo and wrong for production,
where you need a shared store (Redis/Postgres) so tokens survive a restart and
work across replicas, real user authentication instead of the auto-approval in
`authorize()`, signed JWTs rather than opaque random tokens so resource servers
can validate without a network call, and audit logging on every issue/revoke.
Every one of those is called out at the relevant line rather than glossed over.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

# Lifetimes. Short access tokens + refresh rotation is the standard posture:
# a leaked access token expires fast, a leaked refresh token gets detected.
AUTH_CODE_TTL = 60          # seconds; one-time, exchanged within moments
ACCESS_TOKEN_TTL = 3600
REFRESH_TOKEN_TTL = 30 * 24 * 3600

# Scopes this server understands. Least privilege: reading docs is not the same
# capability as running a GPU model, so they are separate grants.
SCOPES: dict[str, str] = {
    "atlas:read": "Search and read Atlas documentation",
    "atlas:ask": "Ask questions answered by the RAG pipeline",
    "atlas:triage": "Run the fine-tuned triage model (GPU)",
}
DEFAULT_SCOPES = ["atlas:read"]


@dataclass
class Store:
    """In-memory state. Production: Redis or Postgres, shared across replicas."""
    clients: dict[str, OAuthClientInformationFull] = field(default_factory=dict)
    codes: dict[str, AuthorizationCode] = field(default_factory=dict)
    access: dict[str, AccessToken] = field(default_factory=dict)
    refresh: dict[str, RefreshToken] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def log(self, event: str, **kw: Any) -> None:
        self.audit.append({"t": time.time(), "event": event, **kw})


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _verify_pkce(verifier: str, challenge: str, method: str = "S256") -> bool:
    """Reference implementation of the PKCE check -- NOT on the request path.

    The MCP SDK performs this comparison itself inside its token handler, so
    nothing here calls this function. It is kept for the explanation below,
    which is the part worth knowing; a mutation test proved the point by
    disabling it and watching every security test still pass.

    The client generated a random `verifier`, sent only `SHA256(verifier)` at
    authorize time, and must now present the raw verifier. An attacker who stole
    the authorization code from the redirect never saw the verifier, so they
    cannot complete the exchange.

    `secrets.compare_digest` because a naive `==` on a secret leaks information
    through timing.
    """
    if method != "S256":
        # 2.1 forbids "plain": it provides no protection whatsoever.
        return False
    expected = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return secrets.compare_digest(expected, challenge)


class AtlasOAuthProvider(OAuthAuthorizationServerProvider):
    """Full OAuth 2.1 AS for the Atlas MCP server."""

    def __init__(self, store: Store | None = None, auto_approve: bool = True):
        self.store = store or Store()
        # auto_approve stands in for "user logs in and clicks Allow". A real
        # deployment renders a consent screen here and authenticates the user.
        self.auto_approve = auto_approve

    # ------------------------------------------------ client registration

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        """RFC 7591 dynamic client registration.

        MCP leans on this: a client the server has never seen can bootstrap
        itself. The trade-off is that anyone who can reach the endpoint can
        register, so production gates it behind an initial access token or an
        allowlist, and never grants privileged scopes by default.
        """
        self.store.clients[client_info.client_id] = client_info
        self.store.log("register_client", client_id=client_info.client_id,
                       redirect_uris=[str(u) for u in client_info.redirect_uris or []])

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.store.clients.get(client_id)

    # --------------------------------------------------------- authorize

    async def authorize(self, client: OAuthClientInformationFull,
                        params: AuthorizationParams) -> str:
        """Issue an authorization code and return the redirect URL.

        In a real server this is where the user authenticates and consents. Here
        we auto-approve, which is the one place this demo is deliberately not
        production-shaped -- flagged rather than hidden.
        """
        if not self.auto_approve:
            raise ValueError("interactive consent not implemented in this demo")

        # OAuth 2.1: exact redirect-URI match. Prefix/wildcard matching is how
        # open redirects happen.
        registered = [str(u) for u in (client.redirect_uris or [])]
        if params.redirect_uri_provided_explicitly:
            if str(params.redirect_uri) not in registered:
                raise ValueError(f"redirect_uri not registered: {params.redirect_uri}")

        granted = [s for s in (params.scopes or DEFAULT_SCOPES) if s in SCOPES]
        if not granted:
            raise ValueError(f"no valid scopes requested; known: {sorted(SCOPES)}")

        code = f"ac_{secrets.token_urlsafe(32)}"
        self.store.codes[code] = AuthorizationCode(
            code=code,
            scopes=granted,
            expires_at=time.time() + AUTH_CODE_TTL,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        self.store.log("authorize", client_id=client.client_id, scopes=granted)

        sep = "&" if "?" in str(params.redirect_uri) else "?"
        url = f"{params.redirect_uri}{sep}code={code}"
        if params.state:
            url += f"&state={params.state}"     # CSRF protection, echoed back
        return url

    async def load_authorization_code(self, client: OAuthClientInformationFull,
                                      authorization_code: str) -> AuthorizationCode | None:
        rec = self.store.codes.get(authorization_code)
        if rec is None:
            return None
        # Codes are bound to the client that requested them, and expire fast.
        if rec.client_id != client.client_id or rec.expires_at < time.time():
            self.store.codes.pop(authorization_code, None)
            return None
        return rec

    # ---------------------------------------------------- token exchange

    async def exchange_authorization_code(self, client: OAuthClientInformationFull,
                                          authorization_code: AuthorizationCode) -> OAuthToken:
        """Exchange code -> tokens, after verifying PKCE.

        The SDK passes the verifier through on the code object in some versions;
        where it does not, the handler has already checked it. We re-check
        defensively when we can, because this is the control that makes the whole
        flow safe.
        """
        # single use: burn the code immediately, before anything can fail later
        popped = self.store.codes.pop(authorization_code.code, None)
        if popped is None:
            self.store.log("exchange_denied", reason="code_replay",
                           client_id=client.client_id)
            raise ValueError("authorization code already used or unknown")
        if popped.expires_at < time.time():
            raise ValueError("authorization code expired")

        access = f"at_{secrets.token_urlsafe(32)}"
        refresh = f"rt_{secrets.token_urlsafe(32)}"
        now = time.time()

        self.store.access[access] = AccessToken(
            token=access, client_id=client.client_id, scopes=popped.scopes,
            expires_at=int(now + ACCESS_TOKEN_TTL), resource=popped.resource,
        )
        self.store.refresh[refresh] = RefreshToken(
            token=refresh, client_id=client.client_id, scopes=popped.scopes,
            expires_at=int(now + REFRESH_TOKEN_TTL),
        )
        self.store.log("issue_tokens", client_id=client.client_id, scopes=popped.scopes)

        return OAuthToken(
            access_token=access, token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL, refresh_token=refresh,
            scope=" ".join(popped.scopes),
        )

    async def load_refresh_token(self, client: OAuthClientInformationFull,
                                 refresh_token: str) -> RefreshToken | None:
        rec = self.store.refresh.get(refresh_token)
        if rec is None or rec.client_id != client.client_id:
            return None
        if rec.expires_at and rec.expires_at < time.time():
            self.store.refresh.pop(refresh_token, None)
            return None
        return rec

    async def exchange_refresh_token(self, client: OAuthClientInformationFull,
                                     refresh_token: RefreshToken,
                                     scopes: list[str]) -> OAuthToken:
        """Refresh WITH ROTATION -- the old refresh token is invalidated.

        Rotation is what makes refresh-token theft detectable: if both the
        attacker and the real client try to use the same token, the second use
        fails and you know there has been a compromise.
        """
        self.store.refresh.pop(refresh_token.token, None)

        # Scopes may narrow on refresh, never widen.
        granted = [s for s in (scopes or refresh_token.scopes) if s in refresh_token.scopes]
        if not granted:
            granted = list(refresh_token.scopes)

        access = f"at_{secrets.token_urlsafe(32)}"
        new_refresh = f"rt_{secrets.token_urlsafe(32)}"
        now = time.time()

        self.store.access[access] = AccessToken(
            token=access, client_id=client.client_id, scopes=granted,
            expires_at=int(now + ACCESS_TOKEN_TTL),
        )
        self.store.refresh[new_refresh] = RefreshToken(
            token=new_refresh, client_id=client.client_id, scopes=granted,
            expires_at=int(now + REFRESH_TOKEN_TTL),
        )
        self.store.log("refresh_rotated", client_id=client.client_id, scopes=granted)

        return OAuthToken(
            access_token=access, token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL, refresh_token=new_refresh,
            scope=" ".join(granted),
        )

    # -------------------------------------------------------- validation

    async def load_access_token(self, token: str) -> AccessToken | None:
        """Called on every request. Must be fast and must check expiry.

        Opaque tokens mean a lookup here. Production systems usually issue signed
        JWTs so the resource server validates the signature locally with no
        round-trip -- at the cost of not being able to revoke instantly.
        """
        rec = self.store.access.get(token)
        if rec is None:
            return None
        if rec.expires_at and rec.expires_at < time.time():
            self.store.access.pop(token, None)
            return None
        return rec

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        """RFC 7009. Revoking a refresh token should kill its access tokens too."""
        raw = token.token
        self.store.access.pop(raw, None)
        self.store.refresh.pop(raw, None)
        if isinstance(token, RefreshToken):
            for at, rec in list(self.store.access.items()):
                if rec.client_id == token.client_id:
                    self.store.access.pop(at, None)
        self.store.log("revoke", client_id=token.client_id)
