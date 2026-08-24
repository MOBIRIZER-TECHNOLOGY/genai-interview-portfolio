"""
Drive the complete OAuth 2.1 flow against the authenticated MCP server, by hand.

    python auth_server.py            # terminal 1
    python auth_client_demo.py       # terminal 2

Every HTTP call is made explicitly with `httpx` rather than hidden behind an
SDK helper, because the point is to *see* the flow:

    1. call a tool with no token          -> 401 + WWW-Authenticate
    2. discover the AS from that header   -> /.well-known/oauth-*
    3. register a client dynamically      -> client_id (RFC 7591)
    4. generate PKCE verifier/challenge   -> S256
    5. /authorize                         -> one-time code
    6. exchange code + verifier           -> access + refresh tokens
    7. call tools with the token          -> 200
    8. call a tool outside your scopes    -> denied
    9. refresh (with rotation)            -> new pair, old refresh dead
   10. revoke                             -> token stops working

Steps 8, 9 and 10 are the ones worth watching: they are what separates "I turned
auth on" from "I understand what auth is doing".
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import secrets
import sys
from urllib.parse import parse_qs, urlparse

import httpx

BASE = "http://127.0.0.1:8765"


def h(title: str) -> None:
    print(f"\n{'=' * 76}\n  {title}\n{'=' * 76}")


def pkce_pair() -> tuple[str, str]:
    """(verifier, challenge). The verifier never leaves the client until exchange."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode().rstrip("=")
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode().rstrip("=")
    return verifier, challenge


def _headers(token: str | None, session: str | None = None) -> dict:
    h = {"Content-Type": "application/json",
         "Accept": "application/json, text/event-stream"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if session:
        h["Mcp-Session-Id"] = session
    return h


async def open_session(client: httpx.AsyncClient, token: str) -> str | None:
    """MCP streamable-HTTP requires an `initialize` handshake first.

    The server replies with an `Mcp-Session-Id` header that every later request
    must carry. Skipping this is why a raw tools/call returns
    "Bad Request: Missing session ID" even with a perfectly valid token --
    an authentication success and a protocol failure look alike from outside.
    """
    r = await client.post(f"{BASE}/mcp", headers=_headers(token), json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "atlas-demo-client", "version": "1.0"}},
    })
    if r.status_code != 200:
        return None
    sid = r.headers.get("mcp-session-id")
    # the spec requires this notification before normal operation
    await client.post(f"{BASE}/mcp", headers=_headers(token, sid),
                      json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return sid


async def call_tool(client: httpx.AsyncClient, name: str, args: dict,
                    token: str | None, session: str | None = None) -> tuple[int, str]:
    """One MCP tools/call over streamable HTTP."""
    r = await client.post(
        f"{BASE}/mcp",
        headers=_headers(token, session),
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": name, "arguments": args}},
    )
    return r.status_code, r.text


def summarise(body: str, limit: int = 300) -> str:
    """Pull the useful payload out of an SSE or JSON-RPC response."""
    for line in body.splitlines():
        if line.startswith("data: "):
            body = line[6:]
            break
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return body[:limit]
    if "error" in obj:
        return f"ERROR {obj['error'].get('code')}: {obj['error'].get('message','')[:limit]}"
    res = obj.get("result", obj)
    if isinstance(res, dict) and res.get("structuredContent"):
        return json.dumps(res["structuredContent"])[:limit]
    if isinstance(res, dict) and res.get("content"):
        return " ".join(c.get("text", "") for c in res["content"])[:limit]
    return json.dumps(res)[:limit]


async def main(scopes: str) -> None:
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
        # ------------------------------------------------------- 1. no token
        h("1. Call a tool with NO token")
        status, body = await call_tool(c, "whoami", {}, None)
        print(f"  HTTP {status}")
        # RFC 9728: the 401 tells the client where to authenticate.
        print(f"  (a 401 here is the server telling us where to get a token)")

        # ------------------------------------------------------ 2. discovery
        h("2. Discover the authorization server")
        meta = None
        for path in ("/.well-known/oauth-authorization-server",
                     "/.well-known/openid-configuration"):
            r = await c.get(BASE + path)
            if r.status_code == 200:
                meta = r.json()
                print(f"  {path} -> 200")
                break
        if meta is None:
            raise SystemExit("no discovery document; is auth_server.py running?")
        for key in ("issuer", "authorization_endpoint", "token_endpoint",
                    "registration_endpoint", "revocation_endpoint"):
            if meta.get(key):
                print(f"    {key:<26} {meta[key]}")
        print(f"    scopes_supported           {meta.get('scopes_supported')}")
        print(f"    code_challenge_methods     {meta.get('code_challenge_methods_supported')}")

        # --------------------------------------------------- 3. registration
        h("3. Register a client dynamically (RFC 7591)")
        redirect_uri = "http://localhost:9999/callback"
        r = await c.post(meta["registration_endpoint"], json={
            "client_name": "atlas-demo-client",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            # OAuth 2.1 mandates PKCE for EVERY client type, not just public
            # ones. We register a confidential client because this SDK requires
            # client authentication on the revocation endpoint (step 10).
            "token_endpoint_auth_method": "client_secret_post",
            "scope": scopes,
        })
        if r.status_code not in (200, 201):
            raise SystemExit(f"registration failed {r.status_code}: {r.text[:400]}")
        client_info = r.json()
        client_id = client_info["client_id"]
        client_secret = client_info.get("client_secret")
        print(f"  client_id      {client_id}")
        print(f"  client_secret  {'issued' if client_secret else 'none (public client)'}")
        print(f"  PKCE is mandatory in OAuth 2.1 regardless of client type")

        # ---------------------------------------------------------- 4. PKCE
        h("4. Generate the PKCE pair")
        verifier, challenge = pkce_pair()
        print(f"  verifier  {verifier[:24]}...   (stays local until step 6)")
        print(f"  challenge {challenge[:24]}...  (SHA256(verifier), sent now)")

        # ----------------------------------------------------- 5. authorize
        h("5. /authorize -> one-time code")
        state = secrets.token_urlsafe(16)
        r = await c.get(meta["authorization_endpoint"], params={
            "response_type": "code", "client_id": client_id,
            "redirect_uri": redirect_uri, "code_challenge": challenge,
            "code_challenge_method": "S256", "state": state, "scope": scopes,
        }, follow_redirects=False)
        loc = r.headers.get("location", "")
        qs = parse_qs(urlparse(loc).query)
        if "code" not in qs:
            raise SystemExit(f"no code returned: HTTP {r.status_code} {loc or r.text[:300]}")
        code = qs["code"][0]
        print(f"  HTTP {r.status_code} -> {urlparse(loc).path}?code={code[:18]}...")
        print(f"  state echoed back correctly: {qs.get('state',[None])[0] == state}")

        # ------------------------------------------------------- 6. exchange
        h("6. Exchange code + verifier -> tokens")
        r = await c.post(meta["token_endpoint"], data={
            "grant_type": "authorization_code", "code": code,
            "redirect_uri": redirect_uri, "client_id": client_id,
            "client_secret": client_secret, "code_verifier": verifier,
        })
        if r.status_code != 200:
            raise SystemExit(f"token exchange failed {r.status_code}: {r.text[:400]}")
        tok = r.json()
        access, refresh = tok["access_token"], tok.get("refresh_token")
        print(f"  access_token  {access[:26]}...  expires_in={tok.get('expires_in')}s")
        print(f"  refresh_token {refresh[:26]}...")
        print(f"  scope         {tok.get('scope')}")

        # --- PKCE negative test: replay the code without the verifier -------
        r2 = await c.post(meta["token_endpoint"], data={
            "grant_type": "authorization_code", "code": code,
            "redirect_uri": redirect_uri, "client_id": client_id,
            "client_secret": client_secret, "code_verifier": "wrong-verifier-entirely",
        })
        print(f"  replaying the same code with a bad verifier -> HTTP {r2.status_code}  "
              f"(must not be 200)")

        # -------------------------------------------------- 7. authorized use
        h("7. Call tools WITH the token")
        session = await open_session(c, access)
        print(f"  MCP session opened: {session}\n")
        for name, args in (("whoami", {}),
                           ("list_atlas_documents", {}),
                           ("search_atlas_docs", {"query": "barcode confidence", "top_k": 1})):
            status, body = await call_tool(c, name, args, access, session)
            print(f"  {name:<22} HTTP {status}  {summarise(body, 200)}")

        # ------------------------------------------------- 8. scope enforced
        h("8. Call a tool OUTSIDE the granted scopes")
        status, body = await call_tool(c, "triage_incident",
                                       {"report": "TLM-330 on atlas-telemetry"}, access, session)
        print(f"  triage_incident        HTTP {status}")
        print(f"  {summarise(body, 320)}")
        print("\n  ^ denied unless atlas:triage was granted. Least privilege at the")
        print("    tool boundary: a docs token cannot occupy the GPU.")

        # ---------------------------------------------------- 9. refresh
        if refresh:
            h("9. Refresh the token (with rotation)")
            r = await c.post(meta["token_endpoint"], data={
                "grant_type": "refresh_token", "refresh_token": refresh,
                "client_id": client_id, "client_secret": client_secret,
            })
            print(f"  HTTP {r.status_code}")
            if r.status_code == 200:
                new = r.json()
                print(f"  new access_token  {new['access_token'][:26]}...")
                print(f"  new refresh_token {new.get('refresh_token','')[:26]}...")
                r2 = await c.post(meta["token_endpoint"], data={
                    "grant_type": "refresh_token", "refresh_token": refresh,
                    "client_id": client_id, "client_secret": client_secret,
                })
                print(f"  reusing the OLD refresh token -> HTTP {r2.status_code}  "
                      f"(must not be 200: rotation makes theft detectable)")
                access = new["access_token"]

        # ---------------------------------------------------- 10. revocation
        if meta.get("revocation_endpoint"):
            h("10. Revoke the access token")
            r = await c.post(meta["revocation_endpoint"],
                             data={"token": access, "client_id": client_id,
                                   "client_secret": client_secret})
            print(f"  revoke -> HTTP {r.status_code}")
            status, body = await call_tool(c, "whoami", {}, access, session)
            print(f"  whoami with the revoked token -> HTTP {status}  (must not be 200)")

        h("done")
        print("  Full OAuth 2.1: dynamic registration, PKCE, code exchange,")
        print("  per-tool scope enforcement, refresh rotation, revocation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", default="atlas:read atlas:ask",
                    help="try 'atlas:read atlas:ask atlas:triage' to pass step 8")
    args = ap.parse_args()
    asyncio.run(main(args.scopes))
