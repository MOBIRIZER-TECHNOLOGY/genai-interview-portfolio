# 🎤 Interview notes — MCP and tool-using agents

---

## The 60-second project pitch

> "An MCP server that exposes my RAG pipeline and my fine-tuned triage model as
> tools any AI client can call. Four tools, two resources, a prompt template,
> over stdio JSON-RPC. The part I'd point at is the integration: `triage_incident`
> runs the LoRA adapter I trained in the other project on the local GPU, and
> `ask_atlas` correctly abstains when the docs don't cover the question — the
> grounding discipline survives the trip through the protocol, which is the
> thing that usually breaks when you wrap a RAG system in a tool call."

---

## Core questions

### "What is MCP and what problem does it solve?"

An open JSON-RPC protocol that standardises how AI applications connect to
external tools and data.

The problem is combinatorial. Before it, N AI clients × M integrations meant N×M
bespoke adapters, each in a different plugin format. MCP makes it N+M: write a
server once, every MCP client can use it.

The useful analogy is the language server protocol. Before LSP, every editor
needed a plugin per language. After, one language server serves every editor.
MCP is that idea for AI clients and tools.

### "Tools vs resources vs prompts — when do you use each?"

- **Tool** — the *model* decides to invoke it. Actions and lookups. `search_docs`,
  `create_ticket`, `run_query`.
- **Resource** — the *application or user* attaches it. Passive context.
  A file, a DB record, a schema. Addressed by URI, and can be templated
  (`atlas://docs/{name}`).
- **Prompt** — the *user* invokes it. A reusable template, typically surfaced as
  a slash command.

The clean test: **who decides this enters the context?** Model → tool. Human →
resource. And a pragmatic note: tool support is universal across clients today
while resource and prompt support varies, so when genuinely torn, ship a tool.

### "How do you design a good tool?"

**The description is the API.** The model chooses tools by reading descriptions,
so that text does more work than the code. Mine say what the tool returns *and*
when to prefer it over its neighbour — `ask_atlas` explicitly says "use
`search_atlas_docs` when you want raw passages instead". Without that, a model
picks near-randomly between two plausible tools.

Then:
- **Return structured JSON**, not prose. The model reasons over the result.
- **Few tools, sharp boundaries.** Thirty overlapping tools behave worse than six
  clear ones — every extra tool is another chance to pick wrong, and they all
  compete for context.
- **Errors should be informative.** `triage_incident` returns
  `{"error": ..., "raw": ...}` when the model output doesn't parse, because
  "here's what it actually said" is actionable and "tool failed" isn't.
- **Idempotent and safe by default.** A tool the model may call speculatively
  should not have side effects you can't undo.

### "What's the most common way to break an MCP stdio server?"

Printing to stdout. On stdio transport **stdout is the JSON-RPC stream** — a
stray `print()` or a library banner injects garbage into the protocol and the
client disconnects with an opaque parse error. Log to stderr.

Second most common: doing heavy work at import. My RAG pipeline takes ~5 s to
load; if that ran at import, every spawn would look hung. Lazy-load on first
call and cache.

Third: assuming the process is long-lived. stdio servers are spawned per client
session and can be restarted at any time. Don't hold unsaved state in memory.

### "You implemented OAuth 2.1. Walk me through it."

Full authorization server: dynamic client registration (RFC 7591),
authorization-code flow with **mandatory PKCE**, token exchange, refresh with
rotation, revocation (RFC 7009), and per-tool scope enforcement.

**What 2.1 changes from 2.0, and why each matters:**
- **PKCE is mandatory for every client type**, not just public ones. The client
  commits to `code_challenge = SHA256(verifier)` up front and must present the
  raw verifier at exchange. An attacker who intercepts the redirect gets a code
  that is useless without the verifier they never saw.
- **Implicit flow removed** — no tokens in URL fragments, where they land in
  browser history and referrer headers.
- **Exact redirect-URI matching** — prefix/wildcard matching was a reliable
  source of open redirects.
- **Refresh rotation** — each refresh invalidates the old token, so if both an
  attacker and the real client use it, the second use fails and you have a
  detectable compromise signal.

**The part I'd emphasise: I demonstrated the negatives.** Anyone can turn auth
on. My client demo proves PKCE rejects a replayed code with a wrong verifier
(400), that reusing a rotated refresh token fails (400), and that a revoked token
stops working (401). Positive-path-only auth demos prove almost nothing.

**Scopes per tool, not per server.** `atlas:read` for docs, `atlas:ask` for RAG,
`atlas:triage` for the GPU model. An MCP server bundles capabilities with very
different blast radii — a docs-browsing integration should not be able to occupy
your GPU. That is least privilege at the tool boundary, and it is what makes
"we exposed internal systems over MCP" defensible.

**What I left non-production, and I'd say so unprompted:** in-memory token
storage (needs Redis/Postgres to survive restarts and work across replicas),
auto-approval standing in for real user authentication and a consent screen, and
opaque tokens rather than signed JWTs. JWTs let the resource server validate
locally with no round-trip, at the cost of instant revocation — that trade is a
real design decision, not an oversight.

### "When does an MCP server need auth at all?"

stdio doesn't. The client spawns it as a subprocess; the OS already decided you
may run that binary, and there is no network surface. Adding OAuth to a stdio
server is pure ceremony.

HTTP does, always, because it is reachable. That's the actual dividing line, and
it's why this project ships both: `server.py` over stdio unauthenticated, and
`auth_server.py` over HTTP with the full flow.

### "How would you secure an MCP server?"

Depends on where the trust boundary is.

**Local stdio server** (this one): the server runs with the user's full
permissions. Anything it can do, a prompt-injected model can do. So: no tool
that shells out or writes arbitrary paths unless that's the actual point;
validate every path from user input (my `read_doc` checks the resolved path is
inside the corpus, *in addition to* the SDK's traversal guard); treat retrieved
document content as untrusted — if an attacker can put text in your corpus, they
can attempt to instruct the model through your tool's return value.

**Remote HTTP server**: now you need real authentication (the SDK supports OAuth
via `auth_server_provider` / `token_verifier`), authorisation per tool, rate
limiting, and audit logging of every call.

The thing worth saying unprompted: **prompt injection through tool results is a
real attack surface** and most MCP deployments haven't thought about it. Content
that comes back from a search tool is data, but the model reads it as text in its
context.

### "How is this different from OpenAI function calling?"

Different layers, and they compose.

Function calling is a **model capability** — the model emits a structured call
instead of prose. MCP is a **transport and discovery protocol** — how a client
finds out what tools exist and invokes them across a process boundary.

In practice: an MCP client calls `list_tools()`, hands those schemas to the model
as function definitions, the model picks one, the client executes it via MCP and
feeds the result back. MCP doesn't replace function calling; it standardises
where the functions come from. `client_demo.py` is exactly the first half of that
loop with the model taken out.

### "Walk me through what happens when Claude Code calls one of your tools."

1. Claude Code reads `.mcp.json`, spawns `python server.py` as a subprocess.
2. **Handshake:** `initialize` request over stdin, server replies with protocol
   version, capabilities, and its `instructions`.
3. **Discovery:** `tools/list`, `resources/list`, `prompts/list`. The tool schemas
   go into the model's context as available functions.
4. The user asks something. The model decides `search_atlas_docs` fits and emits
   a call with arguments matching my JSON Schema.
5. **Invocation:** `tools/call` over stdin. My function runs — embeds the query,
   hits FAISS and BM25, reranks on GPU — and returns a dict.
6. The SDK serialises it into content blocks (plus `structured_content`) and
   writes the JSON-RPC response to stdout.
7. The result lands back in the model's context, and it answers.

The whole thing is line-delimited JSON over two pipes. Being able to say that
plainly is worth more than knowing the SDK's decorator names.

---

## Questions to ask *them*

- "Do you build MCP servers internally, and how do you handle auth for remote ones?"
- "How do you decide what becomes a tool versus what goes in the system prompt?"
- "Have you hit prompt injection through tool results, and how did you handle it?"
- "How do you evaluate whether an agent is *choosing* tools correctly, separately
  from whether the tools work?"

That last one is the senior question. Tool correctness and tool *selection* are
different failure modes with different fixes — code for one, descriptions and
evals for the other.

---

## Related projects

- **[01_rag_local](../01_rag_local/)** — what `ask_atlas` and `search_atlas_docs` wrap
- **[02_lora_text](../02_lora_text/)** — what `triage_incident` wraps
