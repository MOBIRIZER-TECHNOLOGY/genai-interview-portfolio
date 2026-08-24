# 🔌 Project 05 — MCP Server (tools, resources and prompts for an AI agent)

An **MCP (Model Context Protocol)** server that gives any AI client access to the
Atlas knowledge base and the fine-tuned triage model built in projects 01 and 02.

> **In one sentence:** one Python file turns the RAG pipeline and the LoRA
> adapter into tools that Claude Code — or any MCP client — can call, over the
> same JSON-RPC protocol every MCP integration speaks.

---

## 🧠 The idea (for non-experts)

A language model on its own is a text function: text in, text out. To be useful
it has to *do* things — search your docs, query your database, call your API.

Before MCP, every AI product invented its own plugin format. If you had a useful
internal service, you wrote the integration once for each client. **MCP is a
standard connector**: write the server once, and every MCP-speaking client
(Claude Code, Claude Desktop, agent SDKs, IDEs) can use it.

Think USB-C rather than a drawer of proprietary chargers.

### The three primitives — and how to choose

| Primitive | Who triggers it | Use it for |
|---|---|---|
| **Tool** | the **model** decides | actions and lookups: search, query, write, compute |
| **Resource** | the **app/user** attaches | passive context to read: files, records, schemas |
| **Prompt** | the **user** invokes | reusable templates, surfaced as slash commands |

The distinction people get wrong: a **Resource is passive context**, a **Tool is
an action the model chooses to take**. If the model has to decide whether to
fetch it, it's a tool. If a human is attaching it, it's a resource.

---

## ✅ Proof it works

`python client_demo.py` connects over stdio and exercises everything. Real output:

```
connected to: atlas-knowledge v1.0.0

TOOLS
  ask_atlas(question*: string, top_k: integer)
  search_atlas_docs(query*: string, top_k: integer, mode: string)
  list_atlas_documents()
  triage_incident(report*: string)

RESOURCES
  atlas://index/stats   [application/json]
  atlas://docs/{name}   (template)

PROMPTS
  /incident_triage(report)

CALL  search_atlas_docs(query='barcode confidence threshold', top_k=2)
  293 ms, mode=hybrid
  -- 04-vision-service.md > ... > Confidence policy   (rerank 0.6376)
     - Barcode reads below **0.92** confidence are re-attempted up to 3 times...

CALL  ask_atlas(question='What is the Rotterdam rule?')
  grounded=True  abstained=False  652 ms

CALL  ask_atlas('What is the Atlas engineering salary band?')
  abstained=True
  NOT_FOUND: The context does not contain information about salary bands.

CALL  triage_incident('Memphis cell. We're seeing VIS-207 on atlas-vision --
                       gantry 4 GPU has fallen off the bus. Throughput down 42%.')
  {
    "component": "atlas-vision",
    "severity": "SEV2",
    "error_code": "VIS-207",
    "page_oncall": true,
    "action": "power cycle the affected gantry"
  }
```

That last call is the whole portfolio in one line: an MCP client invoked a tool
that ran the **project-02 LoRA adapter** on the local GPU. And `ask_atlas`
correctly **abstained** on the salary question rather than inventing a number —
the grounding discipline from project 01 survives the trip through the protocol.

---

## 📁 What's in this project

```
05_mcp_server/
├── server.py         4 tools, 2 resources, 1 prompt
├── client_demo.py    a Python MCP client that drives all of it
├── .mcp.json         config for Claude Code
└── requirements.txt
```

```
                  ┌──────────────────────────┐
  Claude Code ────┤                          ├──▶ 01_rag_local  (retrieval + RAG)
  or client_demo  │   server.py  (stdio      │
  or any MCP      │     JSON-RPC)            ├──▶ 02_lora_text  (triage adapter)
  client     ─────┤                          │
                  └──────────────────────────┘──▶ corpus/*.md   (as resources)
```

---

## 🚀 How to run it

### 0. Prerequisites

Projects 01 and 02 must have been built — the server wraps them:

```powershell
cd ..\01_rag_local ; python ingest.py
cd ..\02_lora_text ; python make_dataset.py ; python train_lora.py --epochs 3
cd ..\05_mcp_server
```

### 1. Drive it from Python (no AI client needed)

```powershell
python client_demo.py
python client_demo.py --skip-triage      # skip the GPU tool
```

### 2. Wire it into Claude Code

Copy `.mcp.json` to the root of any project (edit the two paths first), then
start Claude Code there. Check it loaded with `/mcp`. Then just ask:

> "What's the Atlas freeze window?"
> "Triage this: Rotterdam, TLM-330 on atlas-telemetry, robots are halted."

### 3. Run it over HTTP instead of stdio

```powershell
python server.py --http --port 8765
```

---

## 🔧 Adding your own tool

```python
@server.tool(
    description=(
        "One clear sentence on WHAT it does, then WHEN to use it versus the "
        "other tools. This text is the model's only guide to choosing."
    )
)
def my_tool(query: str, limit: int = 5) -> dict:
    return {"results": [...]}
```

Type hints become the JSON Schema automatically. Three rules that decide whether
a model uses your tool well:

1. **The description is the API.** The model picks tools by reading descriptions.
   "Search documents" is useless when three tools search something. Say what it
   returns and when to prefer it over the alternatives.
2. **Return structured data, not prose.** The model is going to reason over the
   result. Give it JSON with named fields, not a formatted paragraph.
3. **Keep the tool count small and the boundaries sharp.** Thirty overlapping
   tools produce worse behaviour than six clear ones.

---

## ⚙️ Design decisions in this server

| Decision | Why |
|---|---|
| **Lazy model loading** | The RAG pipeline takes ~5 s to load. Doing that at import makes a stdio server look hung on every spawn. Heavy objects are built on first call and cached in `_STATE`. |
| **`log_level="WARNING"` + muted dependency loggers** | On stdio, **stdout is the protocol stream**. Anything printed there corrupts JSON-RPC. Even stderr matters: 200 lines of HuggingFace download logs makes a working server look broken. |
| **Path validation in `read_doc`** | The SDK's `ResourceSecurity` already rejects traversal, but a resource that reads files from a user-supplied name should never rely solely on the framework. |
| **`triage_incident` returns the raw text on parse failure** | Surfacing "the model returned this and it wasn't JSON" is far more useful to an agent than a generic error. |
| **`search_atlas_docs` exposes `mode`** | Lets a caller (or you, debugging) compare hybrid/dense/bm25 through the protocol. |

---

## 🖥️ Tech stack

- **`mcp` 2.0** — `MCPServer` with `@server.tool` / `@server.resource` / `@server.prompt`
- **Transports:** stdio (default) and streamable HTTP
- **Backends:** project 01's RAG pipeline, project 02's LoRA adapter, both local
- **Validated on:** Python 3.12, Windows 11, Claude Code

---

## ❓ FAQ

**Tool or resource — how do I actually decide?**
Ask "who decides this gets fetched?" If the model must decide based on the
conversation, it's a tool. If the user or app attaches it up front, it's a
resource. When genuinely torn, ship a tool — models handle tools better today,
and resource support varies between clients.

**Why is nothing printed to stdout?**
On stdio transport, stdout **is** the JSON-RPC channel. A stray `print()`
injects garbage into the protocol stream and the client disconnects with a
confusing parse error. Log to stderr, always. This is the single most common way
to break an MCP server.

**stdio or HTTP?**
stdio for a local server the client spawns as a subprocess — simplest, no ports,
no auth, process lifetime is managed for you. HTTP when the server is remote,
shared between clients, or needs to outlive any single client.

**How do I debug a tool the model keeps misusing?**
Call it from `client_demo.py` first to confirm the tool itself is correct. If it
is, the problem is almost always the **description** — the model is choosing
wrong because you told it the wrong thing. Rewrite the description before you
touch the code.

**Is this secure?**
For a local server reading local docs, the threat model is small. Things that
would matter before exposing it: the server runs with your full user
permissions, so any tool that writes files or shells out is a real capability;
tool descriptions and returned content are attacker-controllable if your corpus
is (prompt injection); and over HTTP you need auth, which the SDK supports via
`auth_server_provider` / `token_verifier`.

---

## Related projects

- **[01_rag_local](../01_rag_local/)** — the retrieval engine behind `ask_atlas`
- **[02_lora_text](../02_lora_text/)** — the adapter behind `triage_incident`
