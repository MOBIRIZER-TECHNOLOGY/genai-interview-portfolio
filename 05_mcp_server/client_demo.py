"""
Drive the MCP server from Python, without an AI client in the loop.

    python client_demo.py                 # exercise everything
    python client_demo.py --skip-triage   # skip the GPU tool

This exists for two reasons:

1. **You can debug the server.** When a tool misbehaves inside Claude Code you
   are looking at it through two layers of model behaviour. Here you call it
   directly and see the raw JSON-RPC result.
2. **It's how you'd write an agent.** An MCP client is what any agent framework
   is doing under the hood: discover tools, hand their schemas to a model, let
   the model choose, execute, feed results back.

The client spawns `server.py` as a subprocess and speaks JSON-RPC over its
stdin/stdout. That's the whole stdio transport.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

HERE = Path(__file__).parent


def show(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def unwrap(result) -> str:
    """Tool results are a list of content blocks; pull out the text."""
    if getattr(result, "structured_content", None):
        return json.dumps(result.structured_content, indent=2)
    parts = []
    for block in result.content:
        parts.append(getattr(block, "text", str(block)))
    return "\n".join(parts)


async def main(skip_triage: bool) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(HERE / "server.py")],
    )

    # stdio_client(params) spawns the subprocess and yields the read/write
    # streams; Client wraps those in a JSON-RPC session and does the handshake.
    async with Client(stdio_client(params)) as client:
        info = client.server_info
        print(f"connected to: {info.name} v{info.version}")
        if client.instructions:
            print(f"\ninstructions:\n{client.instructions}")

        # ------------------------------------------------------------ tools
        show("TOOLS the server advertises")
        tools = await client.list_tools()
        for t in tools.tools:
            required = t.input_schema.get("required", [])
            props = ", ".join(
                f"{k}{'*' if k in required else ''}: {v.get('type', '?')}"
                for k, v in t.input_schema.get("properties", {}).items()
            )
            print(f"\n  {t.name}({props})")
            print(f"    {(t.description or '').strip()[:180]}")

        # -------------------------------------------------------- resources
        show("RESOURCES")
        res = await client.list_resources()
        for r in res.resources:
            print(f"  {r.uri}   [{r.mime_type}]  {r.name}")
        templates = await client.list_resource_templates()
        for t in templates.resource_templates:
            print(f"  {t.uri_template}   (template)  {t.name}")

        show("PROMPTS")
        prompts = await client.list_prompts()
        for p in prompts.prompts:
            argnames = ", ".join(a.name for a in (p.arguments or []))
            print(f"  /{p.name}({argnames})  -  {p.description}")

        # ----------------------------------------------------- call: list
        show("CALL  list_atlas_documents()")
        out = json.loads(unwrap(await client.call_tool("list_atlas_documents", {})))
        print(f"  {out['count']} documents:")
        for d in out["documents"]:
            print(f"    {d['name']:<32} {len(d['sections'])} sections  {d['bytes']:>5} B")

        # --------------------------------------------------- call: search
        show("CALL  search_atlas_docs(query='barcode confidence threshold', top_k=2)")
        out = json.loads(
            unwrap(await client.call_tool(
                "search_atlas_docs", {"query": "barcode confidence threshold", "top_k": 2}
            ))
        )
        print(f"  {out['latency_ms']} ms, mode={out['mode']}")
        for r in out["results"]:
            print(f"\n  -- {r['section']}  (rerank {r['rerank_score']})")
            print("     " + r["text"][:260].replace("\n", "\n     "))

        # ------------------------------------------------------ call: ask
        show("CALL  ask_atlas(question='What is the Rotterdam rule?')")
        out = json.loads(unwrap(await client.call_tool("ask_atlas", {"question": "What is the Rotterdam rule?"})))
        print(f"  grounded={out['grounded']}  abstained={out['abstained']}  {out['latency_ms']} ms")
        print(f"\n  {out['answer']}\n")
        for s in out["sources"]:
            print(f"    source: {s['section']}")

        show("CALL  ask_atlas  on something NOT in the corpus")
        out = json.loads(unwrap(await client.call_tool(
            "ask_atlas", {"question": "What is the Atlas engineering salary band?"}
        )))
        print(f"  abstained={out['abstained']}\n  {out['answer']}")

        # -------------------------------------------------- read: resource
        show("READ RESOURCE  atlas://index/stats")
        r = await client.read_resource("atlas://index/stats")
        print("  " + r.contents[0].text.replace("\n", "\n  "))

        show("READ RESOURCE  atlas://docs/05-oncall-runbook.md  (first 400 chars)")
        r = await client.read_resource("atlas://docs/05-oncall-runbook.md")
        print("  " + r.contents[0].text[:400].replace("\n", "\n  "))

        # ---------------------------------------------------- get: prompt
        show("GET PROMPT  incident_triage")
        p = await client.get_prompt(
            "incident_triage",
            {"report": "Rotterdam here, TLM-330 on atlas-telemetry, robots are halted."},
        )
        for m in p.messages:
            print(f"  [{m.role}] {getattr(m.content, 'text', m.content)}")

        # --------------------------------------------------- call: triage
        if not skip_triage:
            show("CALL  triage_incident(...)   [runs the project-02 LoRA on GPU]")
            report = ("Memphis cell. We're seeing VIS-207 on atlas-vision -- gantry 4 "
                      "GPU has fallen off the bus. Throughput is down 42%.")
            print(f"  report: {report}")
            out = json.loads(unwrap(await client.call_tool("triage_incident", {"report": report})))
            print("\n  " + json.dumps(out.get("triage", out), indent=2).replace("\n", "\n  "))

        show("done")
        print("  Every call above went over JSON-RPC to a subprocess. Same protocol")
        print("  Claude Code uses when it loads this server from .mcp.json.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-triage", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.skip_triage))
