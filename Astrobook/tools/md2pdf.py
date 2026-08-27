"""
Markdown -> styled HTML -> PDF, using headless Edge/Chrome. Stdlib only.

    python tools/md2pdf.py IMPLEMENTATION_PLAN.md
    python tools/md2pdf.py in.md --out out.pdf --keep-html

Supports the subset used in this repo: headings, tables, fenced code, lists,
bold/italic/code spans, links, rules, blockquotes.
"""
import argparse, html, os, re, subprocess, sys, tempfile

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
:root {
  --ink:#1a1a1a; --muted:#5b6570; --rule:#d8dde3; --accent:#7c4a2d;
  --code-bg:#f6f7f9; --th-bg:#f0f2f5;
}
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
  font-size: 10.2pt; line-height: 1.55; color: var(--ink);
  margin: 0 auto; max-width: 190mm; padding: 0 2mm;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 20pt; margin: 0 0 2mm; letter-spacing:-.01em; }
h1 + p { color: var(--muted); margin-top: 0; }
h2 {
  font-size: 13.5pt; margin: 9mm 0 3mm; padding-bottom: 1.6mm;
  border-bottom: 1.5px solid var(--accent); break-after: avoid;
}
h3 { font-size: 11.4pt; margin: 6mm 0 2mm; color:#2c3440; break-after: avoid; }
h4 { font-size: 10.4pt; margin: 4mm 0 1.5mm; color: var(--muted);
     text-transform: uppercase; letter-spacing:.04em; break-after: avoid; }
p, ul, ol { margin: 0 0 3mm; }
li { margin-bottom: 1.2mm; }
ul, ol { padding-left: 6mm; }
strong { font-weight: 640; }
a { color: var(--accent); text-decoration: none; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 7mm 0; }
code {
  font-family: Consolas, "Cascadia Mono", monospace; font-size: .88em;
  background: var(--code-bg); padding: .8mm 1.4mm; border-radius: 2px;
}
pre {
  background: var(--code-bg); border: 1px solid var(--rule);
  border-left: 2.5px solid var(--accent); border-radius: 3px;
  padding: 2.6mm 3.2mm; overflow-x: auto; margin: 0 0 3.5mm;
  break-inside: avoid; font-size: 8.9pt; line-height: 1.42;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
  border-collapse: collapse; width: 100%; margin: 0 0 4mm;
  font-size: 9.1pt; break-inside: avoid;
}
th, td {
  border: 1px solid var(--rule); padding: 1.7mm 2.2mm;
  text-align: left; vertical-align: top;
}
th { background: var(--th-bg); font-weight: 620; }
tr:nth-child(even) td { background: #fbfcfd; }
blockquote {
  margin: 0 0 3mm; padding: 1mm 0 1mm 4mm;
  border-left: 2.5px solid var(--rule); color: var(--muted);
}
"""


def inline(s):
    """Inline spans. Code spans are extracted first so nothing formats inside."""
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: f"<code>{html.escape(spans[int(m.group(1))])}</code>", s)


def render_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]          # cells[1] is the |---| separator
    out = ["<table><thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def md_to_html(md):
    lines = md.split("\n")
    out, i = [], 0
    list_tag = None

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):                                    # fenced code
            close_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        if ln.startswith("|") and i + 1 < len(lines) and re.match(
                r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):          # table
            close_list()
            buf = []
            while i < len(lines) and lines[i].startswith("|"):
                buf.append(lines[i])
                i += 1
            out.append(render_table(buf))
            continue

        if re.match(r"^\s*$", ln):
            close_list()
            i += 1
            continue

        if re.match(r"^(---+|\*\*\*+)\s*$", ln):
            close_list()
            out.append("<hr>")
            i += 1
            continue

        h = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if h:
            close_list()
            lvl = len(h.group(1))
            out.append(f"<h{lvl}>{inline(h.group(2))}</h{lvl}>")
            i += 1
            continue

        if ln.startswith(">"):
            close_list()
            out.append(f"<blockquote>{inline(ln.lstrip('> '))}</blockquote>")
            i += 1
            continue

        ul = re.match(r"^\s*[-*]\s+(.*)$", ln)
        ol = re.match(r"^\s*\d+\.\s+(.*)$", ln)
        if ul or ol:
            want = "ul" if ul else "ol"
            if list_tag != want:
                close_list()
                out.append(f"<{want}>")
                list_tag = want
            item = (ul or ol).group(1)
            i += 1
            while i < len(lines) and re.match(r"^\s{2,}\S", lines[i]) \
                    and not re.match(r"^\s*([-*]|\d+\.)\s", lines[i]):
                item += " " + lines[i].strip()      # continuation line
                i += 1
            out.append(f"<li>{inline(item)}</li>")
            continue

        close_list()
        para = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#{1,6}\s|\||```|>|\s*[-*]\s|\s*\d+\.\s|---+\s*$)", lines[i]):
            para.append(lines[i])
            i += 1
        out.append(f"<p>{inline(' '.join(x.strip() for x in para))}</p>")

    close_list()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out")
    ap.add_argument("--keep-html", action="store_true")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out or os.path.splitext(src)[0] + ".pdf")
    md = open(src, encoding="utf-8").read()
    title = next((l[2:].strip() for l in md.split("\n") if l.startswith("# ")),
                 os.path.basename(src))

    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body>{md_to_html(md)}</body></html>")

    html_path = (os.path.splitext(out)[0] + ".html" if a.keep_html
                 else os.path.join(tempfile.gettempdir(), "_md2pdf.html"))
    open(html_path, "w", encoding="utf-8").write(doc)

    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        sys.exit("No Edge/Chrome found; HTML written to " + html_path)

    # A dedicated --user-data-dir is REQUIRED. Without it, an already-running
    # Edge/Chrome swallows the invocation ("Opening in existing browser
    # session."), exits 0, and silently produces no PDF.
    profile = os.path.join(tempfile.gettempdir(), "md2pdf_profile")
    os.makedirs(profile, exist_ok=True)

    url = "file:///" + html_path.replace("\\", "/")
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run",
           "--no-default-browser-check", f"--user-data-dir={profile}",
           "--virtual-time-budget=20000", "--no-pdf-header-footer",
           f"--print-to-pdf={out}", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not os.path.exists(out):
        sys.exit(f"PDF not produced.\n{r.stdout}\n{r.stderr}")

    print(f"{os.path.basename(src)} -> {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    if a.keep_html:
        print(f"html kept: {html_path}")


if __name__ == "__main__":
    main()
