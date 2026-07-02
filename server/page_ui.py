"""
Dossier-styled HTML views for the JSON-bearing endpoints.

Separate from server/landing_ui.py (which owns /) so the shell can stay lean
and be reused across sub-pages. API consumers continue to hit the raw JSON
routes (/metadata, /tasks, /themes/coverage, etc.) — these /ui/* routes are
browser-facing accessors that render the same data inside the SOC-Dossier
aesthetic.
"""

from __future__ import annotations

import html
import json as _json
from collections.abc import Iterable
from typing import Any

from fastapi.responses import HTMLResponse

# ─── shared shell ────────────────────────────────────────────────────────────
_STYLE = r"""
:root{
  --paper:#f1ebdd;--paper-2:#f7f1e2;--paper-edge:#e7dfc9;
  --ink:#161a22;--ink-soft:#2b2f38;--muted:#7a7264;
  --hairline:#c9bfaa;--hairline-soft:#ddd3bc;
  --stamp:#b8321e;--stamp-soft:rgba(184,50,30,.08);
  --cobalt:#2a4980;--cobalt-soft:rgba(42,73,128,.08);
  --phosphor:#8affc3;--phosphor-dim:#4f8a6d;
  --term-bg:#0e130f;--term-ink:#e6e0cd;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--paper);color:var(--ink)}
html{-webkit-font-smoothing:antialiased;font-feature-settings:"ss01","cv11","liga"}
body{font-family:"Fraunces",Georgia,serif;font-variation-settings:"opsz" 14,"SOFT" 40,"wght" 380;font-size:16px;line-height:1.55;letter-spacing:.004em;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;pointer-events:none;background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' seed='5'/><feColorMatrix values='0 0 0 0 0.08  0 0 0 0 0.06  0 0 0 0 0.04  0 0 0 0.09 0'/></filter><rect width='200' height='200' filter='url(%23n)'/></svg>");opacity:.5;mix-blend-mode:multiply;z-index:1}
body::after{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(ellipse at 50% 30%, transparent 45%, rgba(20,16,8,.12) 100%);z-index:2}

.advisory{position:relative;z-index:10;border-bottom:1px solid var(--hairline);background:var(--paper-2);font-family:"JetBrains Mono",ui-monospace,monospace;font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--ink-soft);padding:7px 22px;display:flex;justify-content:space-between;align-items:center}
.advisory a{color:var(--ink-soft);text-decoration:none;display:inline-flex;align-items:center;gap:8px;transition:color .2s}
.advisory a:hover{color:var(--stamp)}
.advisory .tick{display:inline-block;width:6px;height:6px;background:var(--stamp);border-radius:50%;margin-right:10px;animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.35;transform:scale(.9)}50%{opacity:1;transform:scale(1.1)}}

.page{position:relative;z-index:3;max-width:1240px;margin:0 auto;padding:28px 38px 80px}

header.dossier{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:18px 0 14px;border-bottom:1px solid var(--ink)}
.serial{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--ink-soft);display:flex;gap:22px;flex-wrap:wrap}
.serial b{color:var(--ink);font-weight:600}
.emblem{width:44px;height:44px;border:1.5px solid var(--ink);border-radius:50%;display:grid;place-items:center;position:relative;background:var(--paper-2)}
.emblem::after{content:"";position:absolute;inset:4px;border:.5px solid var(--ink);border-radius:50%}
.emblem svg{width:22px;height:22px}

.crumbs{padding:18px 0 6px;font-family:"JetBrains Mono",monospace;font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);display:flex;gap:12px;align-items:center}
.crumbs a{color:var(--muted);text-decoration:none;border-bottom:.5px solid transparent;transition:all .2s}
.crumbs a:hover{color:var(--stamp);border-color:var(--stamp)}
.crumbs .sep{opacity:.4}

.masthead{padding:18px 0 18px;display:grid;grid-template-columns:1fr auto;align-items:end;border-bottom:.5px solid var(--hairline);gap:30px}
.masthead h1{font-family:"Instrument Serif",serif;font-weight:400;font-style:italic;font-size:clamp(48px,7vw,96px);line-height:.92;letter-spacing:-.018em;color:var(--ink)}
.masthead h1 em{font-style:normal;color:var(--stamp);font-family:"Instrument Serif",serif}
.masthead-meta{font-family:"JetBrains Mono",monospace;font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);text-align:right;padding-bottom:10px}
.masthead-meta b{color:var(--ink);font-weight:600;display:block;margin-bottom:4px}
.masthead-meta a{color:var(--cobalt);text-decoration:none;border-bottom:.5px solid var(--cobalt-soft)}
.masthead-meta a:hover{color:var(--stamp);border-color:var(--stamp)}

.lede{font-family:"Instrument Serif",serif;font-size:clamp(22px,2.4vw,30px);line-height:1.22;letter-spacing:-.003em;color:var(--ink);max-width:720px;padding:30px 0 28px;border-bottom:1px solid var(--ink)}
.lede em{font-style:italic;color:var(--stamp)}

.content{padding-top:42px}

.eyebrow{font-family:"JetBrains Mono",monospace;font-size:10.5px;letter-spacing:.28em;text-transform:uppercase;color:var(--stamp);display:flex;align-items:center;gap:12px;margin-bottom:14px}
.eyebrow::before{content:"";width:28px;height:1px;background:var(--stamp)}

h2{font-family:"Instrument Serif",serif;font-size:clamp(30px,3.4vw,46px);font-weight:400;line-height:1;letter-spacing:-.012em;margin-bottom:18px}
h2 em{font-style:italic;color:var(--stamp)}

.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1.5px solid var(--ink)}
.card{padding:24px 26px;border-bottom:.5px solid var(--hairline);border-right:.5px solid var(--hairline);position:relative}
.card:nth-child(even){border-right:none}
.card .n{position:absolute;top:18px;right:22px;font-family:"Instrument Serif",serif;font-style:italic;font-size:42px;color:var(--stamp);opacity:.18;line-height:1}
.card .k{font-family:"JetBrains Mono",monospace;font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.card .v{font-family:"Instrument Serif",serif;font-size:28px;color:var(--ink);line-height:1.1;margin-bottom:10px}
.card .v em{font-style:italic;color:var(--stamp)}
.card .d{font-family:"Fraunces",serif;font-size:14px;line-height:1.55;color:var(--ink-soft);max-width:48ch}
.card code{font-family:"JetBrains Mono",monospace;font-size:.88em;background:var(--paper-2);padding:1px 6px;border:.5px solid var(--hairline);color:var(--ink)}

.task-index{display:grid;grid-template-columns:64px 1fr 120px 100px 90px;font-family:"Fraunces",serif;font-size:15px;margin-top:18px;border-top:1.5px solid var(--ink)}
.task-index .row{display:contents}
.task-index .row>*{padding:16px;border-bottom:.5px solid var(--hairline-soft);display:flex;align-items:center;transition:background .2s}
.task-index .row.head>*{border-bottom:1.5px solid var(--ink);font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);padding:0 16px 10px}
.task-index .row:not(.head):hover>*{background:var(--stamp-soft)}
.task-index .code{font-family:"JetBrains Mono",monospace;font-size:12px;letter-spacing:.1em;color:var(--muted)}
.task-index .name{font-family:"Instrument Serif",serif;font-size:20px;line-height:1.1;color:var(--ink)}
.task-index .name em{font-style:italic;color:var(--stamp)}
.task-index .diff{font-family:"JetBrains Mono",monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase}
.task-index .diff.e{color:#2c7a58}.task-index .diff.m{color:#b87a1d}.task-index .diff.h{color:#c35a3a}.task-index .diff.x{color:var(--stamp);font-weight:600}.task-index .diff.a{color:var(--cobalt)}
.task-index .steps,.task-index .mode{font-family:"JetBrains Mono",monospace;font-size:11.5px;color:var(--ink-soft);letter-spacing:.04em}
.task-index .mode{letter-spacing:.2em;text-transform:uppercase;font-size:9.5px;color:var(--ink-soft)}

pre.raw{font-family:"JetBrains Mono",monospace;font-size:12.5px;line-height:1.62;background:var(--term-bg);color:var(--term-ink);padding:22px 26px;margin:28px 0 0;border:1px solid var(--ink);overflow:auto;max-height:640px;position:relative;box-shadow:8px 8px 0 -1px var(--stamp-soft)}
pre.raw::before{content:"// RAW ·  application/json";position:absolute;top:-11px;left:14px;background:var(--paper);color:var(--muted);font-size:9.5px;letter-spacing:.24em;padding:2px 10px;border:.5px solid var(--hairline);text-transform:uppercase}
pre.raw .k{color:#ffd59e}pre.raw .s{color:#a6f4c5}pre.raw .n{color:#f4a6a6}pre.raw .b{color:#9ec7ff}pre.raw .p{color:#7a7264}

.kv{display:grid;grid-template-columns:220px 1fr;gap:0;border-top:1.5px solid var(--ink);margin-top:18px}
.kv>div{padding:14px 0;border-bottom:.5px solid var(--hairline);font-family:"JetBrains Mono",monospace;font-size:12px;color:var(--ink-soft)}
.kv>.key{letter-spacing:.14em;text-transform:uppercase;font-size:10.5px;color:var(--muted);padding-right:20px}
.kv>.val{font-family:"Fraunces",serif;font-size:15px;color:var(--ink);line-height:1.55}
.kv>.val code{font-family:"JetBrains Mono",monospace;font-size:12.5px;background:var(--paper-2);padding:1px 6px;border:.5px solid var(--hairline)}
.kv>.val a{color:var(--cobalt);text-decoration:none;border-bottom:.5px solid var(--cobalt-soft)}
.kv>.val a:hover{color:var(--stamp);border-color:var(--stamp)}
.tags{display:flex;gap:6px;flex-wrap:wrap}
.tag{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--stamp);border:.5px solid var(--stamp);padding:3px 8px;background:var(--stamp-soft)}

.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
.chip{font-family:"JetBrains Mono",monospace;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink);border:.5px solid var(--ink);padding:4px 10px;background:var(--paper-2)}
.chip.green{border-color:#2c7a58;color:#2c7a58}
.chip.red{border-color:var(--stamp);color:var(--stamp)}
.chip.cobalt{border-color:var(--cobalt);color:var(--cobalt)}

.footer{margin-top:60px;padding:28px 0 10px;border-top:2px solid var(--ink);display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
.footer a{font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:.28em;text-transform:uppercase;color:var(--muted);text-decoration:none;border-bottom:.5px solid transparent;padding-bottom:2px;transition:all .2s}
.footer a:hover{color:var(--stamp);border-color:var(--stamp)}

.reveal{opacity:0;transform:translateY(12px);animation:rise .8s cubic-bezier(.2,.7,.2,1) forwards}
@keyframes rise{to{opacity:1;transform:translateY(0)}}
.d1{animation-delay:.05s}.d2{animation-delay:.18s}.d3{animation-delay:.32s}.d4{animation-delay:.48s}.d5{animation-delay:.62s}

::selection{background:var(--stamp);color:var(--paper)}

@media (max-width:900px){
  .page{padding:24px 20px 60px}
  .grid-2{grid-template-columns:1fr}
  .card{border-right:none!important}
  .task-index{grid-template-columns:52px 1fr 90px 60px;font-size:14px}
  .task-index .steps{display:none}
  .kv{grid-template-columns:1fr}
  .kv>.key{border:none;padding:12px 0 0}
  .kv>.val{padding-top:4px}
}
"""


def _shell(title: str, eyebrow: str, file_no: str, lede: str, inner: str) -> str:
    """The full HTML shell — one consistent frame for every sub-page."""
    fonts_link = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1'
        '&family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..900,0..100;1,9..144,300..900,0..100'
        '&family=JetBrains+Mono:ital,wght@0,300..800;1,300..800&display=swap" rel="stylesheet">'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)} — SOC·Triage·Gym / File {file_no}</title>
{fonts_link}
<style>{_STYLE}</style>
</head>
<body>

<div class="advisory">
  <span><i class="tick"></i>OPS/LIVE · FILE {html.escape(file_no)}</span>
  <span>
    <a href="/">← Dossier</a>  &nbsp;·&nbsp;
    <a href="/docs">API Spec</a>  &nbsp;·&nbsp;
    <a href="https://github.com/ROHITCRAFTSYT/-Metas-OpenEnv-2" target="_blank" rel="noopener">GitHub ↗</a>
  </span>
</div>

<div class="page">

<header class="dossier reveal d1">
  <div class="serial"><span>FILE NO. <b>{html.escape(file_no)}</b></span></div>
  <div class="emblem"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M12 2 L3 6 V12 C3 17 7 21 12 22 C17 21 21 17 21 12 V6 Z"/><path d="M9 12 L11.5 14.5 L15.5 10" stroke-width="1.4"/></svg></div>
  <div class="serial" style="justify-content:flex-end"><span>CLS. <b>OPEN / JUDGE</b></span></div>
</header>

<nav class="crumbs reveal d1">
  <a href="/">SOC·Triage·Gym</a>
  <span class="sep">/</span>
  <span>{html.escape(eyebrow)}</span>
</nav>

<div class="masthead reveal d2">
  <h1>{title}</h1>
  <div class="masthead-meta">
    <b>Accessor View</b>
    <a href="#raw">↓ Raw JSON below</a>
  </div>
</div>

<p class="lede reveal d3">{lede}</p>

<div class="content reveal d4">
{inner}
</div>

<div class="footer reveal d5">
  <a href="/">← Return to Dossier</a>
  <a href="/docs">OpenAPI · /docs</a>
  <a href="/themes/coverage">Theme Manifest</a>
  <a href="/tasks">Tasks JSON</a>
  <a href="https://github.com/ROHITCRAFTSYT/-Metas-OpenEnv-2" target="_blank" rel="noopener">Source ↗</a>
</div>

</div>
</body>
</html>"""


# ─── JSON pretty-printer with token colorization ─────────────────────────────
def _format_json(data: Any) -> str:
    """Render JSON with simple span-based syntax highlight."""
    text = _json.dumps(data, indent=2, ensure_ascii=False, default=str)
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            # string literal
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            literal = text[i : j + 1]
            # is it a key? peek forward to next non-space
            k = j + 1
            while k < n and text[k] in " \t":
                k += 1
            is_key = k < n and text[k] == ":"
            cls = "k" if is_key else "s"
            out.append(f'<span class="{cls}">{html.escape(literal)}</span>')
            i = j + 1
        elif ch in "{}[],":
            out.append(f'<span class="p">{ch}</span>')
            i += 1
        elif ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n and (text[j].isdigit() or text[j] in ".eE+-"):
                j += 1
            out.append(f'<span class="n">{html.escape(text[i:j])}</span>')
            i = j
        elif text[i : i + 4] in ("true", "null") or text[i : i + 5] == "false":
            token = "false" if text[i : i + 5] == "false" else text[i : i + 4]
            out.append(f'<span class="b">{token}</span>')
            i += len(token)
        else:
            out.append(html.escape(ch))
            i += 1
    return "".join(out)


def _raw_block(data: Any) -> str:
    return f'<pre class="raw" id="raw">{_format_json(data)}</pre>'


# ─── renderers for each endpoint ─────────────────────────────────────────────
def render_metadata(data: dict) -> HTMLResponse:
    """Render /metadata as a styled dossier page."""
    tags_html = "".join(
        f'<span class="tag">{html.escape(t)}</span>' for t in data.get("tags", [])
    )
    tasks_html = "".join(
        f'<span class="chip">{html.escape(t)}</span>' for t in data.get("tasks", [])
    )
    inner = f"""
<div class="kv">
  <div class="key">Name</div>
  <div class="val"><code>{html.escape(str(data.get('name','')))}</code></div>

  <div class="key">Version</div>
  <div class="val">{html.escape(str(data.get('version','')))}</div>

  <div class="key">Author</div>
  <div class="val">{html.escape(str(data.get('author','')))}
    <a href="https://github.com/ROHITCRAFTSYT/-Metas-OpenEnv-2" target="_blank" rel="noopener" style="margin-left:12px">↗ source</a>
  </div>

  <div class="key">Description</div>
  <div class="val" style="max-width:72ch">{html.escape(str(data.get('description','')))}</div>

  <div class="key">Tasks ({len(data.get('tasks', []))})</div>
  <div class="val"><div class="chips">{tasks_html}</div></div>

  <div class="key">Tags</div>
  <div class="val"><div class="tags">{tags_html}</div></div>
</div>

{_raw_block(data)}
"""
    return HTMLResponse(
        _shell(
            title='Environment <em>Metadata</em>',
            eyebrow="/metadata",
            file_no="003.Ma",
            lede=(
                "OpenEnv runtime descriptor. Name, version, task catalogue and "
                "author, plus the tags judges grep for when they index "
                "<em>the submissions</em>."
            ),
            inner=inner,
        )
    )


def render_tasks(tasks: list[dict]) -> HTMLResponse:
    """Render /tasks as a numbered bibliography."""
    difficulty_class = {
        "easy": "e", "medium": "m", "hard": "h",
        "expert": "x", "super-hard": "x", "adaptive": "a",
    }
    rows = []
    for idx, t in enumerate(tasks, 1):
        diff = str(t.get("difficulty", "—")).lower()
        cls = difficulty_class.get(diff, "m")
        name = html.escape(str(t.get("name", t.get("id", "—"))))
        tid = html.escape(str(t.get("id", "—")))
        steps = t.get("max_steps", "—")
        rows.append(
            f'<div class="row">'
            f'<div class="code">TK·{idx:03d}</div>'
            f'<div class="name"><em>{name}</em></div>'
            f'<div class="diff {cls}">{html.escape(diff)}</div>'
            f'<div class="steps">{steps} steps</div>'
            f'<div class="mode"><code style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:.06em;background:var(--paper-2);border:.5px solid var(--hairline);padding:2px 6px">{tid}</code></div>'
            f'</div>'
        )
    inner = f"""
<div class="task-index">
  <div class="row head">
    <div>CODE</div><div>SCENARIO</div><div>DIFFICULTY</div><div>STEPS</div><div>TASK ID</div>
  </div>
  {''.join(rows)}
</div>

<div style="margin-top:36px;padding:22px 26px;background:var(--paper-2);border-left:3px solid var(--stamp);font-family:'Fraunces',serif;font-size:14px;line-height:1.6;color:var(--ink-soft);max-width:78ch">
  <strong style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--stamp);display:block;margin-bottom:8px">§ Usage</strong>
  Every task is seeded, deterministic, and grader-verifiable.
  Call <code style="font-family:'JetBrains Mono',monospace;background:var(--paper);padding:1px 6px;border:.5px solid var(--hairline)">POST /reset</code>
  with <code style="font-family:'JetBrains Mono',monospace;background:var(--paper);padding:1px 6px;border:.5px solid var(--hairline)">{{"task_id": "...", "seed": 42}}</code>
  to start an episode; <code style="font-family:'JetBrains Mono',monospace;background:var(--paper);padding:1px 6px;border:.5px solid var(--hairline)">POST /baseline</code>
  runs the scripted oracle end-to-end for a measurable reward floor.
</div>

{_raw_block(tasks)}
"""
    return HTMLResponse(
        _shell(
            title='Task <em>Catalogue</em>',
            eyebrow="/tasks",
            file_no="003.Tk",
            lede=(
                "Eight scenarios, from a single phishing email to a 250-step APT "
                "campaign. Each one is a <em>seeded, grader-verifiable</em> "
                "episode — same seed, same score, every time."
            ),
            inner=inner,
        )
    )


def render_themes(data: dict) -> HTMLResponse:
    """Render /themes/coverage as a manifest of hackathon themes + safeguards."""
    coverage = data.get("coverage", {})
    defenses = data.get("reward_hacking_defenses", [])
    rlvr_rlve = data.get("rlvr_rlve", {})

    covered_cards = []
    for i, (theme, ok) in enumerate(coverage.items(), 1):
        badge = '<span class="chip green">✓ covered</span>' if ok else '<span class="chip">· pending</span>'
        covered_cards.append(
            f'<div class="card">'
            f'<div class="n">{i:02d}</div>'
            f'<div class="k">Theme · {i:02d}</div>'
            f'<div class="v"><em>{html.escape(theme.replace("_", " "))}</em></div>'
            f'<div class="d">{badge}</div>'
            f'</div>'
        )

    def_rows = "".join(
        f'<div class="card">'
        f'<div class="n">{i:02d}</div>'
        f'<div class="k">Defense · {i:02d}</div>'
        f'<div class="v" style="font-family:\'JetBrains Mono\',monospace;font-size:15px;letter-spacing:.02em">{html.escape(d)}</div>'
        f'</div>'
        for i, d in enumerate(defenses, 1)
    )

    rlvr_block = ""
    if rlvr_rlve:
        rlvr_block = f"""
<h2 style="margin-top:56px">Two layers of <em>verification</em>.</h2>
<div class="grid-2">
  <div class="card">
    <div class="k">§ RLVR</div>
    <div class="v"><em>Verifiable rewards</em></div>
    <div class="d">Programmatic graders. See <code>{html.escape(str(rlvr_rlve.get('rlvr_verifiers', 'graders/')))}</code>.</div>
  </div>
  <div class="card">
    <div class="k">§ RLVE</div>
    <div class="v"><em>Adaptive environment</em></div>
    <div class="d">Red-team curriculum that rewrites itself. See <code>{html.escape(str(rlvr_rlve.get('rlve_adaptive_environment', 'scenarios/red_team_generator.py')))}</code>.</div>
  </div>
</div>
"""

    inner = f"""
<div class="eyebrow">Theme Coverage · {sum(1 for v in coverage.values() if v)} / {len(coverage)}</div>
<h2>A machine-checkable <em>manifest</em>.</h2>
<div class="grid-2" style="margin-top:22px">
{''.join(covered_cards)}
</div>

<h2 style="margin-top:56px">Six reward-hack <em>defenses</em>.</h2>
<div class="grid-2" style="margin-top:22px">
{def_rows}
</div>

{rlvr_block}

{_raw_block(data)}
"""
    return HTMLResponse(
        _shell(
            title='Theme <em>Coverage</em>',
            eyebrow="/themes/coverage",
            file_no="003.Th",
            lede=(
                "The machine-checkable manifest every hackathon judge wants: "
                "which <em>themes</em> are covered, which <em>defenses</em> "
                "are regression-tested, and where the RLVR/RLVE pieces live."
            ),
            inner=inner,
        )
    )


def render_state(data: dict) -> HTMLResponse:
    """Render /state as a snapshot dashboard.

    `data` is an EnvironmentState dump — it carries scalar counts
    (alert_count, classified_count, step_count, max_steps,
    cumulative_reward), not the nested alert_queue/investigations
    arrays. Reading the wrong keys produces a page of zeros even
    when an episode is live.
    """
    has_episode = bool(data.get("episode_id") and data.get("task_id"))
    step = data.get("step_count", 0) or 0
    max_steps = data.get("max_steps", 0) or 0
    done = data.get("done", False)
    cum = data.get("cumulative_reward", 0.0) or 0.0
    task = data.get("task_id") or "—"
    alert_count = data.get("alert_count", 0) or 0
    classified = data.get("classified_count", 0) or 0
    mode = data.get("episode_mode") or "tier1_solo"

    if not has_episode:
        status_chip = '<span class="chip">· idle</span>'
    elif done:
        status_chip = '<span class="chip red">· episode done</span>'
    else:
        status_chip = '<span class="chip green">· live</span>'

    step_label = f"{step} / {max_steps}" if max_steps else (str(step) if has_episode else "—")
    classified_label = f"{classified} / {alert_count}" if has_episode else "—"

    inner = f"""
<div class="grid-2">
  <div class="card"><div class="k">Task</div><div class="v"><em>{html.escape(str(task))}</em></div><div class="d">{status_chip}</div></div>
  <div class="card"><div class="k">Step</div><div class="v">{html.escape(step_label)}</div><div class="d">of the current episode</div></div>
  <div class="card"><div class="k">Cumulative Reward</div><div class="v">{cum:+.3f}</div><div class="d">episode-to-date total</div></div>
  <div class="card"><div class="k">Mode</div><div class="v">{html.escape(str(mode))}</div><div class="d">solo or team</div></div>
  <div class="card"><div class="k">Alerts</div><div class="v">{alert_count}</div><div class="d">in the queue</div></div>
  <div class="card"><div class="k">Classified</div><div class="v">{html.escape(classified_label)}</div><div class="d">investigations resolved</div></div>
</div>

{_raw_block(data)}
"""
    return HTMLResponse(
        _shell(
            title='Environment <em>State</em>',
            eyebrow="/state",
            file_no="003.St",
            lede=(
                "Current snapshot of the running episode — alert queue, "
                "investigations, reward ledger. Reset by <em>POST /reset</em>; "
                "advanced by <em>POST /step</em>."
            ),
            inner=inner,
        )
    )


def render_schema(data: dict) -> HTMLResponse:
    """Render /schema by name-listing the three models + embedding raw."""

    def pick_keys(obj: Any) -> Iterable[str]:
        if isinstance(obj, dict) and "properties" in obj:
            return list(obj["properties"].keys())
        return []

    def card(name: str, obj: Any) -> str:
        props = list(pick_keys(obj))
        chips = "".join(
            f'<span class="chip" style="margin:2px">{html.escape(p)}</span>'
            for p in props[:18]
        )
        more = (
            f'<span class="chip" style="margin:2px;border-style:dashed">+{len(props)-18} more</span>'
            if len(props) > 18
            else ""
        )
        return (
            f'<div class="card">'
            f'<div class="k">Model</div>'
            f'<div class="v"><em>{html.escape(name)}</em></div>'
            f'<div class="d" style="margin-top:12px"><div class="chips">{chips}{more}</div></div>'
            f'</div>'
        )

    cards = "".join(
        card(n, data.get(n, {})) for n in ("action", "observation", "state")
    )

    inner = f"""
<div class="grid-2" style="grid-template-columns:repeat(3,1fr)">
  {cards}
</div>

<div style="margin-top:36px;padding:22px 26px;background:var(--paper-2);border-left:3px solid var(--cobalt);font-family:'Fraunces',serif;font-size:14px;line-height:1.6;color:var(--ink-soft);max-width:78ch">
  <strong style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--cobalt);display:block;margin-bottom:8px">§ Schema</strong>
  Pydantic-derived JSON Schema for the three OpenEnv contract types. Full
  structure (types, constraints, enums) is in the raw block below — useful
  for generating typed clients or driving structured-decoding during GRPO.
</div>

{_raw_block(data)}
"""
    return HTMLResponse(
        _shell(
            title='JSON <em>Schema</em>',
            eyebrow="/schema",
            file_no="003.Sc",
            lede=(
                "The three OpenEnv contract types — <em>action</em>, "
                "<em>observation</em>, <em>state</em> — surfaced as JSON Schema "
                "for typed client generation."
            ),
            inner=inner,
        )
    )
