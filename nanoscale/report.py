"""Render the transfer analysis as Markdown and self-contained HTML."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

VERDICT_TEXT = {
    "hurts_to_remove": "removing it hurts",
    "helps_to_remove": "removing it helps",
    "practically_equal": "practically equal",
    "unresolved": "unresolved",
}


def _fmt(x: Any, spec: str = ".4f") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:  # NaN
            return "n/a"
        return format(x, spec)
    return str(x)


def to_markdown(a: dict) -> str:
    L: list[str] = []
    L.append("# nanoscale transfer analysis\n")
    L.append(f"Metric: `{a['metric']}`. Equivalence margin: "
             f"{a['equivalence_margin']} nats/token. Runs: {a['n_runs']}.\n")

    proto = a["protocol"]
    if not proto["consistent"]:
        L.append("> **Warning:** runs do not share one protocol/eval set/tokenizer. "
                 "Pooling them is invalid.\n")
        L.append(f"> protocol hashes: {proto['protocol_hashes']}\n")
        L.append(f"> eval-set hashes: {proto['eval_set_hashes']}\n")
    else:
        L.append("Protocol, evaluation set and tokenizer are consistent across runs.\n")

    L.append("\n## Seed noise (the floor every effect must clear)\n")
    L.append("| scale | n | baseline mean | sd | range |")
    L.append("|---|---|---|---|---|")
    for s, n in a["seed_noise"].items():
        L.append(f"| {s} | {n.get('n')} | {_fmt(n.get('mean'))} | "
                 f"{_fmt(n.get('sd'))} | {_fmt(n.get('range'))} |")

    L.append("\n## Paired effects vs baseline\n")
    L.append("Positive delta means the recipe is worse than baseline, so the removed "
             "component was helping. Pairing is by seed.\n")
    for scale, effects in a["paired_effects"].items():
        L.append(f"\n### Scale {scale}\n")
        L.append("| recipe | n | mean delta | 95% CI | verdict |")
        L.append("|---|---|---|---|---|")
        for recipe, e in sorted(effects.items(), key=lambda kv: -kv[1]["mean_delta"]):
            L.append(f"| {recipe} | {e['n_pairs']} | {_fmt(e['mean_delta'])} | "
                     f"[{_fmt(e['ci_low'])}, {_fmt(e['ci_high'])}] | "
                     f"{VERDICT_TEXT.get(e['verdict'], e['verdict'])} |")

    L.append("\n## Selection regret\n")
    L.append("Pick the best recipe at the small scale, then read its loss at the "
             "large scale. Zero regret means the cheap experiment chose correctly.\n")
    L.append("| small | large | chosen | best | regret | correct |")
    L.append("|---|---|---|---|---|---|")
    for r in a["regret"]:
        L.append(f"| {r['small']} | {r['large']} | {r.get('chosen_at_small','n/a')} | "
                 f"{r.get('best_at_large','n/a')} | {_fmt(r.get('regret'))} | "
                 f"{r.get('correct_selection','n/a')} |")

    L.append("\n## Selection probability (seed resampling)\n")
    L.append("| small | large | P(small pick == large best) | mean regret | 95% CI |")
    L.append("|---|---|---|---|---|")
    for p in a["selection_probability"]:
        ci = p.get("regret_ci")
        ci_s = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "n/a"
        L.append(f"| {p['small']} | {p['large']} | {_fmt(p.get('p_correct'), '.3f')} | "
                 f"{_fmt(p.get('mean_regret'))} | {ci_s} |")

    L.append("\n## Rank transfer (descriptive)\n")
    L.append("| small | large | recipes | Spearman | Kendall tau | note |")
    L.append("|---|---|---|---|---|---|")
    for t in a["rank_transfer"]:
        note = "underpowered (few recipes)" if t.get("underpowered") else ""
        L.append(f"| {t['small']} | {t['large']} | {t['n_recipes']} | "
                 f"{_fmt(t['spearman'], '.3f')} | {_fmt(t['kendall_tau'], '.3f')} | {note} |")

    L.append("\n## Effect trajectory across scale\n")
    L.append("| recipe | trajectory | per-scale mean deltas |")
    L.append("|---|---|---|")
    for recipe, t in a["trajectories"].items():
        per = ", ".join(f"{s}={_fmt(e['mean_delta'])}" for s, e in t["per_scale"].items())
        L.append(f"| {recipe} | {t['trajectory']} | {per} |")

    L.append("\n---\n")
    L.append("One-at-a-time flips measure conditional effects given the rest of the "
             "baseline, not independent contributions. Rank correlations over a handful "
             "of recipes are descriptive, not tests. `unresolved` means the data could "
             "not distinguish an effect from none; it is not evidence of no effect.\n")
    return "\n".join(L)


_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:60rem;
margin:2rem auto;padding:0 1rem;line-height:1.5;color:#111}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}
th{background:#f4f4f4}
code{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}
blockquote{border-left:4px solid #e5b567;background:#fdf6e3;margin:1rem 0;padding:.6rem 1rem}
h1,h2,h3{line-height:1.25}
.wrap{overflow-x:auto}
@media (prefers-color-scheme:dark){body{background:#111;color:#eee}
th{background:#222}th,td{border-color:#333}code{background:#222}
blockquote{background:#2a2415;border-color:#8a6d3b}}
"""


def to_html(a: dict) -> str:
    """Very small Markdown-subset renderer: tables, headings, blockquotes, paragraphs."""
    md = to_markdown(a)
    body: list[str] = []
    in_table = False
    for raw in md.split("\n"):
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue  # separator row
            if not in_table:
                body.append('<div class="wrap"><table>')
                in_table = True
                body.append("<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cells) + "</tr>")
            else:
                body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table></div>")
            in_table = False
        if not line:
            continue
        if line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        elif line.startswith("---"):
            body.append("<hr>")
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table></div>")
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>nanoscale transfer analysis</title><style>{_CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")


def write_report(a: dict, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, html_path = out_dir / "report.md", out_dir / "report.html"
    md_path.write_text(to_markdown(a), encoding="utf-8")
    html_path.write_text(to_html(a), encoding="utf-8")
    return {"markdown": md_path, "html": html_path}
