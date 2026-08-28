"""Build the dashboard from the analysis output.

`dashboard_body.html` is the template; every number in it comes from
`results.json`, which `analysis.py` writes. Nothing is typed in by hand, so the
page cannot drift from the model behind it.

    python analysis.py && python build_dashboard.py

Writes `index.html` — a single self-contained file (GitHub Pages serves it as-is).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "dashboard_body.html"
RESULTS = HERE / "results.json"
PAGE = HERE / "index.html"
BODY = HERE / "build" / "body.html"  # same content, no document wrapper

PLACEHOLDER = "/*__RESULTS__*/{}"

HEAD = """<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A credit default scorecard on 30,000 accounts: why accuracy is the wrong metric, what dropping the protected attributes costs, and where the decision threshold should actually sit.">
<meta name="color-scheme" content="light dark">
</head>
<body>
"""
FOOT = "</body>\n</html>\n"


def main() -> None:
    if not RESULTS.exists():
        raise SystemExit("results.json is missing — run `python analysis.py` first")

    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit(f"template has no {PLACEHOLDER} placeholder to fill")

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    # separators=(",", ":") keeps the payload small; the JSON is inlined into a
    # <script>, so guard the one sequence that could close it early.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = template.replace(PLACEHOLDER, payload)

    PAGE.write_text(HEAD + body + FOOT, encoding="utf-8")
    BODY.parent.mkdir(exist_ok=True)
    BODY.write_text(body, encoding="utf-8")

    kb = len(PAGE.read_bytes()) / 1024
    print(f"wrote {PAGE.name} ({kb:.0f} KB, self-contained) and {BODY.parent.name}/{BODY.name}")


if __name__ == "__main__":
    main()
