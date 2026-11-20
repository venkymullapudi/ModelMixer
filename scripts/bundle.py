#!/usr/bin/env python3
"""Inline css/ and js/ into a single self-contained HTML file: dist/modelmixer.html"""
import pathlib, re

root = pathlib.Path(__file__).resolve().parent.parent
html = (root / "index.html").read_text()
html = html.replace('<link rel="stylesheet" href="css/styles.css" />',
                    "<style>\n" + (root / "css/styles.css").read_text() + "\n</style>")
def inline(m):
    return "<script>\n" + (root / m.group(1)).read_text() + "\n</script>"
html = re.sub(r'<script src="(js/[^"]+)"></script>', inline, html)
out = root / "dist" / "modelmixer.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"wrote {out} ({len(html)/1024:.0f} KB)")
