#!/usr/bin/env python3
"""Inline css/ and js/ into a single self-contained HTML file: dist/modelmixer.html"""
import pathlib, re

root = pathlib.Path(__file__).resolve().parent.parent
html = (root / "index.html").read_text()
html = html.replace('<link rel="stylesheet" href="css/styles.css" />',
                    "<style>\n" + (root / "css/styles.css").read_text() + "\n</style>")
ENGINE = {"js/nn.js", "js/data.js", "js/methods.js", "js/merge.js", "js/lab.js"}
def inline(m):
    # engine scripts are tagged so the app can rebuild them inside a Web Worker (no duplicate copy needed)
    tag = '<script data-engine>' if m.group(1) in ENGINE else '<script>'
    return tag + "\n" + (root / m.group(1)).read_text() + "\n</script>"
html = re.sub(r'<script src="(js/[^"]+)"></script>', inline, html)
out = root / "dist" / "modelmixer.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"wrote {out} ({len(html)/1024:.0f} KB)")

# Optional: `python3 scripts/bundle.py --tests` also writes tests/_browser.html, a single-file page that
# runs tests/suite.js in a browser (handy where Node isn't installed). It is git-ignored.
import sys
if "--tests" in sys.argv:
    eng = "".join("<script>\n" + (root / f"js/{f}.js").read_text() + "\n</script>\n" for f in ["nn", "data", "methods", "merge", "lab", "advisor", "share", "mkimport", "report", "gallery", "transfer-snapshot"])
    eng += "<script>\n" + (root / "tests/suite.js").read_text() + "\n</script>\n"
    eng += """<script>
setTimeout(() => { const t0 = performance.now(); const res = MMTests.run(MM); window.RES = res;
  document.body.innerHTML = '<pre>' + res.map(r => (r.ok ? '\\u2713 ' : '\\u2717 ') + r.name + (r.detail ? '  \\u2014 ' + r.detail : '')).join('\\n')
    + '\\n\\n' + res.filter(r => r.ok).length + '/' + res.length + ' passed in ' + ((performance.now() - t0) / 1000).toFixed(1) + 's</pre>'; }, 50);
</script>"""
    t = root / "tests" / "_browser.html"
    t.write_text('<!doctype html><meta charset="utf-8"><title>ModelMixer tests</title><body>running…</body>' + eng)
    print(f"wrote {t}")
