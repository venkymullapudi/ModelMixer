"""Accessibility audit: run axe-core on every ModelMixer view and fail on any violation.

Usage:  python3 scripts/axe_audit.py [repo_root]
Needs:  pip install playwright   (local: your Google Chrome; CI: PW_CHANNEL="" uses Playwright's Chromium)
Rebuild dist/ first (python3 scripts/bundle.py). Exits 1 if any rule is violated or the page logs an error.
"""
import os, pathlib, sys
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent
CHANNEL = os.environ.get('PW_CHANNEL', 'chrome') or None
AXE = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.9.1/axe.min.js"
READY = "() => { const T = window.ModelMixer && window.ModelMixer.trained; return !!(T && T.experts) && !/Training/.test(document.querySelector('#statusText').textContent); }"


def audit(pg, label, include=None):
    res = pg.evaluate("""async (inc) => {
      const r = await axe.run(inc ? { include: [inc] } : document, { resultTypes: ['violations'] });
      return r.violations.map(v => ({ id: v.id, impact: v.impact, n: v.nodes.length, help: v.help, eg: v.nodes.slice(0, 3).map(x => x.target.join(' ')) }));
    }""", include)
    return label, res


with sync_playwright() as p:
    b = p.chromium.launch(channel=CHANNEL, headless=True)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script("try { localStorage.setItem('modelmixer:tour', 'done') } catch (e) {}")  # audit the app, not the tour
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto((ROOT / "dist/modelmixer.html").as_uri())
    pg.wait_for_function(READY, timeout=120000)
    pg.add_script_tag(url=AXE)
    pg.wait_for_function("() => !!window.axe", timeout=30000)
    out = [audit(pg, "lab")]
    pg.click('[data-tab="arena"]'); pg.wait_for_function("() => /methods compared/.test(document.querySelector('#statusText').textContent)", timeout=120000)
    out.append(audit(pg, "arena"))
    pg.click('[data-tab="atlas"]'); pg.wait_for_timeout(400); out.append(audit(pg, "atlas"))
    pg.click('[data-tab="export"]'); pg.wait_for_timeout(400); out.append(audit(pg, "export"))
    pg.click('[data-tab="lab"]'); pg.wait_for_timeout(300)
    pg.click("#openAdvisor"); pg.wait_for_timeout(600); out.append(audit(pg, "advisor", "#advisor"))
    pg.click("#closeAdvisor"); pg.wait_for_timeout(200)
    pg.click("#openReport"); pg.wait_for_timeout(600); out.append(audit(pg, "report", "#reportDlg"))
    b.close()

seen = {}
for label, vs in out:
    for v in vs:
        seen.setdefault(v["id"], {"impact": v["impact"], "help": v["help"], "where": [], "eg": v["eg"]})
        seen[v["id"]]["where"].append(f"{label}:{v['n']}")
for k, v in sorted(seen.items(), key=lambda kv: ["critical", "serious", "moderate", "minor"].index(kv[1]["impact"] or "minor")):
    print(f"[{v['impact']}] {k} — {v['help']}  ({', '.join(v['where'])})")
    for e in v["eg"]:
        print("     e.g.", e[:110])
print("TOTAL rules violated:", len(seen))
print("errors:", errs or "none")
sys.exit(1 if seen or errs else 0)
