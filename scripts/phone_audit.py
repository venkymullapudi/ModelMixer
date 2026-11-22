"""Audit ModelMixer at phone width (375 px, touch): page-level horizontal overflow per view/state, plus screenshots.

Usage:  python3 scripts/phone_audit.py [repo_root] [screenshot_dir]
Needs:  pip install playwright   (local: your Google Chrome; CI: PW_CHANNEL="" uses Playwright's Chromium)
Rebuild dist/ first (python3 scripts/bundle.py). Exits 1 if any view overflows or the page logs an error.
"""
import os, pathlib, sys
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path('/tmp/modelmixer-phone')
OUT.mkdir(parents=True, exist_ok=True)
READY = "() => { const T = window.ModelMixer && window.ModelMixer.trained; return !!(T && T.experts) && !/Training/.test(document.querySelector('#statusText').textContent); }"
FIND = """() => {
  const W = document.documentElement.clientWidth, sw = document.documentElement.scrollWidth;
  const bad = [];
  for (const el of document.querySelectorAll('body *')) {
    if (!el.offsetParent && getComputedStyle(el).position !== 'fixed') continue;
    // elements inside intentional horizontal scrollers are fine
    let p = el.parentElement, scroller = false;
    while (p && p !== document.body) { const o = getComputedStyle(p).overflowX; if (o === 'auto' || o === 'scroll' || o === 'hidden') { scroller = true; break; } p = p.parentElement; }
    if (scroller) continue;
    const r = el.getBoundingClientRect();
    if (r.width && (r.right > W + 1 || r.left < -1)) bad.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}${el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.') : ''} [${Math.round(r.left)}→${Math.round(r.right)}]`);
  }
  return { W, sw, bad: bad.slice(0, 8), n: bad.length };
}"""


FAILS = []
errs = []


def check(pg, label, shot=None):
    pg.wait_for_timeout(500)
    r = pg.evaluate(FIND)
    flag = '✓' if r['sw'] <= r['W'] and not r['n'] else '✗'
    if flag == '✗': FAILS.append(label)
    print(f"{flag} {label:28s} scrollWidth {r['sw']} / {r['W']}  offenders {r['n']}")
    for b in r['bad']: print('     ', b)
    if shot: pg.screenshot(path=str(OUT / f'{shot}.png'), full_page=False)
    return r


with sync_playwright() as p:
    b = p.chromium.launch(channel=os.environ.get('PW_CHANNEL', 'chrome') or None, headless=True)
    ctx = b.new_context(viewport={"width": 375, "height": 812}, device_scale_factor=2, is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    pg.add_init_script("try { localStorage.setItem('modelmixer:tour', 'done') } catch (e) {}")  # no first-run tour in captures
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto((ROOT / 'dist/modelmixer.html').as_uri()); pg.wait_for_function(READY, timeout=60000)
    pg.add_style_tag(content=".toast{display:none!important}")
    check(pg, 'lab (top)', 'p_lab')
    pg.locator('#stageHead').scroll_into_view_if_needed(); check(pg, 'lab stage', 'p_stage')
    pg.evaluate("document.querySelector('[data-pin=\"pin\"]').click()"); pg.wait_for_timeout(300)
    pg.locator("#picker button", has_text="DARE").first.click(); pg.wait_for_timeout(800)
    pg.locator('#stageHead').scroll_into_view_if_needed(); check(pg, 'compare mode', 'p_compare')
    pg.locator('.cmp-box').scroll_into_view_if_needed(); check(pg, 'compare box', 'p_cmpbox')
    pg.evaluate("document.querySelector('[data-pin=\"unpin\"]').click()"); pg.wait_for_timeout(300)
    pg.locator("#picker button", has_text="SLERP").first.click(); pg.wait_for_timeout(600)
    pg.evaluate("document.querySelector('#gradToggle').click()"); pg.wait_for_selector('#heatSum p', timeout=30000)
    pg.locator('#layerPanel').scroll_into_view_if_needed(); check(pg, 'per-layer heat map', 'p_heat')
    pg.evaluate("document.querySelector('#gradToggle').click()"); pg.wait_for_timeout(300)
    pg.locator("#picker button", has_text="TIES").first.click(); pg.wait_for_selector('#sweepSum p', timeout=30000)
    pg.locator('#sweepPanel').scroll_into_view_if_needed(); check(pg, 'knob sensitivity', 'p_sweep')
    pg.locator('#insight').scroll_into_view_if_needed(); check(pg, 'insight', 'p_insight')
    pg.click('[data-tab="arena"]'); pg.wait_for_function("() => /methods compared/.test(document.querySelector('#statusText').textContent)", timeout=60000)
    check(pg, 'arena top', 'p_arena')
    pg.click('#robustSeeds [data-n="3"]'); pg.click('#runRobust'); pg.wait_for_selector('#h2h .verdict', timeout=120000)
    pg.locator('.robust-panel').scroll_into_view_if_needed(); check(pg, 'robustness panel', 'p_robust')
    pg.locator('#h2h').scroll_into_view_if_needed(); check(pg, 'head-to-head', 'p_h2h')
    pg.locator('#pathChart').scroll_into_view_if_needed(); check(pg, 'arena charts', 'p_charts')
    pg.click('[data-tab="atlas"]'); check(pg, 'atlas', 'p_atlas')
    pg.click('[data-tab="export"]'); check(pg, 'export', 'p_export')
    pg.click('#mkExample'); pg.click('#mkLoad'); pg.wait_for_timeout(500)
    pg.locator('.import-panel').scroll_into_view_if_needed(); check(pg, 'import panel', 'p_import')
    pg.click('[data-tab="lab"]'); pg.wait_for_timeout(300)
    pg.click('#openAdvisor'); pg.wait_for_function("() => /Re-ranked|agree|close|disagree/.test(document.querySelector('#evVerdict').textContent)", timeout=120000)
    check(pg, 'advisor dialog', 'p_advisor')
    pg.locator('#advRecs').scroll_into_view_if_needed(); check(pg, 'advisor recs', 'p_advrecs')
    pg.click('#closeAdvisor')
    pg.locator("#ingredients button.scenario", has_text="Digits").first.click(); pg.wait_for_timeout(500); pg.wait_for_function(READY, timeout=60000)
    pg.locator('#stageHead').scroll_into_view_if_needed(); check(pg, 'digits gallery', 'p_digits')
    print('errors:', errs or 'none')
    b.close()
sys.exit(1 if FAILS or errs else 0)
