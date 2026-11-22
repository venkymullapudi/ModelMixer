"""Keyboard-only audit: build, pin, compare and export a recipe using only Tab, arrows, Enter, Space and Esc.

Every control is reached by pressing Tab (never clicked or focused directly). Each focus stop must show a
visible focus indicator. Exits 1 if a step can't be reached or done, or the page logs an error.

Usage:  python3 scripts/keyboard_audit.py [repo_root]
Needs:  pip install playwright   (local: your Google Chrome; CI: PW_CHANNEL="" uses Playwright's Chromium)
Rebuild dist/ first (python3 scripts/bundle.py).
"""
import os, pathlib, sys
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent
CHANNEL = os.environ.get('PW_CHANNEL', 'chrome') or None
READY = "() => { const T = window.ModelMixer && window.ModelMixer.trained; return !!(T && T.experts) && !/Training/.test(document.querySelector('#statusText').textContent); }"
MAX_TABS = 400
FAILS, errs = [], []

# true if the focused element draws a focus ring (outline or box-shadow), else a description of it
FOCUS_INFO = """() => {
  const e = document.activeElement;
  if (!e || e === document.body) return { ok: false, what: 'body' };
  const cs = getComputedStyle(e);
  const ring = (cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0) || (cs.boxShadow && cs.boxShadow !== 'none');
  const what = e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.dataset.method ? `[data-method=${e.dataset.method}]` : '')
    + (e.dataset.scenario ? `[data-scenario=${e.dataset.scenario}]` : '') + (e.dataset.param ? `[data-param=${e.dataset.param}]` : '')
    + (e.dataset.pin ? `[data-pin=${e.dataset.pin}]` : '') + (e.dataset.tab ? `[data-tab=${e.dataset.tab}]` : '');
  return { ok: ring, what };
}"""


def step(name, ok, detail=''):
    print(('✓ ' if ok else '✗ ') + name + (f'  — {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def tab_to(pg, selector, name, back=False):
    """Press Tab (or Shift+Tab) until the focused element matches selector. Checks every stop shows focus."""
    no_ring, seen = [], []
    for n in range(1, MAX_TABS + 1):
        pg.keyboard.press('Shift+Tab' if back else 'Tab')
        info = pg.evaluate(FOCUS_INFO)
        seen.append(info['what'])
        if not info['ok'] and info['what'] != 'body':
            no_ring.append(info['what'])
        if pg.evaluate("(s) => !!document.activeElement && document.activeElement.matches(s)", selector):
            step(f'reach {name} by keyboard', True, f'{n} Tab press{"es" if n > 1 else ""}')
            if no_ring:
                step(f'visible focus on the way to {name}', False, ', '.join(sorted(set(no_ring))[:6]))
            return True
    return step(f'reach {name} by keyboard', False, f'not reached in {MAX_TABS} presses; stops: {" ".join(dict.fromkeys(seen))[:400]}')


def run(pg):
    pg.goto((ROOT / 'dist/modelmixer.html').as_uri())
    pg.wait_for_function(READY, timeout=120000)

    # 1. scenario: Tab to "Conflicting skills" (overlap) and press Enter → retrains
    if tab_to(pg, '[data-scenario="overlap"]', 'the Overlap scenario'):
        pg.keyboard.press('Enter')
        pg.wait_for_timeout(300); pg.wait_for_function(READY, timeout=120000)
        step('Enter picks the scenario', pg.evaluate("() => document.querySelector('[data-scenario=\"overlap\"]').classList.contains('active')"))

    # 2. method: Tab into the picker, arrow to TIES, press Enter
    if tab_to(pg, '#picker button.chip', 'the method picker'):
        for _ in range(30):
            if pg.evaluate("() => document.activeElement.dataset.method") == 'ties':
                break
            pg.keyboard.press('ArrowRight')
        pg.keyboard.press('Enter'); pg.wait_for_timeout(300)
        step('arrows + Enter pick TIES', pg.evaluate("() => (document.querySelector('#picker .chip.active') || {}).dataset?.method") == 'ties')

    # 3. pin the recipe as A
    if tab_to(pg, '[data-pin="pin"]', 'the Pin button'):
        pg.keyboard.press('Enter'); pg.wait_for_timeout(300)
        step('Enter pins recipe A', pg.evaluate("() => !document.querySelector('#pinPane').hidden"))

    # 4. change a knob with the arrow keys, then undo it with the keyboard shortcut
    if tab_to(pg, '#controls input[type=range][data-param]', 'a recipe knob'):
        before = pg.evaluate("() => document.activeElement.value")
        for _ in range(3):
            pg.keyboard.press('ArrowRight')
        pg.wait_for_timeout(200)  # inside the 700 ms undo window: the pin step must survive the knob edit
        after = pg.evaluate("() => document.activeElement.value")
        step('arrow keys move the knob', after != before, f'{before} → {after}')
        param = pg.evaluate("() => document.activeElement.dataset.param")
        pg.keyboard.press('Control+z'); pg.wait_for_timeout(300)
        undone = pg.evaluate("(p) => document.querySelector(`#controls input[data-param=\"${p}\"]`).value", param)
        step('Ctrl+Z undoes the knob change', undone == before, f'now {undone}')
        step('focus stays on the knob after undo', pg.evaluate("(p) => document.activeElement.dataset.param === p", param))
        step('undo keeps the pin', pg.evaluate("() => !document.querySelector('#pinPane').hidden"))
        # nudge it again (focus is still on it) so B differs from the pinned A
        for _ in range(4):
            pg.keyboard.press('ArrowRight')
        pg.wait_for_timeout(300)

    # 5. compare: the paired "is B's edge real?" check
    if tab_to(pg, '#edgeBtn', "the compare edge check", back=True):
        pg.keyboard.press('Enter')
        pg.wait_for_function("() => !!document.querySelector('#edge .verdict')", timeout=180000)
        step('Enter runs the multi-seed edge check', True, pg.inner_text('#edge .verdict')[:80])

    # 6. tabs: Tab to the selected tab, arrow to Export (roving tab stop)
    if tab_to(pg, '#tabs [aria-selected="true"]', 'the tab list', back=True):
        for _ in range(4):
            if pg.evaluate("() => document.activeElement.dataset.tab") == 'export':
                break
            pg.keyboard.press('ArrowRight')
        pg.wait_for_timeout(300)
        step('arrows open the Export tab', pg.evaluate("() => document.querySelector('#tab-export').classList.contains('active')"))
        yaml = pg.inner_text('#yaml')
        step('export config reflects the recipe', 'merge_method: ties' in yaml, 'merge_method: ties' if 'merge_method: ties' in yaml else yaml[:80])
        if tab_to(pg, '#copyYaml', 'Copy config'):
            pg.keyboard.press('Enter'); pg.wait_for_timeout(300)
            step('Enter on Copy gives feedback', bool(pg.inner_text('#toast').strip()), pg.inner_text('#toast').strip()[:60])

    # 7. dialogs: open the advisor, close it with Esc, focus returns to the opener
    if tab_to(pg, '#openAdvisor', 'the Recipe advisor button', back=True):
        pg.keyboard.press('Enter'); pg.wait_for_timeout(400)
        step('Enter opens the advisor', pg.evaluate("() => document.querySelector('#advisor').open"))
        inside = pg.evaluate("() => document.querySelector('#advisor').contains(document.activeElement)")
        step('focus moves into the advisor', inside)
        pg.keyboard.press('Escape'); pg.wait_for_timeout(300)
        step('Esc closes the advisor', not pg.evaluate("() => document.querySelector('#advisor').open"))
        step('focus returns to the opener', pg.evaluate("() => document.activeElement && document.activeElement.id === 'openAdvisor'"))


with sync_playwright() as p:
    b = p.chromium.launch(channel=CHANNEL, headless=True)
    ctx = b.new_context(viewport={'width': 1440, 'height': 900})
    ctx.add_init_script("try { localStorage.setItem('modelmixer:tour', 'done') } catch (e) {}")
    pg = ctx.new_page()
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
    try:
        run(pg)
    except Exception as e:  # a timeout means a keyboard action didn't do what it should
        step('flow completed', False, str(e).splitlines()[0][:160])
    b.close()

print('errors:', errs or 'none')
print('failed steps:', len(FAILS))
sys.exit(1 if FAILS or errs else 0)
