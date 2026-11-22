"""Regenerate the README screenshots in docs/screenshots/ (WebP, max 1800 px).

Usage:  python3 scripts/screenshots.py
Needs:  pip install playwright pillow   (uses your installed Google Chrome via channel="chrome";
        or run `playwright install chromium` and drop the channel argument)
Rebuild dist/ first (python3 scripts/bundle.py): the shots are taken from dist/modelmixer.html.
"""
import os, pathlib, sys
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
URL = (ROOT / "dist" / "modelmixer.html").as_uri()

READY = """() => { const T = window.ModelMixer && window.ModelMixer.trained;
  return !!(T && T.experts) && !/Training/.test(document.querySelector('#statusText').textContent); }"""


def wait_ready(page, kind=None):
    page.wait_for_function(READY, timeout=60000)
    if kind:
        page.wait_for_function(f"() => window.ModelMixer.trained.tasks[0].kind === '{kind}'", timeout=60000)
        page.wait_for_function(READY, timeout=60000)
    page.wait_for_timeout(600)


def pick_method(page, label):
    page.locator("#picker button", has_text=label).filter(has_text=label).first.click()
    page.wait_for_timeout(700)


def pick_scenario(page, label, kind=None):
    page.locator("#ingredients button.scenario", has_text=label).first.click()
    page.wait_for_timeout(500)
    wait_ready(page, kind)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="dark")
    page = ctx.new_page()
    page.add_init_script("try { localStorage.setItem('modelmixer:tour', 'done') } catch (e) {}")  # no first-run tour in captures
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(URL)
    wait_ready(page)
    # captures only: keep the sticky top bar from overlapping element shots, and hide toasts
    page.add_style_tag(content=".topbar{position:static!important} .toast{display:none!important}")

    # 1. Mix Lab, Two skills, TIES
    pick_method(page, "TIES")
    page.screenshot(path=str(OUT / "lab.png"))
    print("lab")

    # 2. Compare mode: pin TIES, switch to DARE
    page.click('[data-pin="pin"]')
    page.wait_for_timeout(300)
    pick_method(page, "DARE")
    page.wait_for_timeout(800)
    page.evaluate("document.querySelector('.stage').scrollIntoView({block: 'start'}); window.scrollBy(0, -12)")
    page.wait_for_timeout(300)
    page.screenshot(path=str(OUT / "compare.png"))
    page.click('[data-pin="unpin"]')
    page.wait_for_timeout(300)
    print("compare")

    # 3. Knob sensitivity with the 3-seed band (TIES density)
    pick_method(page, "TIES")
    page.wait_for_selector("#sweepSum p", timeout=30000)
    page.click("#sweepSeeds")
    page.wait_for_selector("#sweepSum p", timeout=60000)
    page.locator("#sweepPanel").screenshot(path=str(OUT / "sweep.png"))
    page.click("#sweepSeeds")
    print("sweep")

    # 4. Recipe advisor with measured evidence
    page.click("#openAdvisor")
    # show the measured re-ranking: no shared base, any serving → rules pick Ensemble, measurement promotes another
    page.click('[data-adv="base"] [data-v="no"]')
    page.click('[data-adv="deploy"] [data-v="any"]')
    page.wait_for_function("() => /Re-ranked|agree|close|disagree/.test(document.querySelector('#evVerdict').textContent)", timeout=90000)
    page.wait_for_timeout(300)
    page.locator("#advisor").screenshot(path=str(OUT / "advisor.png"))
    page.click('[data-adv="base"] [data-v="yes"]')
    page.click('[data-adv="deploy"] [data-v="single"]')
    page.click("#closeAdvisor")
    print("advisor")

    # 5. Arena: leaderboard + multi-seed robustness
    page.click('[data-tab="arena"]')
    page.wait_for_function("() => /methods compared/.test(document.querySelector('#statusText').textContent)", timeout=60000)
    page.click("#runRobust")
    page.wait_for_selector("#robust table", timeout=120000)
    page.wait_for_timeout(300)
    page.locator(".robust-panel").screenshot(path=str(OUT / "robustness.png"))
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(200)
    page.screenshot(path=str(OUT / "arena.png"))
    print("arena")

    # 6. Digits scenario (image-like inputs), Linear
    page.click('[data-tab="lab"]')
    page.wait_for_timeout(300)
    pick_scenario(page, "Digits", "digits")
    pick_method(page, "Linear")
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)
    page.screenshot(path=str(OUT / "digits.png"))
    print("digits")

    browser.close()
    print("errors:", errors or "none")

# shrink and convert: full-colour WebP keeps the UI colours exact at a fraction of PNG size
for p in sorted(OUT.glob("*.png")):
    im = Image.open(p).convert("RGB")
    im.thumbnail((1800, 1800), Image.LANCZOS)
    im.save(p.with_suffix(".webp"), "WEBP", quality=90, method=6)
    p.unlink()
    print(p.with_suffix(".webp").name, im.size, os.path.getsize(p.with_suffix(".webp")) // 1024, "KB")
