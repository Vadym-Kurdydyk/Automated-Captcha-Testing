# End-to-end captcha strength test: a real browser (Playwright) opens our challenge page,
# the model reads the captcha, types the answer and submits; we record the pass rate per difficulty.
# This measures how well OUR captcha holds up against a trained model.
#
#   python server.py                                   # in another terminal, or use --start-server
#   python e2e_captcha_test.py --rounds 50
#   python e2e_captcha_test.py --start-server --headed --rounds 5
#   python e2e_captcha_test.py --screenshot            # read what the browser rendered, not the raw image

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

from PIL import Image
from playwright.sync_api import sync_playwright

from predict import CaptchaReader, default_checkpoint


def wait_for_server(url: str, timeout: float = 60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "/api/info", timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"Server at {url} did not come up")


def captcha_image(page, screenshot: bool) -> Image.Image:
    img = page.locator("#captcha")
    if screenshot:
        return Image.open(io.BytesIO(img.screenshot()))
    src = img.get_attribute("src")
    return Image.open(io.BytesIO(base64.b64decode(src.split(",", 1)[1])))


def main():
    p = argparse.ArgumentParser(description="Playwright pass-rate test against our own captcha page")
    p.add_argument("--checkpoint", default=None, help="default: newest model, same pick as server.py")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--rounds", type=int, default=20, help="captchas per difficulty level")
    p.add_argument("--levels", default="1,2,3,4,5")
    p.add_argument("--screenshot", action="store_true")
    p.add_argument("--headed", action="store_true")
    p.add_argument("--start-server", action="store_true", help="launch server.py for the duration of the test")
    p.add_argument("--out", default="e2e_results.json")
    args = p.parse_args()
    args.checkpoint = args.checkpoint or default_checkpoint()

    server = None
    if args.start_server:
        env = {**os.environ, "CAPTCHA_MODEL": args.checkpoint}
        server = subprocess.Popen([sys.executable, "server.py"], env=env)
    try:
        wait_for_server(args.url)
        reader = CaptchaReader(args.checkpoint)
        levels = [int(x) for x in args.levels.split(",")]
        stats = defaultdict(lambda: {"passed": 0, "total": 0, "failures": []})

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not args.headed)
            page = browser.new_page()
            for level in levels:
                page.goto(f"{args.url}/challenge?difficulty={level}")
                for _ in range(args.rounds):
                    page.wait_for_function("document.querySelector('#captcha').complete && "
                                           "document.querySelector('#captcha').naturalWidth > 0 && "
                                           "document.querySelector('#captcha').dataset.challengeId")
                    challenge_id = page.locator("#captcha").get_attribute("data-challenge-id")
                    pred = reader.read([captcha_image(page, args.screenshot)])[0]
                    guess = str(pred.answer) if reader.is_math else pred.text
                    page.fill("#answer", guess)
                    page.click("#submit")
                    page.wait_for_function("document.querySelector('#result').textContent.length > 0")
                    result = page.text_content("#result")
                    s = stats[level]
                    s["total"] += 1
                    if result.startswith("Correct"):
                        s["passed"] += 1
                    else:
                        s["failures"].append({"guess": guess, "result": result, "confidence": round(pred.confidence, 3)})
                    # Load the next captcha and wait until it actually replaced the old one
                    page.click("#new")
                    page.wait_for_function(f"document.querySelector('#captcha').dataset.challengeId !== '{challenge_id}'")
                s = stats[level]
                print(f"L{level}: model passed {s['passed']}/{s['total']} ({100 * s['passed'] / s['total']:.0f}%)")
            browser.close()

        total = sum(s["passed"] for s in stats.values()), sum(s["total"] for s in stats.values())
        print(f"Overall: {total[0]}/{total[1]} ({100 * total[0] / total[1]:.1f}%) - "
              f"higher = weaker captcha against this model")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": args.checkpoint, "screenshot": args.screenshot, "levels": stats}, f, indent=2)
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    main()
