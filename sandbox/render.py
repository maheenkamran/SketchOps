"""Render HTML -> PNG screenshot.
 - render_local(): Playwright in this process (used INSIDE the sandbox container)
 - render():       what the agent calls. Sends HTML to the sandbox container if SANDBOX_URL is set.
Kept free of other project imports so the sandbox image stays tiny.
"""
import io
import os

# Same defaults as shared/config.py (not imported here, to keep the sandbox image tiny)
VIEW_W = int(os.getenv("VIEWPORT_W", 1280))
VIEW_H = int(os.getenv("VIEWPORT_H", 800))
VENDOR_TAILWIND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "tailwind.js")


def _handler(route):
    url = route.request.url
    if url.startswith(("data:", "about:")):
        return route.continue_()
    if "tailwindcss" in url:
        if os.path.exists(VENDOR_TAILWIND):  # sandbox: serve vendored copy, no network needed
            return route.fulfill(status=200, content_type="application/javascript",
                                 body=open(VENDOR_TAILWIND, "rb").read())
        if os.getenv("ALLOW_TAILWIND_CDN", "1") == "1":  # local dev only
            return route.continue_()
    return route.abort()


def render_local(html: str, timeout_ms: int = 15000) -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": VIEW_W, "height": VIEW_H})
        ctx.route("**/*", _handler)
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        page.set_content(html, wait_until="load")
        page.wait_for_timeout(600)  # let Tailwind's JS apply styles
        png = page.screenshot(full_page=True)
        browser.close()
    return png


def _mock_png() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (VIEW_W, VIEW_H), "white")
    ImageDraw.Draw(img).text((20, 20), "MOCK RENDER", fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render(html: str) -> bytes:
    if os.getenv("MOCK_MODE", "0").lower() in ("1", "true"):
        return _mock_png()
    url = os.getenv("SANDBOX_URL", "")
    if url:
        import requests

        r = requests.post(url.rstrip("/") + "/render", data=html.encode("utf-8"),
                          headers={"Content-Type": "text/html"}, timeout=60)
        r.raise_for_status()
        return r.content
    return render_local(html)
