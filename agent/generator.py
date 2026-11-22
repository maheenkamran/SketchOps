"""Code generation: tries the Gemini models in config.GEMINI_MODELS in order (free tier = 20 requests/day per model).
Groq is an optional last resort (GROQ_FALLBACK=1); its free tier is too small for whole pages."""
import base64
import logging
import os
import re

from shared import config
from shared.images import mime as _mime
from shared.images import shrink as _shrink

log = logging.getLogger("generator")

MOCK_HTML = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.5"></script></head>
<body class="p-6 font-sans"><h1 class="text-2xl font-bold mb-4">Mock App</h1>
<input class="border p-2 w-full mb-3" placeholder="Email"><button class="bg-blue-600 text-white px-4 py-2 rounded">Login</button>
</body></html>"""


def extract_html(text: str) -> str:
    """Take the code out of a ```html ... ``` block (also works if the closing ``` is missing)."""
    m = re.search(r"```(?:html)?\s*(.*?)(?:```|$)", text or "", re.S)
    return (m.group(1) if m else text or "").strip()


_out_of_quota = set()  # Gemini models that hit their daily free-tier limit in this run


def _is_daily_quota(e) -> bool:
    """429 with a per-day quota (resets tomorrow). A per-minute 429 or a 503 'overloaded' is only temporary."""
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s and "PerDay" in s


def _gemini(text, images, model):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    contents = [text] + [types.Part.from_bytes(data=i, mime_type=_mime(i)) for i in images]
    r = client.models.generate_content(model=model, contents=contents)
    if not r.text:  # e.g. blocked by a safety filter -> treat as a failure so Groq takes over
        raise RuntimeError("Gemini returned no text")
    tokens = getattr(getattr(r, "usage_metadata", None), "total_token_count", 0) or 0
    return r.text, tokens


def _groq(text, images):
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    content = [{"type": "text", "text": text}]
    for i in images:
        b64 = base64.b64encode(i).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{_mime(i)};base64,{b64}"}})
    r = client.chat.completions.create(
        model=config.GROQ_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=8192,  # desktop pages are long; 4096 could cut the HTML off
    )
    return r.choices[0].message.content, getattr(r.usage, "total_tokens", 0) or 0


def generate(prompt: str, sketch: bytes, prev_html: str = None, screenshot: bytes = None) -> dict:
    """Returns dict(html, model_used, api_calls, fallback_used, tokens)."""
    if config.MOCK_MODE:
        return dict(html=MOCK_HTML, model_used="mock", api_calls=1, fallback_used=False, tokens=0)

    text = prompt
    images = [_shrink(sketch)]
    text += "\n\nImage 1 is the hand-drawn sketch."
    if prev_html and screenshot:
        images.append(_shrink(screenshot))
        text += "\nImage 2 is a screenshot of the CURRENT code rendered.\n\nCURRENT CODE:\n" + prev_html

    api_calls, errors = 0, []
    for model in [m for m in config.GEMINI_MODELS if m not in _out_of_quota]:
        try:
            api_calls += 1
            out, tokens = _gemini(text, images, model)
            break
        except Exception as e:  # quota, overloaded, model not available, network ...
            if _is_daily_quota(e):
                _out_of_quota.add(model)  # don't try it again until the process restarts
            log.warning("%s failed (%s) -> trying the next model", model, str(e)[:120])
            errors.append(f"{model}: {str(e)[:80]}")
    else:  # no Gemini model worked
        if not config.GROQ_FALLBACK:
            raise RuntimeError("all Gemini models failed or are out of quota: " + " | ".join(errors))
        api_calls += 1
        out, tokens = _groq(text, images)
        model = config.GROQ_VISION_MODEL
    return dict(html=extract_html(out), model_used=model, api_calls=api_calls,
                fallback_used=model != config.GEMINI_MODELS[0], tokens=tokens)
