"""LLM-as-judge: does the screenshot contain the elements drawn in the sketch? (Llama via Groq)"""
import base64
import json
import logging
import os
import re
from typing import List, Optional

from shared import config
from shared.images import mime, shrink

log = logging.getLogger("judge")

PROMPT = """You are a strict UI reviewer. Image 1 is a hand-drawn sketch of a webpage.
Image 2 is a screenshot of the webpage that was built from it.
{extra}
List the UI elements visible in the sketch (buttons, inputs, headings, lists, nav, images, ...).
For each, decide if the screenshot contains it in roughly the same place/layout.
Reply ONLY with JSON: {{"elements": [{{"name": str, "present": bool}}], "coverage": float between 0 and 1}}"""


def _img(b: bytes):
    b = shrink(b)  # Groq rejects very large images
    return {"type": "image_url", "image_url": {"url": f"data:{mime(b)};base64,{base64.b64encode(b).decode()}"}}


def judge(sketch: bytes, screenshot: bytes, expected_elements: Optional[List[str]] = None) -> Optional[float]:
    """Returns coverage 0-1, or None if the judge is unavailable (no key / API error)."""
    if config.MOCK_MODE:
        return 0.5
    if not os.getenv("GROQ_API_KEY"):
        return None
    try:
        from groq import Groq

        extra = f"The sketch should contain: {', '.join(expected_elements)}." if expected_elements else ""
        r = Groq(api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(
            model=config.GROQ_JUDGE_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT.format(extra=extra)}, _img(sketch), _img(screenshot)]}],
            temperature=0, max_tokens=800)
        txt = r.choices[0].message.content
        data = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
        return max(0.0, min(1.0, float(data["coverage"])))
    except Exception as e:  # API error or reply that isn't valid JSON
        log.warning("judge failed: %s", e)
        return None
