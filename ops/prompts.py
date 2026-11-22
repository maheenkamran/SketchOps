"""get_prompt(version) -> {"generate": str, "fix": str}
Loads from Langfuse (prompt names sketch_to_html_generate / sketch_to_html_fix, label = version).
Falls back to the local copies below, so everything works offline."""
import logging
import os

log = logging.getLogger("prompts")

_RULES = """Rules:
- Output ONE complete self-contained HTML file inside a single ```html code block, nothing else.
- Use Tailwind via exactly this tag in <head>: <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.5"></script>
- Desktop webpage (designed for about 1280px wide), responsive so it still works on a phone.
  Include <meta name="viewport" content="width=device-width,initial-scale=1">.
- No external images or fonts; use emoji, inline SVG or coloured divs as placeholders.
- Small inline <script> for interactivity is fine (e.g. a menu toggle). No network calls."""

_SKETCH_LEGEND = """How to read the sketch:
- Squiggly lines are text. Thick or dark squiggles are headings; thin or grey ones are paragraphs or links.
  Write real, sensible text that fits the page's topic (guess the topic from the layout). No lorem ipsum.
- A box with an X through it is an image: use a coloured gradient block with a fitting emoji or inline SVG icon.
- Small boxes with a word inside are buttons or inputs. Filled or coloured areas keep a similar colour.
- Do NOT draw the sketch's outlines, squiggles or X lines; replace them with real content and styling
  (colours, spacing, shadows, rounded corners, a consistent font and colour scheme).
"""

LOCAL = {
    "v1": {
        "generate": "You are a front-end engineer. Turn the hand-drawn sketch into a working webpage. "
                    "Reproduce every element and the layout as drawn, with readable labels.\n" + _RULES,
        "fix": "You are a front-end engineer. The webpage below was built from the sketch but scored {score} "
               "for similarity (target >= 0.75). Compare the sketch with the screenshot, find missing, extra or "
               "misplaced elements, and return the full corrected HTML.\n" + _RULES,
    },
    # v2: a finished, realistic website instead of a literal copy of the drawing (v1 pages look like wireframes)
    "v2": {
        "generate": "You are a senior web designer. The image is a rough hand-drawn wireframe of a website. "
                    "Build the finished, realistic website it describes, as a real company would publish it.\n"
                    + _SKETCH_LEGEND +
                    "First silently list every element and its position, top to bottom, left to right. "
                    "Then build exactly that layout: same sections, same order, same columns and proportions. "
                    "Do not add sections that are not in the sketch and do not drop any.\n" + _RULES,
        "fix": "You are a senior web designer. The website in the screenshot was built from the hand-drawn "
               "wireframe and scored {score} for how well it matches (target >= 0.75).\n" + _SKETCH_LEGEND +
               "Compare them section by section. Fix only what is missing, extra or in the wrong place, and keep "
               "everything that already matches exactly as it is. It must still look like a finished, realistic "
               "website, not a wireframe. Return the full corrected HTML.\n" + _RULES,
    },
}


def get_prompt(version: str) -> dict:
    if version not in LOCAL:  # fail loudly: a typo must not silently run v1 and get logged as another version
        raise KeyError(f"unknown prompt version {version!r}; known: {list(LOCAL)}")
    out = dict(LOCAL[version])
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import Langfuse

            lf = Langfuse(host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"))
            for kind in ("generate", "fix"):
                out[kind] = lf.get_prompt(f"sketch_to_html_{kind}", label=version).prompt
        except Exception as e:
            log.warning("Langfuse prompt fetch failed (%s) -> using local '%s'", e, version)
    return out


def push_local_prompts():
    """One-off: python -c 'from ops.prompts import push_local_prompts as p; p()'"""
    from langfuse import Langfuse

    lf = Langfuse(host=os.getenv("LANGFUSE_HOST", "http://localhost:3001"))
    for version, ps in LOCAL.items():
        for kind, text in ps.items():
            lf.create_prompt(name=f"sketch_to_html_{kind}", prompt=text, labels=[version])
    lf.flush()
    print("pushed", list(LOCAL))
