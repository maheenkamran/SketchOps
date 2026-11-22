"""Small image helpers shared by the generator (agent) and the judge (scoring)."""
import io

MAX_IMAGE_SIDE = 1600  # px; bigger photos are shrunk before sending (API size limits, fewer tokens)


def mime(b: bytes) -> str:
    """PNG files start with fixed 'magic' bytes; everything else is treated as JPEG."""
    return "image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


def shrink(b: bytes) -> bytes:
    """Downscale a large photo/screenshot so a request stays under the APIs' size limits."""
    from PIL import Image

    img = Image.open(io.BytesIO(b))
    if max(img.size) <= MAX_IMAGE_SIDE and len(b) <= 3_000_000:
        return b
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()
