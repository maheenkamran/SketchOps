"""Tiny HTTP server that runs inside the sandbox container."""
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from render import render_local

app = FastAPI(title="SketchOps sandbox")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/render")
async def do_render(request: Request):
    html = (await request.body()).decode("utf-8", "ignore")
    if len(html) > 500_000:
        raise HTTPException(413, "html too large")
    try:
        png = await run_in_threadpool(render_local, html, 15000)
    except Exception as e:
        raise HTTPException(500, f"render failed: {e}")
    return Response(png, media_type="image/png")
