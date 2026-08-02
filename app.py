import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from crew import generate_trip

app = FastAPI(title="Debug AI Trip Planner")

BASE_DIR = Path(__file__).resolve().parent

# --- DEBUG ENDPOINTS ---

@app.get("/debug")
@app.get("/debug/")
async def debug_info(request: Request):
    """
    Open https://visitme.lv/debug in browser.
    If you see JSON output, FastAPI is working!
    """
    return {
        "status": "FastAPI server is RUNNING",
        "client_host": request.client.host if request.client else "unknown",
        "request_url": str(request.url),
        "headers": dict(request.headers),
        "python_version": sys.version,
        "available_routes": [route.path for route in app.routes]
    }

# Dynamic catch-all logger to diagnose missed requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"--> [FASTAPI REQUEST] {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"<-- [FASTAPI RESPONSE] Status Code: {response.status_code}")
    return response


# --- FILE SERVING ---

@app.get("/")
async def home():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/style.css")
async def get_css():
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/script.js")
async def get_js():
    js_path = BASE_DIR / "script.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")


# --- TRIP GENERATION ---

class TripRequest(BaseModel):
    starting_city: str
    days: int
    travel_style: str
    transport: str
    budget: str


@app.post("/generate-trip")
@app.post("/generate-trip/")
async def generate(request: TripRequest):
    result = await generate_trip(
        request.starting_city,
        request.days,
        request.travel_style,
        request.transport,
        request.budget
    )

    return {
        "trip": result
    }