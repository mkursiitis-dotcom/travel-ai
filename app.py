import os
import traceback

import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crew import generate_trip


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Trip Planner Backend"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("")
    print("=" * 70)
    print("AI TRIP PLANNER BACKEND STARTED")
    print("=" * 70)

    print("RENDER:", os.getenv("RENDER", "NOT SET"))

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    ors_key = os.getenv("ORS_API_KEY")

    print("")
    print("Environment variables:")

    print(
        "OPENROUTER_API_KEY:",
        "SET" if openrouter_key else "NOT SET"
    )

    if openrouter_key:
        print(
            "  Prefix:",
            openrouter_key[:15]
        )
        print(
            "  Length:",
            len(openrouter_key)
        )

    print(
        "SERPER_API_KEY:",
        "SET" if serper_key else "NOT SET"
    )

    if serper_key:
        print(
            "  Prefix:",
            serper_key[:10]
        )
        print(
            "  Length:",
            len(serper_key)
        )

    print(
        "ORS_API_KEY:",
        "SET" if ors_key else "NOT SET"
    )

    if ors_key:
        print(
            "  Prefix:",
            ors_key[:10]
        )
        print(
            "  Length:",
            len(ors_key)
        )

    print("=" * 70)
    print("")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "FastAPI backend on Render is running!"
    }


@app.get("/debug")
@app.get("/debug/")
async def debug():

    return {
        "status": "FastAPI backend connection verified!"
    }


# ============================================================
# ENVIRONMENT DEBUG
# ============================================================
#
# TEMPORARY DEBUG ENDPOINT.
#
# This endpoint NEVER returns the complete API keys.
#
# Remove this endpoint after debugging.
# ============================================================

@app.get("/debug-env")
@app.get("/debug-env/")
async def debug_env():

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")
    ors_key = os.getenv("ORS_API_KEY")

    return {
        "render": os.getenv("RENDER", "NOT SET"),

        "openrouter": {
            "exists": bool(openrouter_key),
            "length": len(openrouter_key) if openrouter_key else 0,
            "prefix": (
                openrouter_key[:15]
                if openrouter_key
                else None
            )
        },

        "serper": {
            "exists": bool(serper_key),
            "length": len(serper_key) if serper_key else 0,
            "prefix": (
                serper_key[:10]
                if serper_key
                else None
            )
        },

        "ors": {
            "exists": bool(ors_key),
            "length": len(ors_key) if ors_key else 0,
            "prefix": (
                ors_key[:10]
                if ors_key
                else None
            )
        }
    }


# ============================================================
# OPENROUTER DEBUG
# ============================================================
#
# This tests the exact OPENROUTER_API_KEY loaded by Render.
#
# TEMPORARY DEBUG ENDPOINT.
#
# Remove this endpoint after debugging.
# ============================================================

@app.get("/debug-openrouter")
@app.get("/debug-openrouter/")
async def debug_openrouter():

    print("")
    print("=" * 70)
    print("DIRECT OPENROUTER DEBUG TEST")
    print("=" * 70)

    key = os.getenv("OPENROUTER_API_KEY")

    if not key:

        print("OPENROUTER_API_KEY is NOT configured.")

        return {
            "key_exists": False,
            "message": "OPENROUTER_API_KEY is not configured"
        }

    print("Key exists: True")
    print("Key prefix:", key[:15])
    print("Key length:", len(key))

    try:

        response = requests.get(
            "https://openrouter.ai/api/v1/models",

            headers={
                "Authorization": f"Bearer {key}"
            },

            timeout=30
        )

        print(
            "OpenRouter HTTP status:",
            response.status_code
        )

        print(
            "OpenRouter response:",
            response.text[:1000]
        )

        print("=" * 70)
        print("END DIRECT OPENROUTER DEBUG TEST")
        print("=" * 70)

        return {
            "key_exists": True,
            "key_length": len(key),
            "key_prefix": key[:15],

            "openrouter_status": response.status_code,

            "openrouter_response": response.text[:1000]
        }

    except Exception as e:

        print("")
        print("OPENROUTER CONNECTION ERROR")
        print("Type:", type(e).__name__)
        print("Message:", str(e))
        traceback.print_exc()

        print("=" * 70)

        return {
            "key_exists": True,
            "key_length": len(key),
            "key_prefix": key[:15],

            "error_type": type(e).__name__,
            "error": str(e)
        }


# ============================================================
# REQUEST MODEL
# ============================================================

class TripRequest(BaseModel):

    starting_city: str

    days: int

    travel_style: str

    transport: str

    budget: str


# ============================================================
# GENERATE TRIP
# ============================================================

@app.post("/generate-trip")
@app.post("/generate-trip/")
async def generate(request: TripRequest):

    print("")
    print("=" * 70)
    print("POST /generate-trip")
    print("=" * 70)

    print("Request:")
    print("  starting_city:", request.starting_city)
    print("  days:", request.days)
    print("  travel_style:", request.travel_style)
    print("  transport:", request.transport)
    print("  budget:", request.budget)

    print("=" * 70)

    try:

        result = await generate_trip(

            request.starting_city,

            request.days,

            request.travel_style,

            request.transport,

            request.budget
        )

        print("")
        print("=" * 70)
        print("TRIP GENERATION SUCCESSFUL")
        print("=" * 70)

        return {
            "trip": result
        }

    except Exception as e:

        print("")
        print("=" * 70)
        print("GENERATE-TRIP ERROR")
        print("=" * 70)

        print("Exception type:")
        print(type(e).__name__)

        print("")
        print("Exception:")
        print(repr(e))

        print("")
        print("Message:")
        print(str(e))

        print("")
        print("FULL TRACEBACK:")
        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )
