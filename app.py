import os
import traceback
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crew import (
    generate_trip,
    debug_crewai_llm,
    CREWAI_VERSION,
    CREWAI_TOOLS_VERSION,
    LITELLM_VERSION,
    OPENAI_VERSION,
)


# ============================================================
# FASTAPI
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

    print(
        "RENDER:",
        os.getenv("RENDER", "NOT SET")
    )

    print("")
    print("PACKAGE VERSIONS:")

    print(
        "  crewai:",
        CREWAI_VERSION
    )

    print(
        "  crewai-tools:",
        CREWAI_TOOLS_VERSION
    )

    print(
        "  litellm:",
        LITELLM_VERSION
    )

    print(
        "  openai:",
        OPENAI_VERSION
    )

    openrouter_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    serper_key = os.getenv(
        "SERPER_API_KEY"
    )

    ors_key = os.getenv(
        "ORS_API_KEY"
    )

    print("")
    print("ENVIRONMENT VARIABLES:")

    print("")
    print("OPENROUTER_API_KEY:")

    print(
        "  Exists:",
        bool(openrouter_key)
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

    print("")
    print("SERPER_API_KEY:")

    print(
        "  Exists:",
        bool(serper_key)
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

    print("")
    print("ORS_API_KEY:")

    print(
        "  Exists:",
        bool(ors_key)
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

    print("")
    print("=" * 70)
    print("STARTUP COMPLETE")
    print("=" * 70)
    print("")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "FastAPI backend on Render is running!"
    }


# ============================================================
# BASIC DEBUG
# ============================================================

@app.get("/debug")
@app.get("/debug/")
async def debug():

    return {
        "status": "FastAPI backend connection verified!"
    }


# ============================================================
# DEBUG ENVIRONMENT
# ============================================================

@app.get("/debug-env")
@app.get("/debug-env/")
async def debug_env():

    openrouter_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    serper_key = os.getenv(
        "SERPER_API_KEY"
    )

    ors_key = os.getenv(
        "ORS_API_KEY"
    )

    return {

        "render": os.getenv(
            "RENDER",
            "NOT SET"
        ),

        "openrouter": {

            "exists": bool(openrouter_key),

            "length": (
                len(openrouter_key)
                if openrouter_key
                else 0
            ),

            "prefix": (
                openrouter_key[:15]
                if openrouter_key
                else None
            )
        },

        "serper": {

            "exists": bool(serper_key),

            "length": (
                len(serper_key)
                if serper_key
                else 0
            ),

            "prefix": (
                serper_key[:10]
                if serper_key
                else None
            )
        },

        "ors": {

            "exists": bool(ors_key),

            "length": (
                len(ors_key)
                if ors_key
                else 0
            ),

            "prefix": (
                ors_key[:10]
                if ors_key
                else None
            )
        },

        "versions": {

            "crewai": CREWAI_VERSION,

            "crewai_tools": CREWAI_TOOLS_VERSION,

            "litellm": LITELLM_VERSION,

            "openai": OPENAI_VERSION
        }
    }


# ============================================================
# DIRECT OPENROUTER TEST
# ============================================================

@app.get("/debug-openrouter")
@app.get("/debug-openrouter/")
async def debug_openrouter():

    print("")
    print("=" * 70)
    print("DIRECT OPENROUTER DEBUG TEST")
    print("=" * 70)

    key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not key:

        print(
            "OPENROUTER_API_KEY is NOT configured."
        )

        return {

            "key_exists": False,

            "message": (
                "OPENROUTER_API_KEY is not configured"
            )
        }

    print(
        "Key exists:",
        True
    )

    print(
        "Key prefix:",
        key[:15]
    )

    print(
        "Key length:",
        len(key)
    )

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
        print(
            "END DIRECT OPENROUTER DEBUG TEST"
        )
        print("=" * 70)

        return {

            "key_exists": True,

            "key_length": len(key),

            "key_prefix": key[:15],

            "openrouter_status": (
                response.status_code
            ),

            "openrouter_response": (
                response.text[:1000]
            )
        }

    except Exception as e:

        print("")
        print(
            "OPENROUTER CONNECTION ERROR"
        )

        print(
            "Type:",
            type(e).__name__
        )

        print(
            "Message:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 70)

        return {

            "key_exists": True,

            "key_length": len(key),

            "key_prefix": key[:15],

            "error_type": (
                type(e).__name__
            ),

            "error": str(e)
        }


# ============================================================
# DIRECT CREWAI TEST
# ============================================================
#
# THIS IS THE IMPORTANT TEST.
#
# It tests:
#
# Render
#   ↓
# OPENROUTER_API_KEY
#   ↓
# CrewAI LLM
#   ↓
# LiteLLM
#   ↓
# OpenRouter
#
# without using the agents, tasks, Serper or Crew.
#
# TEMPORARY DEBUG ENDPOINT.
# Remove after debugging.
# ============================================================

@app.get("/debug-crewai")
@app.get("/debug-crewai/")
async def debug_crewai():

    print("")
    print("=" * 70)
    print("CREWAI LLM DEBUG ENDPOINT")
    print("=" * 70)

    key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    print(
        "OPENROUTER_API_KEY exists:",
        bool(key)
    )

    if key:

        print(
            "OpenRouter key prefix:",
            key[:15]
        )

        print(
            "OpenRouter key length:",
            len(key)
        )

    print("")
    print("CrewAI version:", CREWAI_VERSION)
    print(
        "CrewAI Tools version:",
        CREWAI_TOOLS_VERSION
    )
    print(
        "LiteLLM version:",
        LITELLM_VERSION
    )
    print(
        "OpenAI version:",
        OPENAI_VERSION
    )

    try:

        result = await debug_crewai_llm()

        print("")
        print("=" * 70)
        print("CREWAI DEBUG SUCCESS")
        print("=" * 70)

        return {

            "success": True,

            "message": (
                "CrewAI successfully called OpenRouter"
            ),

            "result": result,

            "versions": {

                "crewai": CREWAI_VERSION,

                "crewai_tools": CREWAI_TOOLS_VERSION,

                "litellm": LITELLM_VERSION,

                "openai": OPENAI_VERSION
            }
        }

    except Exception as e:

        print("")
        print("=" * 70)
        print("CREWAI DEBUG FAILED")
        print("=" * 70)

        print(
            "Exception type:",
            type(e).__name__
        )

        print(
            "Exception:",
            repr(e)
        )

        print(
            "Message:",
            str(e)
        )

        print("")
        print("FULL TRACEBACK:")

        traceback.print_exc()

        print("=" * 70)

        return {

            "success": False,

            "error_type": (
                type(e).__name__
            ),

            "error": str(e),

            "versions": {

                "crewai": CREWAI_VERSION,

                "crewai_tools": CREWAI_TOOLS_VERSION,

                "litellm": LITELLM_VERSION,

                "openai": OPENAI_VERSION
            }
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

    print(
        "Starting city:",
        request.starting_city
    )

    print(
        "Days:",
        request.days
    )

    print(
        "Travel style:",
        request.travel_style
    )

    print(
        "Transport:",
        request.transport
    )

    print(
        "Budget:",
        request.budget
    )

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

        print(
            "Exception type:",
            type(e).__name__
        )

        print("")
        print(
            "Exception representation:",
            repr(e)
        )

        print("")
        print(
            "Exception message:",
            str(e)
        )

        print("")
        print("FULL TRACEBACK:")

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )
