import os
import traceback
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crew import (
    generate_trip,
    debug_litellm,
    debug_crewai_llm,
    CREWAI_VERSION,
    CREWAI_TOOLS_VERSION,
    LITELLM_VERSION,
    OPENAI_VERSION,
)


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
    print("AI TRIP PLANNER STARTED")
    print("=" * 70)

    print(
        "OPENROUTER_API_KEY exists:",
        bool(os.getenv("OPENROUTER_API_KEY"))
    )

    key = os.getenv("OPENROUTER_API_KEY")

    if key:

        print(
            "OPENROUTER key prefix:",
            key[:15]
        )

        print(
            "OPENROUTER key length:",
            len(key)
        )

    print("")
    print("VERSIONS:")
    print("CrewAI:", CREWAI_VERSION)
    print("CrewAI Tools:", CREWAI_TOOLS_VERSION)
    print("LiteLLM:", LITELLM_VERSION)
    print("OpenAI:", OPENAI_VERSION)

    print("=" * 70)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "FastAPI backend on Render is running!"
    }


# ============================================================
# DEBUG ENV
# ============================================================

@app.get("/debug-env")
@app.get("/debug-env/")
async def debug_env():

    key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    return {

        "render": os.getenv(
            "RENDER",
            "NOT SET"
        ),

        "openrouter": {

            "exists": bool(key),

            "length": (
                len(key)
                if key
                else 0
            ),

            "prefix": (
                key[:15]
                if key
                else None
            )
        },

        "openai_base": os.getenv(
            "OPENAI_API_BASE"
        ),

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

    key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not key:

        return {
            "success": False,
            "error": "OPENROUTER_API_KEY missing"
        }

    try:

        response = requests.get(

            "https://openrouter.ai/api/v1/models",

            headers={
                "Authorization": f"Bearer {key}"
            },

            timeout=30
        )

        return {

            "success": (
                response.status_code == 200
            ),

            "status": response.status_code,

            "response": response.text[:1000]
        }

    except Exception as e:

        return {

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e)
        }


# ============================================================
# DIRECT LITELLM TEST
# ============================================================

@app.get("/debug-litellm")
@app.get("/debug-litellm/")
async def debug_litellm_endpoint():

    print("")
    print("=" * 70)
    print("DEBUG LITELLM ENDPOINT")
    print("=" * 70)

    try:

        result = await debug_litellm()

        return {

            "test": "LiteLLM -> OpenRouter",

            "result": result,

            "versions": {

                "crewai": CREWAI_VERSION,

                "crewai_tools": CREWAI_TOOLS_VERSION,

                "litellm": LITELLM_VERSION,

                "openai": OPENAI_VERSION
            }
        }

    except Exception as e:

        traceback.print_exc()

        return {

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e)
        }


# ============================================================
# CREWAI LLM TEST
# ============================================================

@app.get("/debug-crewai")
@app.get("/debug-crewai/")
async def debug_crewai():

    print("")
    print("=" * 70)
    print("DEBUG CREWAI ENDPOINT")
    print("=" * 70)

    try:

        result = await debug_crewai_llm()

        return {

            "test": "CrewAI -> LiteLLM -> OpenRouter",

            "result": result,

            "versions": {

                "crewai": CREWAI_VERSION,

                "crewai_tools": CREWAI_TOOLS_VERSION,

                "litellm": LITELLM_VERSION,

                "openai": OPENAI_VERSION
            }
        }

    except Exception as e:

        traceback.print_exc()

        return {

            "success": False,

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

    print(
        "starting_city:",
        request.starting_city
    )

    print(
        "days:",
        request.days
    )

    print(
        "travel_style:",
        request.travel_style
    )

    print(
        "transport:",
        request.transport
    )

    print(
        "budget:",
        request.budget
    )

    try:

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

    except Exception as e:

        print("")
        print("=" * 70)
        print("GENERATE TRIP FAILED")
        print("=" * 70)

        print(
            "Exception:",
            type(e).__name__
        )

        print(
            "Message:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )
