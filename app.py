import os
import traceback
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crew import (
    generate_trip,
    debug_direct_openrouter,
    debug_crewai_llm,
    CREWAI_VERSION,
    CREWAI_TOOLS_VERSION,
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

    key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    print(
        "OPENROUTER_API_KEY exists:",
        bool(key)
    )

    if key:

        print(
            "Key prefix:",
            key[:15]
        )

        print(
            "Key length:",
            len(key)
        )

    print("")
    print("Versions:")

    print(
        "CrewAI:",
        CREWAI_VERSION
    )

    print(
        "CrewAI Tools:",
        CREWAI_TOOLS_VERSION
    )

    print(
        "OpenAI:",
        OPENAI_VERSION
    )

    print("")
    print("Architecture:")
    print(
        "CrewAI -> Custom DirectOpenRouterLLM -> OpenAI client -> OpenRouter"
    )

    print(
        "LiteLLM: NOT USED BY CUSTOM LLM"
    )

    print("=" * 70)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {

        "status":
        "FastAPI backend on Render is running!"
    }


# ============================================================
# BASIC DEBUG
# ============================================================

@app.get("/debug")
@app.get("/debug/")
async def debug():

    return {

        "status":
        "FastAPI backend connection verified!"
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

        "render":
        os.getenv(
            "RENDER",
            "NOT SET"
        ),

        "openrouter": {

            "exists":
            bool(key),

            "length":
            len(key)
            if key
            else 0,

            "prefix":
            key[:15]
            if key
            else None
        },

        "versions": {

            "crewai":
            CREWAI_VERSION,

            "crewai_tools":
            CREWAI_TOOLS_VERSION,

            "openai":
            OPENAI_VERSION
        },

        "architecture":
        "CrewAI -> DirectOpenRouterLLM -> OpenAI -> OpenRouter"
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

            "error":
            "OPENROUTER_API_KEY missing"
        }


    try:

        response = requests.get(

            "https://openrouter.ai/api/v1/models",

            headers={

                "Authorization":
                f"Bearer {key}"
            },

            timeout=30
        )


        return {

            "success":
            response.status_code == 200,

            "status":
            response.status_code,

            "response":
            response.text[:1000]
        }


    except Exception as e:

        return {

            "success": False,

            "error_type":
            type(e).__name__,

            "error":
            str(e)
        }


# ============================================================
# DIRECT OPENROUTER CHAT TEST
# ============================================================
#
# This is different from /debug-openrouter.
#
# /debug-openrouter only asks OpenRouter for /models.
#
# This endpoint actually sends an AI prompt.
#
# It uses the OpenAI client directly.
# ============================================================

@app.get("/debug-direct-openrouter")
@app.get("/debug-direct-openrouter/")
async def debug_direct_openrouter_endpoint():

    print("")
    print("=" * 70)
    print("DEBUG DIRECT OPENROUTER CHAT")
    print("=" * 70)

    try:

        result = await debug_direct_openrouter()

        return {

            "test":
            "OpenAI client -> OpenRouter",

            "result":
            result,

            "versions": {

                "crewai":
                CREWAI_VERSION,

                "crewai_tools":
                CREWAI_TOOLS_VERSION,

                "openai":
                OPENAI_VERSION
            }
        }


    except Exception as e:

        traceback.print_exc()

        return {

            "success": False,

            "error_type":
            type(e).__name__,

            "error":
            str(e)
        }


# ============================================================
# CREWAI DIRECT LLM TEST
# ============================================================

@app.get("/debug-crewai")
@app.get("/debug-crewai/")
async def debug_crewai():

    print("")
    print("=" * 70)
    print("DEBUG CREWAI DIRECT LLM")
    print("=" * 70)

    try:

        result = await debug_crewai_llm()

        return {

            "test":
            "CrewAI -> DirectOpenRouterLLM -> OpenRouter",

            "result":
            result,

            "uses_litellm":
            False,

            "versions": {

                "crewai":
                CREWAI_VERSION,

                "crewai_tools":
                CREWAI_TOOLS_VERSION,

                "openai":
                OPENAI_VERSION
            }
        }


    except Exception as e:

        traceback.print_exc()

        return {

            "success": False,

            "error_type":
            type(e).__name__,

            "error":
            str(e),

            "uses_litellm":
            False
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

            "trip":
            result
        }


    except Exception as e:

        print("")
        print("=" * 70)
        print("TRIP GENERATION FAILED")
        print("=" * 70)

        print(
            "Exception type:",
            type(e).__name__
        )

        print(
            "Exception:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 70)


        raise HTTPException(

            status_code=500,

            detail=str(e)
        )
