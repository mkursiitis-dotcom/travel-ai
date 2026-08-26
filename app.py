# ============================================================
# app.py
# ============================================================

import os
import sys
import traceback
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv()


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
# BASIC ROUTES
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
# DEBUG ENVIRONMENT
# ============================================================

@app.get("/debug-env")
async def debug_env():

    openrouter = os.getenv(
        "OPENROUTER_API_KEY"
    )

    serper = os.getenv(
        "SERPER_API_KEY"
    )

    ors = os.getenv(
        "ORS_API_KEY"
    )

    return {

        "render": os.getenv(
            "RENDER",
            "false"
        ),

        "openrouter": {
            "exists": bool(openrouter),

            "length": (
                len(openrouter)
                if openrouter
                else 0
            ),

            "prefix": (
                openrouter[:15]
                if openrouter
                else None
            ),
        },

        "serper": {
            "exists": bool(serper),

            "length": (
                len(serper)
                if serper
                else 0
            ),

            "prefix": (
                serper[:10]
                if serper
                else None
            ),
        },

        "ors": {
            "exists": bool(ors),

            "length": (
                len(ors)
                if ors
                else 0
            ),

            "prefix": (
                ors[:10]
                if ors
                else None
            ),
        },
    }


# ============================================================
# DEBUG OPENROUTER DIRECT
# ============================================================

@app.get("/debug-openrouter")
async def debug_openrouter():

    key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not key:

        return {
            "success": False,
            "error": "OPENROUTER_API_KEY is missing"
        }

    try:

        response = requests.get(

            "https://openrouter.ai/api/v1/models",

            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },

            timeout=20,
        )

        return {

            "success": (
                response.status_code == 200
            ),

            "status": response.status_code,

            "response": response.text[:3000],
        }

    except Exception as e:

        return {

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e),
        }


# ============================================================
# DEBUG LITELLM
# ============================================================

@app.get("/debug-litellm")
async def debug_litellm():

    key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not key:

        return {
            "test": "LiteLLM -> OpenRouter",

            "result": {
                "success": False,

                "error": (
                    "OPENROUTER_API_KEY is missing"
                ),
            },
        }

    try:

        # Import only here so we can clearly see whether
        # LiteLLM itself is responsible for the failure.

        import litellm

        print("=" * 60)
        print("DEBUG LITELLM")
        print("=" * 60)

        print(
            "LiteLLM version:",
            getattr(
                litellm,
                "__version__",
                "unknown"
            )
        )

        print(
            "OpenRouter key prefix:",
            key[:15]
        )

        print(
            "OpenRouter key length:",
            len(key)
        )

        print("=" * 60)

        result = litellm.completion(

            model="openrouter/openai/gpt-4o-mini",

            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly: "
                        "LITELLM TEST OK"
                    ),
                }
            ],

            api_key=key,

            api_base=(
                "https://openrouter.ai/api/v1"
            ),

            max_tokens=20,
        )

        return {

            "test": "LiteLLM -> OpenRouter",

            "result": {
                "success": True,

                "response": str(
                    result
                )[:3000],
            },

            "versions": {
                "litellm": getattr(
                    litellm,
                    "__version__",
                    "unknown"
                ),
            },
        }

    except Exception as e:

        print("=" * 60)
        print("LITELLM TEST FAILED")
        print("=" * 60)

        print(
            "TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 60)

        versions = {}

        try:
            import litellm

            versions["litellm"] = getattr(
                litellm,
                "__version__",
                "unknown"
            )

        except Exception:
            pass

        try:
            import crewai

            versions["crewai"] = getattr(
                crewai,
                "__version__",
                "unknown"
            )

        except Exception:
            pass

        try:
            import openai

            versions["openai"] = getattr(
                openai,
                "__version__",
                "unknown"
            )

        except Exception:
            pass

        return {

            "test": "LiteLLM -> OpenRouter",

            "result": {

                "success": False,

                "error_type": type(e).__name__,

                "error": str(e),
            },

            "versions": versions,
        }


# ============================================================
# DEBUG CREWAI
# ============================================================

@app.get("/debug-crewai")
async def debug_crewai():

    try:

        print("=" * 60)
        print("DEBUG CREWAI")
        print("=" * 60)

        # Import our CrewAI objects.

        from crew import llm

        print(
            "CrewAI LLM object created successfully."
        )

        print(
            "LLM object:",
            repr(llm)
        )

        print(
            "Calling LLM directly..."
        )

        # CrewAI's LLM object can be called directly.
        #
        # This is a much smaller test than running the
        # entire travel-planning Crew.

        result = llm.call(
            "Reply with exactly: CREWAI TEST OK"
        )

        print(
            "CrewAI LLM call succeeded."
        )

        return {

            "test": (
                "CrewAI -> LiteLLM -> OpenRouter"
            ),

            "result": {

                "success": True,

                "response": str(
                    result
                )[:3000],
            },

        }

    except Exception as e:

        print("=" * 60)
        print("CREWAI TEST FAILED")
        print("=" * 60)

        print(
            "TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 60)

        versions = {}

        try:
            import crewai

            versions["crewai"] = getattr(
                crewai,
                "__version__",
                "unknown"
            )

        except Exception:
            pass

        try:
            import litellm

            versions["litellm"] = getattr(
                litellm,
                "__version__",
                "unknown"
            )

        except Exception:
            pass

        try:
            import openai

            versions["openai"] = getattr(
                openai,
                "__version__",
                "unknown"
            )

        except Exception:
            pass

        return {

            "test": (
                "CrewAI -> LiteLLM -> OpenRouter"
            ),

            "result": {

                "success": False,

                "error_type": type(e).__name__,

                "error": str(e),
            },

            "versions": versions,
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
async def generate(
    request: TripRequest
):

    print("=" * 60)
    print("POST /generate-trip")
    print("=" * 60)

    print(
        "Request:",
        request.model_dump()
    )

    try:

        # Import only when endpoint is actually called.
        #
        # This makes startup more reliable and allows
        # /debug-openrouter to work even if CrewAI has
        # an initialization problem.

        from crew import generate_trip

        result = await generate_trip(

            request.starting_city,

            request.days,

            request.travel_style,

            request.transport,

            request.budget,
        )

        print(
            "Trip generated successfully."
        )

        return {
            "trip": result
        }

    except Exception as e:

        print("=" * 60)
        print("GENERATE TRIP FAILED")
        print("=" * 60)

        print(
            "TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 60)

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )
