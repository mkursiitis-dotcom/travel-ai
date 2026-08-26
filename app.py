import os
import traceback
import json

import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# APP
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
# ENVIRONMENT
# ============================================================

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    ""
).strip()

SERPER_API_KEY = os.environ.get(
    "SERPER_API_KEY",
    ""
).strip()

ORS_API_KEY = os.environ.get(
    "ORS_API_KEY",
    ""
).strip()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "FastAPI backend on Render is running!",
        "service": "AI Trip Planner"
    }


# ============================================================
# DEBUG
# ============================================================

@app.get("/debug")
@app.get("/debug/")
async def debug():

    return {

        "status": "FastAPI backend connection verified!",

        "environment": "Render" if os.environ.get("RENDER") else "unknown",

        "openrouter_configured": bool(
            OPENROUTER_API_KEY
        ),

        "serper_configured": bool(
            SERPER_API_KEY
        ),

        "ors_configured": bool(
            ORS_API_KEY
        )
    }


# ============================================================
# DEBUG ENVIRONMENT
# ============================================================

@app.get("/debug-env")
async def debug_env():

    def key_info(key):

        if not key:

            return {
                "exists": False,
                "length": 0,
                "prefix": ""
            }

        return {

            "exists": True,

            "length": len(key),

            "prefix": key[:15]
        }


    return {

        "render": os.environ.get(
            "RENDER",
            "false"
        ),

        "openrouter": key_info(
            OPENROUTER_API_KEY
        ),

        "serper": key_info(
            SERPER_API_KEY
        ),

        "ors": key_info(
            ORS_API_KEY
        )
    }


# ============================================================
# DEBUG OPENROUTER DIRECT
# ============================================================

@app.get("/debug-openrouter")
async def debug_openrouter():

    if not OPENROUTER_API_KEY:

        return {

            "success": False,

            "error": (
                "OPENROUTER_API_KEY is missing"
            )
        }


    try:

        response = requests.get(

            "https://openrouter.ai/api/v1/models",

            headers={

                "Authorization":
                    f"Bearer {OPENROUTER_API_KEY}",

                "Content-Type":
                    "application/json"
            },

            timeout=30
        )


        return {

            "success": response.ok,

            "status": response.status_code,

            "response": response.text[:5000]
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
# DEBUG LITELLM
# ============================================================

@app.get("/debug-litellm")
async def debug_litellm():

    try:

        from litellm import completion

        print("=" * 70)
        print("DEBUG LITELLM")
        print("=" * 70)

        print(
            "OPENROUTER key exists:",
            bool(OPENROUTER_API_KEY)
        )

        print(
            "OPENROUTER key length:",
            len(OPENROUTER_API_KEY)
        )

        response = completion(

            model="openrouter/openai/gpt-4o-mini",

            messages=[

                {
                    "role": "user",

                    "content":
                        "Reply with exactly: LiteLLM test successful."
                }
            ],

            api_key=OPENROUTER_API_KEY,

            api_base="https://openrouter.ai/api/v1",

            temperature=0,

            max_tokens=50
        )


        print("LITELLM SUCCESS")
        print(str(response)[:2000])

        return {

            "test":
                "LiteLLM -> OpenRouter",

            "result": {

                "success": True,

                "response":
                    str(response)[:5000]
            },

            "versions":
                get_versions()
        }


    except Exception as e:

        print("=" * 70)
        print("LITELLM TEST FAILED")
        print("=" * 70)

        print(type(e).__name__)
        print(str(e))

        traceback.print_exc()

        return {

            "test":
                "LiteLLM -> OpenRouter",

            "result": {

                "success": False,

                "error_type":
                    type(e).__name__,

                "error":
                    str(e)
            },

            "versions":
                get_versions()
        }


# ============================================================
# DEBUG CREWAI
# ============================================================

@app.get("/debug-crewai")
async def debug_crewai():

    try:

        print("=" * 70)
        print("DEBUG CREWAI")
        print("=" * 70)

        from crew import test_crewai_llm

        result = await test_crewai_llm()

        return {

            "test":
                "CrewAI -> LiteLLM -> OpenRouter",

            "result":
                result,

            "versions":
                get_versions()
        }


    except Exception as e:

        print("=" * 70)
        print("CREWAI DEBUG FAILED")
        print("=" * 70)

        print(type(e).__name__)
        print(str(e))

        traceback.print_exc()

        return {

            "test":
                "CrewAI -> LiteLLM -> OpenRouter",

            "result": {

                "success": False,

                "error_type":
                    type(e).__name__,

                "error":
                    str(e)
            },

            "versions":
                get_versions()
        }


# ============================================================
# VERSION INFORMATION
# ============================================================

def get_versions():

    versions = {}


    try:

        import crewai

        versions["crewai"] = getattr(
            crewai,
            "__version__",
            "unknown"
        )

    except Exception:

        versions["crewai"] = "unknown"


    try:

        import crewai_tools

        versions["crewai_tools"] = getattr(
            crewai_tools,
            "__version__",
            "unknown"
        )

    except Exception:

        versions["crewai_tools"] = "unknown"


    try:

        import litellm

        versions["litellm"] = getattr(
            litellm,
            "__version__",
            "unknown"
        )

    except Exception:

        versions["litellm"] = "unknown"


    try:

        import openai

        versions["openai"] = getattr(
            openai,
            "__version__",
            "unknown"
        )

    except Exception:

        versions["openai"] = "unknown"


    return versions


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

    print("=" * 70)
    print("POST /generate-trip")
    print("=" * 70)

    print(
        "Request:",
        request.model_dump()
    )


    try:

        # Import here instead of at application startup.
        #
        # This makes the API startup more robust and makes
        # debugging much easier.

        from crew import generate_trip


        result = await generate_trip(

            request.starting_city,

            request.days,

            request.travel_style,

            request.transport,

            request.budget
        )


        print("=" * 70)
        print("GENERATE TRIP SUCCESS")
        print("=" * 70)


        return {

            "trip": result
        }


    except Exception as e:

        print("=" * 70)
        print("❌ GENERATE TRIP ERROR")
        print("=" * 70)

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 70)


        raise HTTPException(

            status_code=500,

            detail=str(e)
        )
