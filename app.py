import os
import sys
import traceback
import json

import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Trip Planner Backend",
    version="debug-2026-08"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    # With allow_origins=["*"], credentials should normally be
    # False unless you specifically need cookies/auth.
    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ENVIRONMENT
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

SERPER_API_KEY = os.getenv(
    "SERPER_API_KEY",
    ""
).strip()

ORS_API_KEY = os.getenv(
    "ORS_API_KEY",
    ""
).strip()

OPENROUTER_URL = "https://openrouter.ai/api/v1"


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
# DEBUG CONNECTION
# ============================================================

@app.get("/debug")
async def debug():

    return {
        "status": "FastAPI backend connection verified!"
    }


# ============================================================
# DEBUG ENVIRONMENT
# ============================================================

@app.get("/debug-env")
async def debug_env():

    def key_info(value):

        if not value:
            return {
                "exists": False,
                "length": 0,
                "prefix": ""
            }

        return {
            "exists": True,
            "length": len(value),
            "prefix": value[:15]
        }

    return {
        "render": os.getenv("RENDER", "false"),

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
# DEBUG OPENROUTER
# ============================================================

@app.get("/debug-openrouter")
async def debug_openrouter():

    """
    Direct HTTP request to OpenRouter.

    This bypasses LiteLLM and CrewAI.

    If this returns 200, the OpenRouter key itself works.
    """

    if not OPENROUTER_API_KEY:

        return {
            "success": False,
            "error": "OPENROUTER_API_KEY is missing"
        }

    try:

        response = requests.get(
            f"{OPENROUTER_URL}/models",

            headers={
                "Authorization":
                    f"Bearer {OPENROUTER_API_KEY}"
            },

            timeout=30
        )

        return {
            "success": response.status_code == 200,

            "status": response.status_code,

            "response": response.text[:10000]
        }

    except Exception as e:

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }


# ============================================================
# DEBUG LITELLM
# ============================================================

@app.get("/debug-litellm")
async def debug_litellm():

    """
    Tests LiteLLM directly.

    This is intentionally kept as a diagnostic endpoint.
    """

    try:

        import litellm

        print("=" * 70)
        print("DEBUG LITELLM")
        print("=" * 70)

        print(
            f"OPENROUTER_API_KEY exists: "
            f"{bool(OPENROUTER_API_KEY)}"
        )

        if OPENROUTER_API_KEY:

            print(
                f"OPENROUTER_API_KEY length: "
                f"{len(OPENROUTER_API_KEY)}"
            )

            print(
                f"OPENROUTER_API_KEY prefix: "
                f"{OPENROUTER_API_KEY[:15]}..."
            )

        print(
            f"LiteLLM version: "
            f"{getattr(litellm, '__version__', 'unknown')}"
        )

        # Explicitly set it for this process.
        #
        # This is important because we don't want to depend
        # on LiteLLM's automatic environment-variable discovery.
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

        result = litellm.completion(

            model="openrouter/z-ai/glm-5.3-flash",

            messages=[
                {
                    "role": "user",
                    "content": "Reply only with: LITELLM OK"
                }
            ],

            api_key=OPENROUTER_API_KEY,

            api_base=OPENROUTER_URL,

            temperature=0
        )

        content = result.choices[0].message.content

        return {

            "test": "LiteLLM -> OpenRouter",

            "success": True,

            "result": content,

            "versions": {
                "litellm": getattr(
                    litellm,
                    "__version__",
                    "unknown"
                )
            }
        }

    except Exception as e:

        print("=" * 70)
        print("LITELLM TEST FAILED")
        print("=" * 70)

        print(f"Type: {type(e).__name__}")
        print(f"Error: {e}")

        traceback.print_exc()

        print("=" * 70)

        try:
            import litellm
            litellm_version = getattr(
                litellm,
                "__version__",
                "unknown"
            )
        except Exception:
            litellm_version = "unknown"

        try:
            import crewai
            crewai_version = getattr(
                crewai,
                "__version__",
                "unknown"
            )
        except Exception:
            crewai_version = "unknown"

        try:
            import openai
            openai_version = getattr(
                openai,
                "__version__",
                "unknown"
            )
        except Exception:
            openai_version = "unknown"

        return {

            "test": "LiteLLM -> OpenRouter",

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e),

            "versions": {
                "litellm": litellm_version,
                "crewai": crewai_version,
                "openai": openai_version
            }
        }


# ============================================================
# DEBUG CREWAI
# ============================================================

@app.get("/debug-crewai")
async def debug_crewai():

    """
    Tests the EXACT LLM object configured in crew.py.

    This is different from /debug-litellm.
    """

    try:

        from crew import test_llm

        result = await test_llm()

        try:
            import crewai
            crewai_version = getattr(
                crewai,
                "__version__",
                "unknown"
            )
        except Exception:
            crewai_version = "unknown"

        try:
            import litellm
            litellm_version = getattr(
                litellm,
                "__version__",
                "unknown"
            )
        except Exception:
            litellm_version = "unknown"

        try:
            import openai
            openai_version = getattr(
                openai,
                "__version__",
                "unknown"
            )
        except Exception:
            openai_version = "unknown"

        return {

            "test": "Exact CrewAI LLM -> OpenRouter",

            "result": result,

            "versions": {
                "crewai": crewai_version,
                "litellm": litellm_version,
                "openai": openai_version
            }
        }

    except Exception as e:

        print("=" * 70)
        print("CREWAI DEBUG ENDPOINT FAILED")
        print("=" * 70)

        print(f"Type: {type(e).__name__}")
        print(f"Error: {e}")

        traceback.print_exc()

        print("=" * 70)

        return {

            "test": "Exact CrewAI LLM -> OpenRouter",

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

    print("=" * 70)
    print("POST /generate-trip")
    print("=" * 70)

    print(
        f"starting_city: {request.starting_city}"
    )

    print(
        f"days: {request.days}"
    )

    print(
        f"travel_style: {request.travel_style}"
    )

    print(
        f"transport: {request.transport}"
    )

    print(
        f"budget: {request.budget}"
    )

    print("=" * 70)

    try:

        from crew import generate_trip

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

        print("=" * 70)
        print("GENERATE TRIP ERROR")
        print("=" * 70)

        print(f"Type: {type(e).__name__}")
        print(f"Error: {e}")

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )


# ============================================================
# STARTUP INFORMATION
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("")
    print("=" * 70)
    print("AI TRIP PLANNER STARTED")
    print("=" * 70)

    print(
        f"Python: {sys.version}"
    )

    print(
        f"OPENROUTER_API_KEY exists: "
        f"{bool(OPENROUTER_API_KEY)}"
    )

    if OPENROUTER_API_KEY:

        print(
            f"OPENROUTER_API_KEY length: "
            f"{len(OPENROUTER_API_KEY)}"
        )

        print(
            f"OPENROUTER_API_KEY prefix: "
            f"{OPENROUTER_API_KEY[:15]}..."
        )

    print(
        f"SERPER_API_KEY exists: "
        f"{bool(SERPER_API_KEY)}"
    )

    print(
        f"ORS_API_KEY exists: "
        f"{bool(ORS_API_KEY)}"
    )

    print("=" * 70)
    print("")
