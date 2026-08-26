import os
import traceback
import requests

from dotenv import load_dotenv

load_dotenv()

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
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "FastAPI backend on Render is running!"
    }


# ============================================================
# DEBUG
# ============================================================

@app.get("/debug")
@app.get("/debug/")
async def debug():

    return {
        "status": "FastAPI backend connection verified!"
    }


# ============================================================
# DEBUG ENV
# ============================================================

@app.get("/debug-env")
@app.get("/debug-env/")
async def debug_env():

    openrouter_key = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    ).strip()

    serper_key = os.getenv(
        "SERPER_API_KEY",
        ""
    ).strip()

    ors_key = os.getenv(
        "OPENROUTESERVICE_API_KEY",
        ""
    ).strip()


    return {

        "render": os.getenv(
            "RENDER",
            "false"
        ),

        "openrouter": {

            "exists": bool(openrouter_key),

            "length": len(openrouter_key),

            "prefix": (
                openrouter_key[:15]
                if openrouter_key
                else ""
            ),
        },

        "serper": {

            "exists": bool(serper_key),

            "length": len(serper_key),

            "prefix": (
                serper_key[:10]
                if serper_key
                else ""
            ),
        },

        "ors": {

            "exists": bool(ors_key),

            "length": len(ors_key),

            "prefix": (
                ors_key[:10]
                if ors_key
                else ""
            ),
        },
    }


# ============================================================
# DIRECT OPENROUTER TEST
# ============================================================

@app.get("/debug-openrouter")
@app.get("/debug-openrouter/")
async def debug_openrouter():

    key = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    ).strip()


    if not key:

        return {

            "success": False,

            "error": (
                "OPENROUTER_API_KEY is missing"
            ),
        }


    try:

        response = requests.get(

            "https://openrouter.ai/api/v1/models",

            headers={

                "Authorization":
                    f"Bearer {key}",

                "Content-Type":
                    "application/json",
            },

            timeout=20,
        )


        return {

            "success":
                response.status_code == 200,

            "status":
                response.status_code,

            "key_length":
                len(key),

            "key_prefix":
                key[:15],

            "response":
                response.text[:3000],
        }


    except Exception as e:

        traceback.print_exc()

        return {

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e),
        }


# ============================================================
# LITELLM TEST
# ============================================================

@app.get("/debug-litellm")
@app.get("/debug-litellm/")
async def debug_litellm():

    key = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    ).strip()


    if not key:

        return {

            "test":
                "LiteLLM -> OpenRouter",

            "result": {

                "success": False,

                "error":
                    "OPENROUTER_API_KEY is missing",
            },
        }


    try:

        os.environ[
            "OPENROUTER_API_KEY"
        ] = key


        from litellm import completion


        result = completion(

            model=
                "openrouter/z-ai/glm-5.3-flash",

            messages=[

                {

                    "role": "user",

                    "content":
                        "Reply with exactly OK",
                }

            ],

            api_key=key,

            api_base=
                "https://openrouter.ai/api/v1",

            temperature=0,

            max_tokens=10,
        )


        content = (
            result.choices[0]
            .message.content
        )


        return {

            "test":
                "LiteLLM -> OpenRouter",

            "result": {

                "success": True,

                "response": content,
            },

            "versions":
                get_versions(),
        }


    except Exception as e:

        traceback.print_exc()


        return {

            "test":
                "LiteLLM -> OpenRouter",

            "result": {

                "success": False,

                "error_type":
                    type(e).__name__,

                "error":
                    str(e),
            },

            "versions":
                get_versions(),
        }


# ============================================================
# CREWAI TEST
# ============================================================

@app.get("/debug-crewai")
@app.get("/debug-crewai/")
async def debug_crewai():

    try:

        from crew import test_crewai

        result = await test_crewai()


        return {

            "test":
                "CrewAI -> OpenRouter",

            "result":
                result,

            "versions":
                get_versions(),
        }


    except Exception as e:

        traceback.print_exc()


        return {

            "test":
                "CrewAI -> OpenRouter",

            "result": {

                "success": False,

                "error_type":
                    type(e).__name__,

                "error":
                    str(e),
            },

            "versions":
                get_versions(),
        }


# ============================================================
# VERSION INFORMATION
# ============================================================

def get_version(package_name):

    try:

        from importlib.metadata import version

        return version(package_name)

    except Exception:

        return "unknown"


def get_versions():

    return {

        "crewai":
            get_version("crewai"),

        "crewai_tools":
            get_version("crewai-tools"),

        "litellm":
            get_version("litellm"),

        "openai":
            get_version("openai"),
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


    try:

        from crew import generate_trip


        result = await generate_trip(

            request.starting_city,

            request.days,

            request.travel_style,

            request.transport,

            request.budget,
        )


        return {

            "trip": result
        }


    except Exception as e:

        print("=" * 70)
        print("GENERATE TRIP ERROR")
        print("=" * 70)

        print(
            "ERROR TYPE:",
            type(e).__name__
        )

        print(
            "ERROR:",
            str(e)
        )

        traceback.print_exc()


        raise HTTPException(

            status_code=500,

            detail=str(e)
        )
