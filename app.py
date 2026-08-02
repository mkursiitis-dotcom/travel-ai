from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crew import generate_trip


app = FastAPI(title="AI Trip Planner Backend")


# ==========================
# CORS
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# HEALTH CHECK
# ==========================

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


# ==========================
# REQUEST MODEL
# ==========================

class TripRequest(BaseModel):
    starting_city: str
    days: int
    travel_style: str
    transport: str
    budget: str


# ==========================
# GENERATE TRIP
# ==========================

@app.post("/generate-trip")
@app.post("/generate-trip/")
async def generate(request: TripRequest):

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
        print(f"CREWAI ERROR: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
