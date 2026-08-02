from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crew import generate_trip

app = FastAPI(title="AI Trip Planner Backend")

# Enable global CORS to allow cross-origin requests from any frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


# Health check root endpoint
@app.get("/")
async def root():
    return {"status": "FastAPI backend on Render is running!"}


# Debug endpoint for frontend validation
@app.get("/debug")
@app.get("/debug/")
async def debug():
    return {"status": "FastAPI backend connection verified!"}


class TripRequest(BaseModel):
    starting_city: str
    days: int
    travel_style: str
    transport: str
    budget: str


# Generates trip itinerary via CrewAI
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
        raise HTTPException(status_code=500, detail=str(e))
