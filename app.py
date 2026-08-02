from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from crew import generate_trip

app = FastAPI()

# Enable CORS so SiteGround (visitme.lv) can make API calls to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://visitme.lv",
        "https://www.visitme.lv",
        "http://visitme.lv",
        "http://www.visitme.lv"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
async def root():
    return {"status": "FastAPI backend on Render is running!"}


@app.get("/debug")
async def debug():
    return {"status": "FastAPI backend connection verified!"}


class TripRequest(BaseModel):
    starting_city: str
    days: int
    travel_style: str
    transport: str
    budget: str


@app.post("/generate-trip")
@app.post("/generate-trip/")
async def generate(request: TripRequest):
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
