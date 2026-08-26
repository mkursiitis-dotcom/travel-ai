# ============================================================
# crew.py
# ============================================================

import os
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------
# Load environment variables FIRST
# ------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

if not SERPER_API_KEY:
    print("WARNING: SERPER_API_KEY is not set")


# ============================================================
# IMPORTANT:
# Configure LiteLLM/OpenRouter explicitly BEFORE importing
# CrewAI components.
# ============================================================

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# LiteLLM sometimes looks for OPENAI_API_KEY depending on
# the provider/model configuration.
#
# We deliberately DO NOT put the OpenRouter key into
# OPENAI_API_KEY because that can cause provider confusion.
#
# Instead we use the explicit OpenRouter API base below.

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ============================================================
# Imports
# ============================================================

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("=" * 60)
print("CREW.PY INITIALIZATION")
print("=" * 60)

print("OPENROUTER_API_KEY exists:",
      bool(os.getenv("OPENROUTER_API_KEY")))

if OPENROUTER_API_KEY:
    print(
        "OPENROUTER_API_KEY prefix:",
        OPENROUTER_API_KEY[:15]
    )
    print(
        "OPENROUTER_API_KEY length:",
        len(OPENROUTER_API_KEY)
    )

print("SERPER_API_KEY exists:",
      bool(os.getenv("SERPER_API_KEY")))

print("OPENROUTER_BASE_URL:",
      OPENROUTER_BASE_URL)

print("=" * 60)


# ============================================================
# DIRECT OPENROUTER TEST
# ============================================================

def test_openrouter_direct():

    print("=" * 60)
    print("DIRECT OPENROUTER TEST")
    print("=" * 60)

    try:

        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=20,
        )

        print("Status:", response.status_code)

        print(
            "Response preview:",
            response.text[:500]
        )

        if response.status_code == 200:

            print("DIRECT OPENROUTER TEST: SUCCESS")

            return {
                "success": True,
                "status": response.status_code,
                "response": response.text[:1000],
            }

        print("DIRECT OPENROUTER TEST: FAILED")

        return {
            "success": False,
            "status": response.status_code,
            "response": response.text[:1000],
        }

    except Exception as e:

        print(
            "DIRECT OPENROUTER TEST EXCEPTION:",
            repr(e)
        )

        return {
            "success": False,
            "error": repr(e),
        }


# ============================================================
# LLM
# ============================================================

# IMPORTANT:
#
# We explicitly provide the API key and API base to CrewAI's LLM
# rather than relying only on environment autodetection.
#
# OpenRouter model:
#
# openrouter/openai/gpt-4o-mini
#
# If this particular model causes problems, we can switch it
# later to another model available through OpenRouter.

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.3,
)


# ============================================================
# SERPER TOOL
# ============================================================

search_tool = SerperDevTool()


# ============================================================
# AGENT 1
# ============================================================

planner = Agent(
    role="Latvijas ceļojumu plānotājs",

    goal=(
        "Izveidot optimālu un reālistisku "
        "Latvijas ceļojuma koncepciju."
    ),

    backstory="""
Tu esi pieredzējis Latvijas tūrisma plānotājs.

Tu labi pārzini Latvijas pilsētas, reģionus,
dabas objektus, pilis un apskates vietas.

Tu veido loģiskus maršrutus, lai ceļotājs
nepavadītu pārāk daudz laika transportā.
""",

    llm=llm,

    verbose=True,
)


# ============================================================
# AGENT 2
# ============================================================

guide_logistics = Agent(
    role="Latvijas tūrisma un loģistikas eksperts",

    goal=(
        "Atrast piemērotākās apskates vietas, "
        "ēdināšanas vietas un optimālu maršrutu."
    ),

    backstory="""
Tu esi Latvijas ceļojumu eksperts.

Tu pārzini apskates vietas, restorānus,
naktsmītnes un ceļu loģistiku.

Tu veido praktiskus plānus ar reālām izmaksām.
""",

    tools=[search_tool],

    llm=llm,

    verbose=True,
)


# ============================================================
# AGENT 3
# ============================================================

reviewer = Agent(
    role="Ceļojumu plāna redaktors",

    goal=(
        "Izveidot profesionālu gala ceļojuma "
        "plānu Markdown formātā."
    ),

    backstory="""
Tu esi rūpīgs tūrisma satura redaktors.

Tu pārbaudi, lai dienu skaits būtu pareizs,
un izveido skaidras Markdown tabulas.
""",

    llm=llm,

    verbose=True,
)


# ============================================================
# TASK 1
# ============================================================

task1 = Task(
    description="""
Izveido {days} dienu Latvijas ceļojuma konceptu.

Parametri:

Sākuma pilsēta:
{starting_city}

Ceļojuma veids:
{travel_style}

Transports:
{transport}

Budžets:
{budget}

Prasības:

- Katrai dienai norādi galveno reģionu.
- Izvēlies loģisku maršrutu.
- Ņem vērā transportu un budžetu.
""",

    expected_output=(
        "Ceļojuma koncepta plāns pa dienām."
    ),

    agent=planner,
)


# ============================================================
# TASK 2
# ============================================================

task2 = Task(
    description="""
Izveido detalizētu {days} dienu ceļojuma plānu.

Informācija:

Sākuma pilsēta:
{starting_city}

Ceļojuma veids:
{travel_style}

Transports:
{transport}

Budžets:
{budget}

Iekļauj:

- apskates objektus
- dabas takas
- ēdināšanu
- naktsmītnes
- aptuvenās izmaksas
""",

    expected_output=(
        "Detalizēts ceļojuma plāns ar izmaksām."
    ),

    agent=guide_logistics,
)


# ============================================================
# TASK 3
# ============================================================

task3 = Task(
    description="""
Izveido gala ceļojuma plānu Markdown formātā.

OBLIGĀTI:

Katrai dienai izveido Markdown tabulu:

| Laiks | Atrašanās vieta | Objekts / Darbība | Apraksts |
|---|---|---|---|

Pēc katras dienas pievieno:

### Ēdināšana un naktsmītnes

- Pusdienas:
- Vakariņas:
- Naktsmītne:

Beigās pievieno:

## Aptuvenās izmaksas

- Transports
- Ēdiens
- Naktsmītnes
- Ieejas maksas
- Kopā

Un praktiskus ceļošanas padomus.
""",

    expected_output=(
        "Pilns Markdown ceļojuma plāns."
    ),

    agent=reviewer,
)


# ============================================================
# CREW
# ============================================================

crew = Crew(
    agents=[
        planner,
        guide_logistics,
        reviewer,
    ],

    tasks=[
        task1,
        task2,
        task3,
    ],

    process=Process.sequential,

    verbose=True,
)


# ============================================================
# GENERATE TRIP
# ============================================================

async def generate_trip(
    starting_city: str,
    days: int,
    travel_style: str,
    transport: str,
    budget: str,
):

    print("=" * 60)
    print("GENERATE_TRIP STARTED")
    print("=" * 60)

    inputs = {
        "starting_city": starting_city,
        "days": days,
        "travel_style": travel_style,
        "transport": transport,
        "budget": budget,
    }

    print("Inputs:", inputs)

    try:

        print("Calling CrewAI kickoff_async...")

        result = await crew.kickoff_async(
            inputs=inputs
        )

        print("CrewAI completed successfully.")

        return result.raw

    except Exception as e:

        print("=" * 60)
        print("CREWAI ERROR")
        print("=" * 60)

        print("Exception type:")
        print(type(e).__name__)

        print("Exception:")
        print(str(e))

        print("repr:")
        print(repr(e))

        print("=" * 60)

        raise
