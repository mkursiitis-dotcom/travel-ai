import os
import asyncio
import traceback

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# CONFIG
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

SERPER_API_KEY = os.getenv(
    "SERPER_API_KEY",
    ""
).strip()

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)

MODEL_NAME = (
    "openrouter/z-ai/glm-5.3-flash"
)


# ============================================================
# ENVIRONMENT
# ============================================================

if not OPENROUTER_API_KEY:
    print(
        "WARNING: OPENROUTER_API_KEY is missing"
    )

else:
    print(
        f"OpenRouter key loaded: "
        f"{OPENROUTER_API_KEY[:15]}..."
    )


# ============================================================
# LLM
# ============================================================

llm = LLM(

    model=MODEL_NAME,

    api_key=OPENROUTER_API_KEY,

    base_url=OPENROUTER_BASE_URL,

    temperature=0.3,

    max_tokens=8000
)


# ============================================================
# SERPER
# ============================================================

search_tool = None

if SERPER_API_KEY:

    try:

        search_tool = SerperDevTool()

        print(
            "SerperDevTool initialized."
        )

    except Exception as e:

        print(
            f"WARNING: Serper failed: {e}"
        )


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

Tu labi pārzini Latvijas pilsētas,
reģionus, dabas objektus, pilis
un apskates vietas.

Tu veido loģiskus maršrutus,
lai ceļotājs nepavadītu
nevajadzīgi daudz laika transportā.
""",

    llm=llm,

    verbose=False
)


# ============================================================
# AGENT 2
# ============================================================

guide_tools = []

if search_tool:
    guide_tools.append(search_tool)


guide_logistics = Agent(

    role="Latvijas tūrisma un loģistikas eksperts",

    goal=(
        "Atrast piemērotākās apskates vietas, "
        "ēdināšanas vietas un optimālu maršrutu."
    ),

    backstory="""
Tu esi Latvijas ceļojumu eksperts.

Tu pārzini apskates vietas,
restorānus, naktsmītnes
un ceļu loģistiku.

Tu veido praktiskus plānus
ar aptuvenām izmaksām.

Ja ir pieejama meklēšana,
izmanto to aktuālai informācijai.
""",

    tools=guide_tools,

    llm=llm,

    verbose=False
)


# ============================================================
# AGENT 3
# ============================================================

reviewer = Agent(

    role="Ceļojumu plāna redaktors",

    goal=(
        "Izveidot profesionālu gala "
        "ceļojuma plānu Markdown formātā."
    ),

    backstory="""
Tu esi rūpīgs tūrisma satura redaktors.

Tu pārbaudi, lai dienu skaits
būtu pareizs, maršruts būtu loģisks
un gala rezultāts būtu viegli lasāms.

Tu izveido skaidras Markdown tabulas.
""",

    llm=llm,

    verbose=False
)


# ============================================================
# TASK 1
# ============================================================

task1 = Task(

    description="""
Izveido {days} dienu Latvijas ceļojuma konceptu.

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
- Ņem vērā transportu.
- Ņem vērā budžetu.
- Samazini nevajadzīgu braukšanu.
""",

    expected_output=(
        "Ceļojuma koncepta plāns pa dienām."
    ),

    agent=planner
)


# ============================================================
# TASK 2
# ============================================================

task2 = Task(

    description="""
Izveido detalizētu {days} dienu
ceļojuma plānu.

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
- transporta loģiku

Izmanto reālas Latvijas vietas.
""",

    expected_output=(
        "Detalizēts ceļojuma plāns ar izmaksām."
    ),

    agent=guide_logistics
)


# ============================================================
# TASK 3
# ============================================================

task3 = Task(

    description="""
    Izveido gala ceļojuma plānu, izmantojot iepriekšējo
    plānotāja un tūrisma eksperta darbu.

    Apvieno un pārbaudi iepriekš iegūto informāciju.

OBLIGĀTI:

Katrai dienai:

| Laiks | Atrašanās vieta | Objekts / Darbība | Apraksts |
|---|---|---|---|

Pēc katras dienas:

### Ēdināšana un naktsmītnes

- Pusdienas:
- Vakariņas:
- Naktsmītne:

Beigās:

## Aptuvenās izmaksas

- Transports
- Ēdiens
- Naktsmītnes
- Ieejas maksas
- Kopā

Pievieno praktiskus ceļošanas padomus.

Atbildi tikai ar gala ceļojuma plānu.
""",

    expected_output=(
        "Pilns Markdown ceļojuma plāns."
    ),

    agent=reviewer,

    context=[task1, task2]
)


# ============================================================
# CREW
# ============================================================

crew = Crew(

    agents=[
        planner,
        guide_logistics,
        reviewer
    ],

    tasks=[
        task1,
        task2,
        task3
    ],

    process=Process.sequential,

    verbose=False
)


# ============================================================
# TEST EXACT CREWAI LLM
# ============================================================

async def test_llm():

    try:

        result = await asyncio.to_thread(
            llm.call,
            "Atbildi tikai ar: TEST OK"
        )

        return {
            "success": True,
            "result": str(result)
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }


# ============================================================
# GENERATE TRIP
# ============================================================

async def generate_trip(
    starting_city: str,
    days: int,
    travel_style: str,
    transport: str,
    budget: str
):

    inputs = {

        "starting_city":
            starting_city,

        "days":
            days,

        "travel_style":
            travel_style,

        "transport":
            transport,

        "budget":
            budget
    }

    try:

        result = await crew.kickoff_async(
            inputs=inputs
        )

        return result.raw

    except Exception as e:

        print(
            "CREWAI GENERATION FAILED"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        raise
