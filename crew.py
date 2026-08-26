import os
import asyncio
import traceback

from dotenv import load_dotenv

# Load .env locally.
# Render environment variables are automatically available.
load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

SERPER_API_KEY = os.getenv(
    "SERPER_API_KEY",
    ""
).strip()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME = "openrouter/z-ai/glm-5.3-flash"


# ============================================================
# STARTUP DEBUG
# ============================================================

print("=" * 70)
print("CREW.PY STARTUP")
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
else:
    print(
        "ERROR: OPENROUTER_API_KEY IS EMPTY"
    )

print(
    f"OPENROUTER_BASE_URL: "
    f"{OPENROUTER_BASE_URL}"
)

print(
    f"MODEL_NAME: "
    f"{MODEL_NAME}"
)

print("=" * 70)


# ============================================================
# VALIDATION
# ============================================================

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. "
        "Set it in Render Environment Variables."
    )


# ============================================================
# LLM
# ============================================================

# IMPORTANT:
#
# We use OpenRouter directly.
#
# No OpenAI API key is required.
#
# CrewAI -> LiteLLM -> OpenRouter
#
# The OpenRouter API key is explicitly supplied here.
# ============================================================

llm = LLM(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.3,
)


# ============================================================
# TOOLS
# ============================================================

search_tool = None

if SERPER_API_KEY:

    try:

        search_tool = SerperDevTool()

        print(
            "SerperDevTool initialized successfully."
        )

    except Exception as e:

        print(
            "WARNING: Could not initialize "
            "SerperDevTool."
        )

        print(
            f"Error: {e}"
        )

else:

    print(
        "WARNING: SERPER_API_KEY is missing."
    )


# ============================================================
# AGENTS
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

    verbose=True
)


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

Tu pārzini apskates vietas, restorānus,
naktsmītnes un ceļu loģistiku.

Tu veido praktiskus plānus ar reālām izmaksām.

Ja tev ir pieejama meklēšanas funkcija,
izmanto to, lai pārbaudītu aktuālu informāciju.
""",

    tools=guide_tools,

    llm=llm,

    verbose=True
)


reviewer = Agent(
    role="Ceļojumu plāna redaktors",

    goal=(
        "Izveidot profesionālu gala ceļojuma "
        "plānu Markdown formātā."
    ),

    backstory="""
Tu esi rūpīgs tūrisma satura redaktors.

Tu pārbaudi, lai dienu skaits būtu pareizs,
maršruts būtu loģisks un gala rezultāts
būtu viegli lasāms.

Tu izveido skaidras Markdown tabulas.
""",

    llm=llm,

    verbose=True
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
- Ņem vērā transportu.
- Ņem vērā budžetu.
- Centies samazināt nevajadzīgu braukšanu.
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
- transporta loģiku

Ja izmanto meklēšanas rīku, prioritizē reālus
un aktuālus Latvijas objektus un vietas.
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

Atbildi tikai ar gala ceļojuma plānu.
""",

    expected_output=(
        "Pilns Markdown ceļojuma plāns."
    ),

    agent=reviewer
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

    verbose=True
)


# ============================================================
# DIRECT CREWAI LLM TEST
# ============================================================

async def test_llm():

    print("=" * 70)
    print("DIRECT CREWAI LLM TEST")
    print("=" * 70)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Base URL: {OPENROUTER_BASE_URL}"
    )

    print(
        f"API key exists: "
        f"{bool(OPENROUTER_API_KEY)}"
    )

    if OPENROUTER_API_KEY:

        print(
            f"API key length: "
            f"{len(OPENROUTER_API_KEY)}"
        )

        print(
            f"API key prefix: "
            f"{OPENROUTER_API_KEY[:15]}..."
        )

    try:

        result = await asyncio.to_thread(
            llm.call,
            "Atbildi tikai ar: TEST OK"
        )

        print(
            "DIRECT LLM TEST SUCCESS"
        )

        print(
            f"Result: {result}"
        )

        return {
            "success": True,
            "result": str(result)
        }

    except Exception as e:

        print(
            "DIRECT LLM TEST FAILED"
        )

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

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

    print("=" * 70)
    print("GENERATE TRIP")
    print("=" * 70)

    print(
        f"starting_city = {starting_city}"
    )

    print(
        f"days = {days}"
    )

    print(
        f"travel_style = {travel_style}"
    )

    print(
        f"transport = {transport}"
    )

    print(
        f"budget = {budget}"
    )

    print("=" * 70)

    inputs = {
        "starting_city": starting_city,
        "days": days,
        "travel_style": travel_style,
        "transport": transport,
        "budget": budget
    }

    try:

        print(
            "Starting CrewAI kickoff_async..."
        )

        result = await crew.kickoff_async(
            inputs=inputs
        )

        print(
            "CrewAI completed successfully."
        )

        return result.raw

    except Exception as e:

        print("=" * 70)
        print("CREWAI GENERATION FAILED")
        print("=" * 70)

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

        traceback.print_exc()

        print("=" * 70)

        raise
