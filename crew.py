import os
import importlib.metadata
import traceback

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# ENVIRONMENT
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
ORS_API_KEY = os.getenv("ORS_API_KEY")


# ============================================================
# VERSION INFORMATION
# ============================================================

def get_package_version(package_name: str):
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "UNKNOWN"


CREWAI_VERSION = get_package_version("crewai")
CREWAI_TOOLS_VERSION = get_package_version("crewai-tools")
LITELLM_VERSION = get_package_version("litellm")
OPENAI_VERSION = get_package_version("openai")


# ============================================================
# SAFE ENVIRONMENT DEBUG
# ============================================================

def print_environment_debug():

    print("")
    print("=" * 70)
    print("ENVIRONMENT DEBUG")
    print("=" * 70)

    print("RENDER:", os.getenv("RENDER", "NOT SET"))

    print("")
    print("OPENROUTER_API_KEY:")
    print("  Exists:", bool(OPENROUTER_API_KEY))
    print(
        "  Length:",
        len(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else 0
    )
    print(
        "  Prefix:",
        OPENROUTER_API_KEY[:15]
        if OPENROUTER_API_KEY
        else "NONE"
    )

    print("")
    print("SERPER_API_KEY:")
    print("  Exists:", bool(SERPER_API_KEY))
    print(
        "  Length:",
        len(SERPER_API_KEY) if SERPER_API_KEY else 0
    )
    print(
        "  Prefix:",
        SERPER_API_KEY[:10]
        if SERPER_API_KEY
        else "NONE"
    )

    print("")
    print("ORS_API_KEY:")
    print("  Exists:", bool(ORS_API_KEY))
    print(
        "  Length:",
        len(ORS_API_KEY) if ORS_API_KEY else 0
    )
    print(
        "  Prefix:",
        ORS_API_KEY[:10]
        if ORS_API_KEY
        else "NONE"
    )

    print("")
    print("PACKAGE VERSIONS:")
    print("  crewai:", CREWAI_VERSION)
    print("  crewai-tools:", CREWAI_TOOLS_VERSION)
    print("  litellm:", LITELLM_VERSION)
    print("  openai:", OPENAI_VERSION)

    print("=" * 70)
    print("END ENVIRONMENT DEBUG")
    print("=" * 70)
    print("")


print_environment_debug()


# ============================================================
# OPENROUTER CONFIGURATION
# ============================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openrouter/openai/gpt-4o-mini"


# ============================================================
# CREATE LLM
# ============================================================

def create_openrouter_llm():

    print("")
    print("=" * 70)
    print("CREATING OPENROUTER LLM")
    print("=" * 70)

    print("Model:", OPENROUTER_MODEL)
    print("Base URL:", OPENROUTER_BASE_URL)

    print(
        "API key exists:",
        bool(OPENROUTER_API_KEY)
    )

    print(
        "API key length:",
        len(OPENROUTER_API_KEY)
        if OPENROUTER_API_KEY
        else 0
    )

    print(
        "API key prefix:",
        OPENROUTER_API_KEY[:15]
        if OPENROUTER_API_KEY
        else "NONE"
    )

    if not OPENROUTER_API_KEY:

        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    llm = LLM(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.3,
    )

    print("LLM object created successfully.")

    print("=" * 70)

    return llm


# ============================================================
# MAIN LLM
# ============================================================

llm = create_openrouter_llm()


# ============================================================
# TOOLS
# ============================================================

print("")
print("Initializing SerperDevTool...")

search_tool = SerperDevTool()

print("SerperDevTool initialized.")


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
un izveido skaidras Markdown tabulas.
""",

    llm=llm,
    verbose=True
)


# ============================================================
# TASKS
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

    agent=planner
)


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

    agent=guide_logistics
)


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

    agent=reviewer
)


# ============================================================
# CREW
# ============================================================

print("")
print("=" * 70)
print("INITIALIZING CREW")
print("=" * 70)

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

print("Crew initialized successfully.")

print("=" * 70)


# ============================================================
# DEBUG CREWAI LLM
# ============================================================

async def debug_crewai_llm():

    print("")
    print("=" * 70)
    print("DEBUG CREWAI LLM")
    print("=" * 70)

    print("Testing:")
    print("CrewAI -> LLM -> OpenRouter")

    print("")
    print("Model:", OPENROUTER_MODEL)
    print("Base URL:", OPENROUTER_BASE_URL)

    print(
        "API key exists:",
        bool(OPENROUTER_API_KEY)
    )

    print(
        "API key length:",
        len(OPENROUTER_API_KEY)
        if OPENROUTER_API_KEY
        else 0
    )

    print(
        "API key prefix:",
        OPENROUTER_API_KEY[:15]
        if OPENROUTER_API_KEY
        else "NONE"
    )

    if not OPENROUTER_API_KEY:

        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    try:

        test_llm = create_openrouter_llm()

        print("")
        print("Calling CrewAI LLM...")
        print("Prompt: Reply with exactly: OK")

        # CrewAI's LLM object uses LiteLLM underneath.
        # Calling call() gives us a direct test of the LLM
        # without running the agents/tasks.

        result = test_llm.call(
            "Reply with exactly: OK"
        )

        print("")
        print("CREWAI LLM TEST SUCCESSFUL")
        print("Result:", result)

        print("=" * 70)

        return {
            "success": True,
            "result": str(result)
        }

    except Exception as e:

        print("")
        print("=" * 70)
        print("CREWAI LLM TEST FAILED")
        print("=" * 70)

        print("Exception type:")
        print(type(e).__name__)

        print("")
        print("Exception representation:")
        print(repr(e))

        print("")
        print("Exception message:")
        print(str(e))

        print("")
        print("FULL TRACEBACK:")

        traceback.print_exc()

        print("=" * 70)

        raise


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

    print("")
    print("=" * 70)
    print("GENERATE TRIP STARTED")
    print("=" * 70)

    print("Starting city:", starting_city)
    print("Days:", days)
    print("Travel style:", travel_style)
    print("Transport:", transport)
    print("Budget:", budget)

    print("")
    print(
        "OpenRouter key configured:",
        bool(OPENROUTER_API_KEY)
    )

    if OPENROUTER_API_KEY:

        print(
            "OpenRouter key prefix:",
            OPENROUTER_API_KEY[:15]
        )

        print(
            "OpenRouter key length:",
            len(OPENROUTER_API_KEY)
        )

    else:

        print("OpenRouter key: NONE")

    inputs = {
        "starting_city": starting_city,
        "days": days,
        "travel_style": travel_style,
        "transport": transport,
        "budget": budget
    }

    print("")
    print("CrewAI inputs:")
    print(inputs)

    print("")
    print("Starting crew.kickoff_async()...")
    print("=" * 70)

    try:

        result = await crew.kickoff_async(
            inputs=inputs
        )

        print("")
        print("=" * 70)
        print("CREWAI COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(
            "Result type:",
            type(result).__name__
        )

        if hasattr(result, "raw"):

            print(
                "Raw result length:",
                len(result.raw)
            )

        print("=" * 70)

        return result.raw

    except Exception as e:

        print("")
        print("=" * 70)
        print("CREWAI ERROR")
        print("=" * 70)

        print(
            "Exception type:",
            type(e).__name__
        )

        print("")
        print(
            "Exception representation:",
            repr(e)
        )

        print("")
        print(
            "Exception message:",
            str(e)
        )

        print("")
        print("FULL TRACEBACK:")

        traceback.print_exc()

        print("=" * 70)

        raise
