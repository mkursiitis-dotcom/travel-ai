import os
import asyncio
import traceback

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# ENVIRONMENT
# ============================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("=" * 70)
print("CREW.PY STARTING")
print("=" * 70)

print(f"OPENROUTER_API_KEY exists: {bool(OPENROUTER_API_KEY)}")
print(f"OPENROUTER_API_KEY length: {len(OPENROUTER_API_KEY)}")

if OPENROUTER_API_KEY:
    print(
        f"OPENROUTER_API_KEY prefix: "
        f"{OPENROUTER_API_KEY[:15]}..."
    )

print(f"SERPER_API_KEY exists: {bool(SERPER_API_KEY)}")
print(f"SERPER_API_KEY length: {len(SERPER_API_KEY)}")

print("=" * 70)


# ============================================================
# VALIDATE OPENROUTER KEY
# ============================================================

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing from Render environment variables."
    )


# ============================================================
# LLM
# ============================================================
#
# IMPORTANT:
#
# We explicitly provide:
#
#   api_key
#   base_url
#
# instead of relying on LiteLLM/CrewAI to automatically
# discover the OpenRouter environment variable.
#
# This is the important change.
#
# ============================================================

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",

    api_key=OPENROUTER_API_KEY,

    base_url="https://openrouter.ai/api/v1",

    temperature=0.3,

    max_tokens=4000,
)


print("=" * 70)
print("CREWAI LLM CONFIGURED")
print("=" * 70)
print("Model: openrouter/openai/gpt-4o-mini")
print("Base URL: https://openrouter.ai/api/v1")
print(f"Explicit API key supplied: {bool(OPENROUTER_API_KEY)}")
print("=" * 70)


# ============================================================
# SERPER TOOL
# ============================================================

search_tool = None

try:

    if SERPER_API_KEY:

        search_tool = SerperDevTool()

        print("✅ SerperDevTool initialized")

    else:

        print(
            "⚠️ SERPER_API_KEY missing. "
            "Search tool will not be used."
        )

except Exception as e:

    print("⚠️ Could not initialize SerperDevTool:")
    print(str(e))

    search_tool = None


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


# ------------------------------------------------------------

guide_logistics_tools = []

if search_tool is not None:
    guide_logistics_tools.append(search_tool)


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

Izmanto interneta meklēšanu, ja tā ir pieejama.
""",

    tools=guide_logistics_tools,

    llm=llm,

    verbose=True
)


# ------------------------------------------------------------

reviewer = Agent(

    role="Ceļojumu plāna redaktors",

    goal=(
        "Izveidot profesionālu gala ceļojuma "
        "plānu Markdown formātā."
    ),

    backstory="""
Tu esi rūpīgs tūrisma satura redaktors.

Tu pārbaudi, lai dienu skaits būtu pareizs,
maršruts būtu loģisks un informācija būtu
skaidri strukturēta.

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
- Izvairies no nevajadzīgiem gariem pārbraucieniem.
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

Ja izmanto meklēšanas rīku, priekšroku dod
reālām un pārbaudāmām vietām.
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

Gala rezultātam jābūt skaidram un lietojamam
reālam ceļotājam.
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
# SIMPLE LLM TEST
# ============================================================

async def test_crewai_llm():

    """
    Minimal CrewAI LLM test.

    This is intentionally separate from the full trip
    generation process.

    If this fails with 401, the problem is still in
    CrewAI/LiteLLM/OpenRouter configuration.
    """

    print("=" * 70)
    print("STARTING CREWAI LLM TEST")
    print("=" * 70)

    try:

        response = await asyncio.to_thread(
            llm.call,
            [
                {
                    "role": "user",
                    "content": (
                        "Atbildi tikai ar tekstu: "
                        "CrewAI OpenRouter tests successful."
                    )
                }
            ]
        )

        print("=" * 70)
        print("✅ CREWAI LLM TEST SUCCESS")
        print("=" * 70)

        print("Response:")
        print(str(response)[:1000])

        print("=" * 70)

        return {
            "success": True,
            "response": str(response)
        }

    except Exception as e:

        print("=" * 70)
        print("❌ CREWAI LLM TEST FAILED")
        print("=" * 70)

        print("Error type:")
        print(type(e).__name__)

        print("Error:")
        print(str(e))

        traceback.print_exc()

        print("=" * 70)

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
    print("GENERATE TRIP STARTED")
    print("=" * 70)

    print(f"starting_city = {starting_city}")
    print(f"days = {days}")
    print(f"travel_style = {travel_style}")
    print(f"transport = {transport}")
    print(f"budget = {budget}")

    inputs = {

        "starting_city": starting_city,

        "days": days,

        "travel_style": travel_style,

        "transport": transport,

        "budget": budget
    }

    try:

        print("🚀 Calling CrewAI kickoff_async...")

        result = await crew.kickoff_async(
            inputs=inputs
        )

        print("=" * 70)
        print("✅ CREWAI TRIP GENERATION SUCCESS")
        print("=" * 70)

        print(f"Result type: {type(result).__name__}")

        if hasattr(result, "raw"):

            print(
                f"Result length: "
                f"{len(str(result.raw))}"
            )

            return result.raw

        return str(result)

    except Exception as e:

        print("=" * 70)
        print("❌ CREWAI TRIP GENERATION FAILED")
        print("=" * 70)

        print("Error type:")
        print(type(e).__name__)

        print("Error:")
        print(str(e))

        traceback.print_exc()

        print("=" * 70)

        raise
