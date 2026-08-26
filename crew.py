import os
import traceback

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_MODEL = "z-ai/glm-5.3-flash"


print("=" * 70)
print("CREW.PY START")
print("=" * 70)

print("OPENROUTER_API_KEY exists:", bool(OPENROUTER_API_KEY))
print("OPENROUTER_API_KEY length:", len(OPENROUTER_API_KEY))

if OPENROUTER_API_KEY:
    print(
        "OPENROUTER_API_KEY prefix:",
        OPENROUTER_API_KEY[:15]
    )
else:
    print("OPENROUTER_API_KEY prefix: EMPTY")

print("OPENROUTER_BASE_URL:", OPENROUTER_BASE_URL)
print("OPENROUTER_MODEL:", OPENROUTER_MODEL)

print("=" * 70)


if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing."
    )


# ============================================================
# LLM
# ============================================================

#
# IMPORTANT
#
# We explicitly provide:
#
#   api_key
#   base_url
#
# We do NOT rely on LiteLLM automatically discovering the key.
#
# CrewAI's LLM interface internally uses the configured
# OpenAI-compatible endpoint.
#

llm = LLM(
    model=f"openrouter/{OPENROUTER_MODEL}",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.3,
)


print("CrewAI LLM configured.")
print("=" * 70)


# ============================================================
# TOOLS
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

Tu labi pārzini Latvijas pilsētas,
reģionus, dabas objektus, pilis
un apskates vietas.

Tu veido loģiskus maršrutus,
lai ceļotājs nepavadītu pārāk daudz
laika transportā.
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

Tu pārzini apskates vietas,
restorānus, naktsmītnes
un ceļu loģistiku.

Tu veido praktiskus plānus
ar reālām izmaksām.
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
        "Izveidot profesionālu gala "
        "ceļojuma plānu Markdown formātā."
    ),

    backstory="""
Tu esi rūpīgs tūrisma satura redaktors.

Tu pārbaudi, lai dienu skaits
būtu pareizs.

Tu izveido skaidras Markdown tabulas
un praktisku gala ceļojuma plānu.
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
# SIMPLE CREWAI TEST
# ============================================================

async def test_crewai():

    print("=" * 70)
    print("CREWAI TEST START")
    print("=" * 70)

    try:

        test_agent = Agent(

            role="Test assistant",

            goal="Test whether the configured LLM can answer.",

            backstory="""
Tu esi tehnisks testa asistents.
Atbildi ļoti īsi.
""",

            llm=llm,

            verbose=True,
        )


        test_task = Task(

            description=(
                "Atbildi tikai ar vārdu OK."
            ),

            expected_output="OK",

            agent=test_agent,
        )


        test_crew = Crew(

            agents=[test_agent],

            tasks=[test_task],

            process=Process.sequential,

            verbose=True,
        )


        result = await test_crew.kickoff_async()


        print("=" * 70)
        print("CREWAI TEST SUCCESS")
        print("=" * 70)

        return {

            "success": True,

            "result": str(result),
        }


    except Exception as e:

        print("=" * 70)
        print("CREWAI TEST FAILED")
        print("=" * 70)

        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))

        traceback.print_exc()

        return {

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e),
        }


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

    print("=" * 70)
    print("GENERATE TRIP START")
    print("=" * 70)

    print("Starting city:", starting_city)
    print("Days:", days)
    print("Travel style:", travel_style)
    print("Transport:", transport)
    print("Budget:", budget)


    inputs = {

        "starting_city": starting_city,

        "days": days,

        "travel_style": travel_style,

        "transport": transport,

        "budget": budget,
    }


    try:

        result = await crew.kickoff_async(

            inputs=inputs
        )


        print("=" * 70)
        print("GENERATE TRIP SUCCESS")
        print("=" * 70)


        return result.raw


    except Exception as e:

        print("=" * 70)
        print("GENERATE TRIP FAILED")
        print("=" * 70)

        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))

        traceback.print_exc()

        raise
