import os
import importlib.metadata
import traceback


# ============================================================
# IMPORTANT:
# Configure environment BEFORE importing CrewAI
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not configured on Render."
    )


# Force the provider-specific environment variable.
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# OpenRouter is OpenAI-compatible.
# These variables help libraries that use the OpenAI-compatible
# configuration path.
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_MODEL = "openrouter/openai/gpt-4o-mini"


# ============================================================
# PACKAGE VERSIONS
# ============================================================

def get_package_version(package_name):

    try:
        return importlib.metadata.version(package_name)
    except Exception as e:
        return f"UNKNOWN: {e}"


CREWAI_VERSION = get_package_version("crewai")
CREWAI_TOOLS_VERSION = get_package_version("crewai-tools")
LITELLM_VERSION = get_package_version("litellm")
OPENAI_VERSION = get_package_version("openai")


# ============================================================
# SAFE DEBUG
# ============================================================

print("")
print("=" * 70)
print("STARTING CREW.PY")
print("=" * 70)

print("OPENROUTER_API_KEY exists:", bool(OPENROUTER_API_KEY))
print("OPENROUTER_API_KEY length:", len(OPENROUTER_API_KEY))
print("OPENROUTER_API_KEY prefix:", OPENROUTER_API_KEY[:15])

print("")
print("OPENAI_API_BASE:")
print(os.getenv("OPENAI_API_BASE"))

print("")
print("VERSIONS:")
print("CrewAI:", CREWAI_VERSION)
print("CrewAI Tools:", CREWAI_TOOLS_VERSION)
print("LiteLLM:", LITELLM_VERSION)
print("OpenAI:", OPENAI_VERSION)

print("=" * 70)


# ============================================================
# IMPORT CREWAI
# ============================================================

from crewai import Agent, Task, Crew, Process, LLM


# ============================================================
# IMPORT LITELLM
# ============================================================

try:

    import litellm

    print("")
    print("=" * 70)
    print("LITELLM IMPORT SUCCESS")
    print("=" * 70)

    print(
        "LiteLLM version:",
        getattr(litellm, "__version__", "UNKNOWN")
    )

    print(
        "litellm.api_key exists:",
        bool(getattr(litellm, "api_key", None))
    )

    print(
        "litellm.openrouter_key exists:",
        bool(getattr(litellm, "openrouter_key", None))
    )

    if getattr(litellm, "api_key", None):

        print(
            "WARNING: LiteLLM global api_key is set!"
        )

        print(
            "Global api_key prefix:",
            str(litellm.api_key)[:15]
        )

    if getattr(litellm, "openrouter_key", None):

        print(
            "OpenRouter key inside LiteLLM prefix:",
            str(litellm.openrouter_key)[:15]
        )

    print("=" * 70)

except Exception as e:

    print("")
    print("=" * 70)
    print("LITELLM IMPORT FAILED")
    print("=" * 70)

    print(type(e).__name__)
    print(str(e))

    traceback.print_exc()

    print("=" * 70)


# ============================================================
# DIRECT LITELLM TEST
# ============================================================

async def debug_litellm():

    print("")
    print("=" * 70)
    print("DIRECT LITELLM TEST")
    print("=" * 70)

    try:

        import litellm

        print("Testing:")
        print("LiteLLM -> OpenRouter")

        print("")
        print("Model:")
        print(OPENROUTER_MODEL)

        print("")
        print("Environment:")
        print(
            "OPENROUTER_API_KEY exists:",
            bool(os.getenv("OPENROUTER_API_KEY"))
        )

        print(
            "OPENROUTER_API_KEY prefix:",
            os.getenv("OPENROUTER_API_KEY")[:15]
        )

        print(
            "OPENROUTER_API_KEY length:",
            len(os.getenv("OPENROUTER_API_KEY"))
        )

        print("")
        print("Calling litellm.acompletion()...")

        response = await litellm.acompletion(

            model=OPENROUTER_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: OK"
                }
            ],

            api_key=OPENROUTER_API_KEY,

            api_base=OPENROUTER_BASE_URL,

            temperature=0
        )

        print("")
        print("=" * 70)
        print("DIRECT LITELLM TEST SUCCESS")
        print("=" * 70)

        print(
            "Response:",
            response
        )

        try:

            content = response.choices[0].message.content

            print("")
            print("CONTENT:")
            print(content)

        except Exception:

            pass

        print("=" * 70)

        return {
            "success": True,
            "content": (
                response.choices[0].message.content
                if response.choices
                else str(response)
            )
        }

    except Exception as e:

        print("")
        print("=" * 70)
        print("DIRECT LITELLM TEST FAILED")
        print("=" * 70)

        print(
            "Exception type:",
            type(e).__name__
        )

        print(
            "Exception:",
            repr(e)
        )

        print(
            "Message:",
            str(e)
        )

        print("")
        print("FULL TRACEBACK:")

        traceback.print_exc()

        print("=" * 70)

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }


# ============================================================
# CREATE CREWAI LLM
# ============================================================

print("")
print("=" * 70)
print("CREATING CREWAI LLM")
print("=" * 70)

llm = LLM(

    model=OPENROUTER_MODEL,

    api_key=OPENROUTER_API_KEY,

    base_url=OPENROUTER_BASE_URL,

    temperature=0.3
)

print("CrewAI LLM created.")

print("=" * 70)


# ============================================================
# CREWAI LLM TEST
# ============================================================

async def debug_crewai_llm():

    print("")
    print("=" * 70)
    print("CREWAI LLM TEST")
    print("=" * 70)

    try:

        print(
            "Testing CrewAI LLM -> LiteLLM -> OpenRouter"
        )

        print("")
        print(
            "Calling llm.call()..."
        )

        result = llm.call(
            "Reply with exactly: OK"
        )

        print("")
        print("=" * 70)
        print("CREWAI LLM TEST SUCCESS")
        print("=" * 70)

        print("Result:")
        print(result)

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

        print(
            "Exception type:",
            type(e).__name__
        )

        print(
            "Exception:",
            repr(e)
        )

        print(
            "Message:",
            str(e)
        )

        print("")
        print("FULL TRACEBACK:")

        traceback.print_exc()

        print("=" * 70)

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }


# ============================================================
# SERPER
# ============================================================

from crewai_tools import SerperDevTool


search_tool = SerperDevTool()


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

        "starting_city": starting_city,

        "days": days,

        "travel_style": travel_style,

        "transport": transport,

        "budget": budget
    }

    print("")
    print("=" * 70)
    print("STARTING CREW")
    print("=" * 70)

    try:

        result = await crew.kickoff_async(
            inputs=inputs
        )

        print("")
        print("=" * 70)
        print("CREW SUCCESS")
        print("=" * 70)

        return result.raw

    except Exception as e:

        print("")
        print("=" * 70)
        print("CREW ERROR")
        print("=" * 70)

        print(
            "Type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        print("=" * 70)

        raise
