import os
import asyncio
import traceback
import time

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
    print("WARNING: OPENROUTER_API_KEY is missing")
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
        print("SerperDevTool initialized.")
    except Exception as e:
        print(f"WARNING: Serper failed: {e}")


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
# RUN SINGLE TASK
# ============================================================

async def run_single_task(
    agent,
    task,
    stage_name
):

    print("=" * 70)
    print(f"START: {stage_name}")
    print("=" * 70)

    start_time = time.time()

    try:

        single_crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )

        result = await single_crew.kickoff_async()

        elapsed = time.time() - start_time

        print("=" * 70)
        print(
            f"FINISHED: {stage_name} "
            f"({elapsed:.1f} seconds)"
        )
        print("=" * 70)

        return result.raw

    except Exception as e:

        elapsed = time.time() - start_time

        print("=" * 70)
        print(
            f"FAILED: {stage_name} "
            f"after {elapsed:.1f} seconds"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        raise


# ============================================================
# TEST CREWAI LLM
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
    budget: str,
    progress_callback=None
):

    async def progress(event):

        if progress_callback:

            try:
                await progress_callback(event)

            except Exception as e:

                print(
                    f"Progress callback error: {e}"
                )


    # ========================================================
    # TASK 1
    # ========================================================

    await progress({
        "type": "task_started",
        "stage": 1,
        "icon": "🧭",
        "title": "Ceļojumu plānotājs",
        "message": (
            "Veido maršruta koncepciju "
            "un sadala ceļojumu pa dienām."
        )
    })


    task1 = Task(

        description=f"""
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
- Izmanto reālas Latvijas vietas.

Izveido praktisku pamatu nākamajam
tūrisma ekspertam.
""",

        expected_output=(
            "Ceļojuma koncepta plāns pa dienām."
        ),

        agent=planner
    )


    try:

        plan = await run_single_task(
            planner,
            task1,
            "TASK 1 - PLANNER"
        )

    except Exception as e:

        await progress({
            "type": "error",
            "stage": 1,
            "message": (
                f"Maršruta plānošana neizdevās: "
                f"{str(e)}"
            )
        })

        raise


    await progress({
        "type": "task_completed",
        "stage": 1,
        "icon": "✅",
        "title": "Ceļojumu plānotājs",
        "message": (
            "Maršruta koncepcija ir gatava."
        )
    })


    # ========================================================
    # TASK 2
    # ========================================================

    await progress({
        "type": "task_started",
        "stage": 2,
        "icon": "🏰",
        "title": "Tūrisma un loģistikas eksperts",
        "message": (
            "Meklē apskates vietas, ēdināšanu "
            "un praktisku maršrutu."
        )
    })


    task2 = Task(

        description=f"""
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

Zemāk ir pirmā plānotāja izveidotais
maršruta koncepts:

---------------- PLAN ----------------

{plan}

-------------- END PLAN --------------

Izmanto šo konceptu kā pamatu.

Iekļauj:

- apskates objektus
- dabas takas
- ēdināšanu
- naktsmītnes
- aptuvenās izmaksas
- transporta loģiku

Izmanto reālas Latvijas vietas.

Ja ir pieejama meklēšana,
izmanto to tikai tad, ja tā
palīdz pārbaudīt konkrētu vietu,
restorānu vai naktsmītni.

Neveic nevajadzīgus atkārtotus meklējumus.
""",

        expected_output=(
            "Detalizēts ceļojuma plāns ar izmaksām."
        ),

        agent=guide_logistics
    )


    try:

        detailed_plan = await run_single_task(
            guide_logistics,
            task2,
            "TASK 2 - TOURISM AND LOGISTICS"
        )

    except Exception as e:

        await progress({
            "type": "error",
            "stage": 2,
            "message": (
                f"Tūrisma informācijas izveide "
                f"neizdevās: {str(e)}"
            )
        })

        raise


    await progress({
        "type": "task_completed",
        "stage": 2,
        "icon": "✅",
        "title": "Tūrisma un loģistikas eksperts",
        "message": (
            "Apskates vietas, maršruts un "
            "praktiskā informācija ir sagatavota."
        )
    })


    # ========================================================
    # TASK 3
    # ========================================================

    await progress({
        "type": "task_started",
        "stage": 3,
        "icon": "📝",
        "title": "Ceļojumu plāna redaktors",
        "message": (
            "Pārbauda informāciju un veido "
            "gala ceļojuma plānu."
        )
    })


    task3 = Task(

        description=f"""
Izveido gala ceļojuma plānu
Markdown formātā.

Izmanto iepriekšējo plānotāja
un tūrisma eksperta darbu.

Pirmā plānotāja koncepts:

---------------- PLAN ----------------

{plan}

-------------- END PLAN --------------


Detalizētais tūrisma eksperta plāns:

------------- DETAILS -----------------

{detailed_plan}

----------- END DETAILS ---------------


Nepārplāno ceļojumu no nulles.

Apvieno iepriekš iegūto informāciju,
pārbaudi dienu skaitu un maršruta loģiku.

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

        agent=reviewer
    )


    try:

        final_plan = await run_single_task(
            reviewer,
            task3,
            "TASK 3 - FINAL EDITOR"
        )

    except Exception as e:

        await progress({
            "type": "error",
            "stage": 3,
            "message": (
                f"Gala plāna izveide neizdevās: "
                f"{str(e)}"
            )
        })

        raise


    await progress({
        "type": "task_completed",
        "stage": 3,
        "icon": "✅",
        "title": "Ceļojumu plāna redaktors",
        "message": (
            "Gala ceļojuma plāns ir sagatavots."
        )
    })


    return final_plan
