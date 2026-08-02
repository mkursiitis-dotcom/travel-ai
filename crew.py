```python
import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


load_dotenv()


# ==========================
# LLM
# ==========================

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0.3
)


# ==========================
# TOOLS
# ==========================

search_tool = SerperDevTool()


# ==========================
# AGENTS
# ==========================

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


# ==========================
# TASKS
# ==========================

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


# ==========================
# CREW
# ==========================

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


# ==========================
# API FUNCTION
# ==========================

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


    result = await crew.kickoff_async(
        inputs=inputs
    )


    return result.raw
```
