import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool


load_dotenv()


# ==========================
# LLM & TOOLS
# ==========================

llm = LLM(
    model="openrouter/openai/gpt-4o-mini"
)

search_tool = SerperDevTool()


# ==========================
# AĢENTI
# ==========================

planner = Agent(
    role="Latvijas ceļojumu plānotājs",
    goal="Izveidot optimālu un reālistisku Latvijas ceļojuma koncepciju.",
    backstory="""
Tu esi pieredzējis tūrisma plānotājs, kurš izcili pārzina Latvijas ģeogrāfiju, 
reģionus un ceļojumu loģistiku. Tu spēj izveidot sabalansētu dienas grafiku,
lai ceļotājs nepavadītu visu dienu pie stūres.
""",
    llm=llm,
    verbose=True
)


guide_logistics = Agent(
    role="Latvijas tūrisma un loģistikas eksperts",
    goal="Atrast piemērotākās apskates vietas, ēdināšanas punktus un plānot maršrutu.",
    backstory="""
Tu ļoti labi pārzini Latvijas pilis, muižas, dabas takas, muzejus, restoraānus 
un naktsmītnes. Tu māki loģiski sakārtot apskates objektus pēc to atrašanās vietas 
un aprēķināt reālistiskas izmaksas.
""",
    tools=[search_tool],
    llm=llm,
    verbose=True
)


reviewer = Agent(
    role="Ceļojumu plāna redaktors",
    goal="Izveidot nevainojamu gala ceļojuma plānu strukturētā Markdown formātā.",
    backstory="""
Tu esi pedantisks tūrisma satura redaktors. Tavs uzdevums ir pārbaudīt visa plāna
loģiku, dienu skaitu un noformēt gala atbildi ar Markdown tabulām, lai to varētu 
viegli attēlot mājaslapā.
""",
    llm=llm,
    verbose=True
)


# ==========================
# TASKI
# ==========================

task1 = Task(
    description="""
Izveido {days} dienu ceļojuma konceptu Latvijā.

Parametri:
- Sākuma pilsēta: {starting_city}
- Ceļojuma veids: {travel_style}
- Transporta veids: {transport}
- Budžets: {budget}

Prasības:
1. Norādi katras dienas galveno reģionu vai pilsētas.
2. Pārliecinies, ka ceļojuma dienu skaits ir TIEŠI {days}.
3. Ņem vērā izvēlēto transporta veidu un budžeta līmeni.
""",
    expected_output="Kopsavilkums par katras dienas tēmu un reģionu.",
    agent=planner
)


task2 = Task(
    description="""
Izstrādā detalizētu {days} dienu maršrutu ar apskates objektiem un izmaksām.

Sākuma pilsēta: {starting_city}
Ceļojuma veids: {travel_style}
Transporta veids: {transport}
Budžets: {budget}

Norādi katrai dienai:
- Konkrētus apskates objektus un dabas takas
- Ieteicamās ēdināšanas vietas (pusdienas/vakariņas)
- Naktsmītņu ieteikumus
- Aptuvenās izmaksas (degviela/biļetes/ēdiens/naktsmītne)
""",
    expected_output="Detalizēts saraksts ar objektiem, ēdināšanu un izmaksām katrai dienai.",
    agent=guide_logistics
)


task3 = Task(
    description="""
Sagatavo GALA ceļojuma plānu tīrā Markdown formātā.

Parametri:
- Sākuma pilsēta: {starting_city}
- Dienas: {days}
- Ceļojuma veids: {travel_style}
- Transporta veids: {transport}
- Budžets: {budget}

SVARĪGI FORMATĒŠANAS NOTEIKUMI:
- Katrai dienai OBLIGĀTI izveido Markdown tabulu!
- Tabulas kolonnas: | Laiks | Atrašanās vieta | Objekts / Darbība | Apraksts |
- Neizmanto parastus sarakstus dienas grafikam, TIKAI tabulas.

Izmanto šādu struktūru:

# {days} Dienu Ceļojuma Plāns ({starting_city})

## 1. Diena: [Dienas nosaukums]

| Laiks | Atrašanās vieta | Objekts / Darbība | Apraksts |
| --- | --- | --- | --- |
| 09:00 - 10:00 | {starting_city} | Izbraukšana | Došanās ceļā... |

### 🍽️ Ēdināšana un naktsmītnes
- **Pusdienas:** ...
- **Vakariņas:** ...
- **Naktsmītne:** ...

---

(Atkārto šo struktūru katrai no {days} dienām)

## 💰 Aptuvenās kopējās izmaksas
- Transporta izmaksas: ...
- Apskates objekti: ...
- Ēdināšana: ...
- Naktsmītnes: ...
- **KOPĀ:** ...

## 💡 Praktiski padomi ceļotājiem
- Tip 1...
- Tip 2...
""",
    expected_output="Pilnībā gatavs Markdown formāta ceļojuma plāns ar tabulām.",
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
# FUNKCIJA FASTAPI
# ==========================

async def generate_trip(
    starting_city: str,
    days: int,
    travel_style: str,
    transport: str,
    budget: str
):
    trip_inputs = {
        "starting_city": starting_city,
        "days": days,
        "travel_style": travel_style,
        "transport": transport,
        "budget": budget
    }

    result = await crew.kickoff_async(
        inputs=trip_inputs
    )

    return result.raw