import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()


# ==========================
# LLM (Ātrākais un efektīvākais modelis)
# ==========================

llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    temperature=0.3
)


# ==========================
# OPTIMIZĒTI AĢENTI (2 aģenti 3 vietā)
# ==========================

planner = Agent(
    role="Latvijas ceļojumu un loģistikas eksperts",
    goal="Izveidot optimālu, reālistisku un detalizētu Latvijas ceļojuma plānu.",
    backstory="""
Tu esi pieredzējis tūrisma eksperts, kurš izcili pārzina Latvijas ģeogrāfiju, 
pilis, dabas takas, restorānus un naktsmītnes. Tu spēj ātri saplānot loģisku 
maršrutu, lai ceļotājs nepavadītu visu dienu pie stūres un iekļautos budžetā.
""",
    llm=llm,
    verbose=True
)

reviewer = Agent(
    role="Ceļojumu plāna redaktors",
    goal="Noformēt gala ceļojuma plānu perfektā Markdown tabulu formātā.",
    backstory="""
Tu esi pedantisks tūrisma satura redaktors. Tavs vienīgais uzdevums ir strukturēt 
sniegto maršrutu nevainojamās Markdown tabulās atbilstoši norādītajām prasībām.
""",
    llm=llm,
    verbose=True
)


# ==========================
# OPTIMIZĒTI TASKI
# ==========================

task1 = Task(
    description="""
Saplāno detalizētu {days} dienu ceļojuma maršrutu Latvijā.

Parametri:
- Sākuma pilsēta: {starting_city}
- Ceļojuma veids: {travel_style}
- Transporta veids: {transport}
- Budžets: {budget}

Prasības:
1. Katrai no TIEŠI {days} dienām piemeklē reālistiskus apskates objektus, dabas takas un pilsētas.
2. Ieteic ēdināšanas vietas (pusdienas un vakariņas) un naktsmītnes atbilstoši budžetam: {budget}.
3. Nodrošini, ka loģistika ir loģiska no {starting_city} un atbilst transportam: {transport}.
""",
    expected_output="Katrai dienai sakārtots apskates objektu, ēdināšanas un naktsmītņu saraksts ar izmaksām.",
    agent=planner
)

task2 = Task(
    description="""
Sagatavo GALA ceļojuma plānu tīrā Markdown formātā par {days} dienām no {starting_city}.

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
- Padoms 1...
- Padoms 2...
""",
    expected_output="Gala Markdown ceļojuma plāns ar tabulām par katru dienu.",
    agent=reviewer
)


# ==========================
# CREW
# ==========================

crew = Crew(
    agents=[planner, reviewer],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True
)


# ==========================
# FUNKCIJA FASTAPI
# ==========================

def generate_trip(
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

    # Izmantojam parasto (sinhrono) kickoff, jo FastAPI pusē app.py izmanto run_in_threadpool
    result = crew.kickoff(inputs=trip_inputs)

    return result.raw
