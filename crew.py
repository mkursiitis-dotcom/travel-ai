import os
import traceback
import importlib.metadata

from openai import AsyncOpenAI

from crewai import Agent, Task, Crew, Process
from crewai.llms.base_llm import BaseLLM
from crewai_tools import SerperDevTool


# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_MODEL = "openai/gpt-4o-mini"


# ============================================================
# VERSION INFORMATION
# ============================================================

def get_version(package_name):

    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "UNKNOWN"


CREWAI_VERSION = get_version("crewai")
CREWAI_TOOLS_VERSION = get_version("crewai-tools")
OPENAI_VERSION = get_version("openai")


# ============================================================
# STARTUP DEBUG
# ============================================================

print("")
print("=" * 70)
print("TRAVEL AI - CREW.PY STARTING")
print("=" * 70)

print("CrewAI version:", CREWAI_VERSION)
print("CrewAI Tools version:", CREWAI_TOOLS_VERSION)
print("OpenAI version:", OPENAI_VERSION)

print("")
print("OpenRouter configuration:")

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

print(
    "Base URL:",
    OPENROUTER_BASE_URL
)

print(
    "Model:",
    OPENROUTER_MODEL
)

print("=" * 70)


if not OPENROUTER_API_KEY:

    raise RuntimeError(
        "OPENROUTER_API_KEY is missing."
    )


# ============================================================
# OPENROUTER CLIENT
# ============================================================

openrouter_client = AsyncOpenAI(

    api_key=OPENROUTER_API_KEY,

    base_url=OPENROUTER_BASE_URL,

    default_headers={

        "HTTP-Referer": "https://travel-ai-1-5sae.onrender.com",

        "X-Title": "Latvia Travel AI"
    }
)


# ============================================================
# CUSTOM CREWAI LLM
# ============================================================
#
# IMPORTANT:
#
# We do NOT use:
#
# from crewai import LLM
#
# because CrewAI LLM -> LiteLLM -> OpenRouter
# was producing the incorrect "API key expired" error.
#
# This class calls OpenRouter directly.
#
# Architecture:
#
# CrewAI Agent
#      ↓
# DirectOpenRouterLLM
#      ↓
# OpenAI Python client
#      ↓
# OpenRouter
#
# No LiteLLM.
# ============================================================

class DirectOpenRouterLLM(BaseLLM):

    def __init__(
        self,
        model=OPENROUTER_MODEL,
        temperature=0.3,
        max_tokens=4096
    ):

        super().__init__(
            model=model,
            temperature=temperature
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        print("")
        print("=" * 70)
        print("DIRECT OPENROUTER LLM CREATED")
        print("=" * 70)

        print("Model:", self.model)
        print("Temperature:", self.temperature)
        print("Max tokens:", self.max_tokens)
        print("Uses LiteLLM: NO")
        print("Uses OpenAI client: YES")
        print("OpenRouter: YES")

        print("=" * 70)


    def call(
        self,
        messages,
        **kwargs
    ):

        """
        Synchronous CrewAI entry point.

        CrewAI may use this method for normal LLM calls.
        """

        import asyncio

        try:

            loop = asyncio.get_running_loop()

        except RuntimeError:

            loop = None


        if loop is None:

            return asyncio.run(
                self._acall(
                    messages,
                    **kwargs
                )
            )


        # If an event loop already exists, create a separate
        # thread so asyncio.run() can safely execute.

        import threading

        result_container = {
            "result": None,
            "error": None
        }


        def run():

            try:

                result_container["result"] = asyncio.run(

                    self._acall(
                        messages,
                        **kwargs
                    )

                )

            except Exception as e:

                result_container["error"] = e


        thread = threading.Thread(
            target=run
        )

        thread.start()
        thread.join()


        if result_container["error"]:

            raise result_container["error"]


        return result_container["result"]


    async def _acall(
        self,
        messages,
        **kwargs
    ):

        print("")
        print("=" * 70)
        print("DIRECT OPENROUTER CALL")
        print("=" * 70)

        print("Model:", self.model)

        print(
            "API key prefix:",
            OPENROUTER_API_KEY[:15]
        )

        print(
            "API key length:",
            len(OPENROUTER_API_KEY)
        )

        print(
            "Messages type:",
            type(messages).__name__
        )

        # ----------------------------------------------------
        # Convert CrewAI input to OpenAI/OpenRouter messages
        # ----------------------------------------------------

        if isinstance(messages, str):

            formatted_messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]

        elif isinstance(messages, list):

            formatted_messages = messages

        else:

            formatted_messages = [
                {
                    "role": "user",
                    "content": str(messages)
                }
            ]


        print(
            "Message count:",
            len(formatted_messages)
        )

        print("")
        print("Sending request directly to OpenRouter...")


        try:

            response = await openrouter_client.chat.completions.create(

                model=self.model,

                messages=formatted_messages,

                temperature=kwargs.get(
                    "temperature",
                    self.temperature
                ),

                max_tokens=kwargs.get(
                    "max_tokens",
                    self.max_tokens
                )
            )


            print("")
            print("=" * 70)
            print("OPENROUTER RESPONSE RECEIVED")
            print("=" * 70)

            print(
                "Response model:",
                response.model
            )

            print(
                "Finish reason:",
                response.choices[0].finish_reason
                if response.choices
                else "UNKNOWN"
            )


            if not response.choices:

                raise RuntimeError(
                    "OpenRouter returned no choices."
                )


            content = response.choices[0].message.content


            print(
                "Response length:",
                len(content)
                if content
                else 0
            )

            print("=" * 70)


            return content


        except Exception as e:

            print("")
            print("=" * 70)
            print("DIRECT OPENROUTER ERROR")
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

            raise


    def supports_function_calling(self):

        return False


    def supports_stop_words(self):

        return False


# ============================================================
# CREATE DIRECT LLM
# ============================================================

llm = DirectOpenRouterLLM(

    model=OPENROUTER_MODEL,

    temperature=0.3,

    max_tokens=4096
)


# ============================================================
# DIRECT OPENROUTER TEST
# ============================================================

async def debug_direct_openrouter():

    print("")
    print("=" * 70)
    print("DEBUG DIRECT OPENROUTER")
    print("=" * 70)

    try:

        response = await openrouter_client.chat.completions.create(

            model=OPENROUTER_MODEL,

            messages=[

                {
                    "role": "user",
                    "content": "Reply with exactly: OK"
                }

            ],

            temperature=0,

            max_tokens=10
        )


        content = response.choices[0].message.content


        print("")
        print("DIRECT OPENROUTER TEST SUCCESS")
        print("Response:", content)

        print("=" * 70)


        return {

            "success": True,

            "response": content,

            "model": response.model
        }


    except Exception as e:

        print("")
        print("=" * 70)
        print("DIRECT OPENROUTER TEST FAILED")
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


        return {

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e)
        }


# ============================================================
# CREWAI LLM TEST
# ============================================================

async def debug_crewai_llm():

    print("")
    print("=" * 70)
    print("DEBUG CREWAI DIRECT LLM")
    print("=" * 70)

    try:

        result = await llm._acall(

            "Reply with exactly: OK"
        )


        print("")
        print("CREWAI DIRECT LLM TEST SUCCESS")
        print("Result:", result)

        print("=" * 70)


        return {

            "success": True,

            "result": str(result),

            "model": OPENROUTER_MODEL,

            "uses_litellm": False
        }


    except Exception as e:

        print("")
        print("=" * 70)
        print("CREWAI DIRECT LLM TEST FAILED")
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


        return {

            "success": False,

            "error_type": type(e).__name__,

            "error": str(e),

            "uses_litellm": False
        }


# ============================================================
# SERPER TOOL
# ============================================================

print("")
print("Initializing SerperDevTool...")

search_tool = SerperDevTool()

print("SerperDevTool initialized.")


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
Tu labi pārzini Latvijas pilsētas, reģionus,
dabas objektus, pilis un apskates vietas.

Tu veido loģiskus maršrutus, lai ceļotājs
nepavadītu pārāk daudz laika transportā.
""",

    llm=llm,

    verbose=True
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
Tu pārzini apskates vietas, restorānus,
naktsmītnes un ceļu loģistiku.

Tu veido praktiskus plānus ar reālām izmaksām.
""",

    tools=[search_tool],

    llm=llm,

    verbose=True
)


# ============================================================
# AGENT 3
# ============================================================

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


print("")
print("=" * 70)
print("CREW INITIALIZED")
print("=" * 70)
print("Using direct OpenRouter LLM.")
print("LiteLLM is NOT used by our LLM.")
print("=" * 70)


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
    print("GENERATE TRIP")
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

        "budget": budget
    }


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
