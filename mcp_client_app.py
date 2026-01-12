"""
🤖 MCP Client Demo - Streamlit Frontend

En enkel chat-applikasjon som demonstrerer hvordan en klient
kan bruke MCP-lignende tool-calling med Azure OpenAI.

Kjør med: streamlit run mcp_client_app.py
"""

import streamlit as st
import asyncio
import json
import inspect
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
import httpx

# Last miljøvariabler
load_dotenv()

# ============================================================================
# KONFIGURASJON
# ============================================================================

st.set_page_config(
    page_title="🤖 MCP Client Demo",
    page_icon="🤖",
    layout="wide"
)

# Azure OpenAI-klient
@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

client = get_openai_client()
CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")

# ============================================================================
# MCP TOOLS (samme som i notebook)
# ============================================================================

NORWEGIAN_CITIES = {
    "Oslo": {"lat": 59.91, "lon": 10.75},
    "Bergen": {"lat": 60.39, "lon": 5.32},
    "Trondheim": {"lat": 63.43, "lon": 10.39},
    "Tromsø": {"lat": 69.65, "lon": 18.96},
    "Stavanger": {"lat": 58.97, "lon": 5.73},
    "Drammen": {"lat": 59.74, "lon": 10.20},
    "Kristiansand": {"lat": 58.15, "lon": 8.00},
}

YR_API_BASE = "https://api.met.no/weatherapi/locationforecast/2.0"


async def fetch_yr_weather(latitude: float, longitude: float) -> dict:
    """Hent værdata fra Yr API (met.no)"""
    url = f"{YR_API_BASE}/compact?lat={latitude}&lon={longitude}"
    
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            url,
            headers={"User-Agent": "MCP-Client-Demo/1.0 github.com/workshop"}
        )
        response.raise_for_status()
        return response.json()


def parse_yr_weather(data: dict) -> dict:
    """Parse Yr API response til lesbart format"""
    timeseries = data.get("properties", {}).get("timeseries", [])
    if not timeseries:
        return {"error": "Ingen værdata funnet"}
    
    current = timeseries[0]
    instant = current.get("data", {}).get("instant", {}).get("details", {})
    next_1h = current.get("data", {}).get("next_1_hours", {})
    
    return {
        "temperatur": instant.get("air_temperature"),
        "vind": instant.get("wind_speed"),
        "luftfuktighet": instant.get("relative_humidity"),
        "skydekke": instant.get("cloud_area_fraction"),
        "symbol": next_1h.get("summary", {}).get("symbol_code", "ukjent"),
    }


async def get_weather(city: str) -> str:
    """Hent ekte værmelding fra Yr for en norsk by."""
    if city not in NORWEGIAN_CITIES:
        return f"Beklager, jeg har ikke koordinater for {city}. Tilgjengelige byer: {', '.join(NORWEGIAN_CITIES.keys())}"
    
    coords = NORWEGIAN_CITIES[city]
    
    try:
        data = await fetch_yr_weather(coords["lat"], coords["lon"])
        weather = parse_yr_weather(data)
        
        symbol_map = {
            "clearsky_day": "☀️ Klarvær",
            "clearsky_night": "🌙 Klarvær",
            "fair_day": "🌤️ Lettskyet",
            "fair_night": "🌤️ Lettskyet",
            "partlycloudy_day": "⛅ Delvis skyet",
            "partlycloudy_night": "⛅ Delvis skyet",
            "cloudy": "☁️ Overskyet",
            "rain": "🌧️ Regn",
            "lightrain": "🌦️ Lett regn",
            "heavyrain": "🌧️ Kraftig regn",
            "snow": "❄️ Snø",
            "lightsnow": "🌨️ Lett snø",
            "sleet": "🌨️ Sludd",
            "fog": "🌫️ Tåke",
        }
        
        symbol_text = symbol_map.get(weather["symbol"], weather["symbol"])
        
        return (
            f"Været i {city} nå (fra yr.no):\n"
            f"🌡️ Temperatur: {weather['temperatur']}°C\n"
            f"🌤️ Forhold: {symbol_text}\n"
            f"💨 Vind: {weather['vind']} m/s\n"
            f"💧 Luftfuktighet: {weather['luftfuktighet']}%\n"
            f"☁️ Skydekke: {weather['skydekke']}%"
        )
    except Exception as e:
        return f"Kunne ikke hente værdata for {city}: {str(e)}"


def get_current_time() -> str:
    """Hent nåværende tid i Norge."""
    now = datetime.now()
    return f"Klokken er {now.strftime('%H:%M')} den {now.strftime('%d.%m.%Y')}"


def calculate(expression: str) -> str:
    """Beregn et matematisk uttrykk."""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Ugyldig uttrykk. Kun tall og +, -, *, /, () er tillatt."
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Kunne ikke beregne: {e}"


# Tool-definisjoner for OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Hent ekte værmelding fra yr.no for en norsk by (Oslo, Bergen, Trondheim, Tromsø, Stavanger, Drammen, Kristiansand)",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Navnet på byen"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Hent nåværende tid i Norge",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Beregn et matematisk uttrykk",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Matematisk uttrykk"}
                },
                "required": ["expression"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "calculate": calculate
}

# ============================================================================
# CHAT-FUNKSJON MED TOOLS
# ============================================================================

async def chat_with_tools(user_message: str, message_history: list) -> tuple[str, list]:
    """Chat med LLM som har tilgang til tools."""
    
    tool_calls_log = []
    
    messages = [
        {
            "role": "system", 
            "content": """Du er en hjelpsom assistent. Bruk de tilgjengelige verktøyene for å svare på spørsmål.
            
Tilgjengelige verktøy:
- get_weather: Hent værmelding for norske byer (Oslo, Bergen, Trondheim, Tromsø, Stavanger, Drammen, Kristiansand)
- get_current_time: Hent nåværende tid
- calculate: Beregn matematiske uttrykk

Svar alltid på norsk. Vær vennlig og hjelpsom."""
        }
    ]
    
    # Legg til historikk
    for msg in message_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Legg til ny melding
    messages.append({"role": "user", "content": user_message})
    
    # Første kall til OpenAI
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    
    # Sjekk om LLM vil bruke tools
    if assistant_message.tool_calls:
        messages.append(assistant_message)
        
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            tool_calls_log.append({
                "name": function_name,
                "args": function_args
            })
            
            # Kall funksjonen
            func = TOOL_FUNCTIONS[function_name]
            if inspect.iscoroutinefunction(func):
                function_result = await func(**function_args)
            else:
                function_result = func(**function_args)
            
            tool_calls_log[-1]["result"] = function_result
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(function_result)
            })
        
        # Få endelig svar
        final_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )
        
        return final_response.choices[0].message.content, tool_calls_log
    
    return assistant_message.content, tool_calls_log


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("🤖 MCP Client Demo")
    st.markdown("*En chat-klient som bruker MCP-lignende tool-calling med Azure OpenAI*")
    
    # Sidebar med info
    with st.sidebar:
        st.header("🔧 Tilgjengelige Tools")
        
        st.markdown("### 🌦️ get_weather")
        st.markdown("Hent ekte værdata fra yr.no")
        st.markdown("**Byer:** Oslo, Bergen, Trondheim, Tromsø, Stavanger, Drammen, Kristiansand")
        
        st.markdown("---")
        
        st.markdown("### ⏰ get_current_time")
        st.markdown("Hent nåværende tid")
        
        st.markdown("---")
        
        st.markdown("### 🧮 calculate")
        st.markdown("Beregn matematiske uttrykk")
        
        st.markdown("---")
        
        st.markdown("### 💡 Eksempler")
        st.markdown("""
        - *"Hva er været i Oslo?"*
        - *"Sammenlign temperaturen i Bergen og Tromsø"*
        - *"Hva er 15% av 850?"*
        - *"Hva er klokken?"*
        """)
        
        if st.button("🗑️ Tøm chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.tool_logs = []
            st.rerun()
    
    # Initialiser session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tool_logs" not in st.session_state:
        st.session_state.tool_logs = []
    
    # Vis chat-historikk
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Vis tool calls hvis det finnes
            if message["role"] == "assistant" and i < len(st.session_state.tool_logs):
                tool_log = st.session_state.tool_logs[i]
                if tool_log:
                    with st.expander("🔧 Tool calls", expanded=False):
                        for call in tool_log:
                            st.markdown(f"**{call['name']}**({call['args']})")
                            st.code(call['result'], language=None)
    
    # Chat input
    if prompt := st.chat_input("Skriv en melding..."):
        # Vis brukerens melding
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generer svar
        with st.chat_message("assistant"):
            with st.spinner("Tenker..."):
                # Kjør async funksjon
                response, tool_calls = asyncio.run(
                    chat_with_tools(prompt, st.session_state.messages[:-1])
                )
                
                st.markdown(response)
                
                # Vis tool calls
                if tool_calls:
                    with st.expander("🔧 Tool calls", expanded=True):
                        for call in tool_calls:
                            st.markdown(f"**{call['name']}**(`{call['args']}`)")
                            st.code(call['result'], language=None)
        
        # Lagre i historikk
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.tool_logs.append(tool_calls)


if __name__ == "__main__":
    main()
