import os
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# --- Page Config ---
st.set_page_config(
    page_title="Frank's AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Tool AI Agent")
st.caption("Search web + Weather data, powered by Claude Sonnet")


# --- Tools ---
search_tool = TavilySearchResults(
    max_results=2,
    tavily_api_key=TAVILY_API_KEY
)


@tool
def get_weather_data(city: str) -> str:
    """Get the current weather for a given city."""
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )
    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"could not fetch weather data for {city}"

    return (
        f"The current temperature in {city} is {data['current']['temperature']}°C, "
        f"with {data['current']['weather_descriptions'][0].lower()} and "
        f"humidity of {data['current']['humidity']}%."
    )


tools = [search_tool, get_weather_data]


# --- Agent (cacheado, se crea una sola vez) ---
@st.cache_resource
def setup_agent():
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use the tools available to answer questions."
    )


agent = setup_agent()


# --- UI ---
user_input = st.text_input(
    "Pregúntame algo:",
    placeholder="Ej: find the capital of india and its current weather"
)

if user_input:
    with st.spinner("Pensando..."):
        result = agent.invoke({"messages": [("user", user_input)]})
        final_message = result["messages"][-1]
    st.write(final_message.content)