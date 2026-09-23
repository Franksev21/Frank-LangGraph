
import os

from dotenv import load_dotenv
import certifi
import streamlit as st
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

import os
from langchain_community.tools.tavily_search import TavilySearchResults


os.environ["TAVILY_API_KEY"] = "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"

search_tool = TavilySearchResults(
    max_results=2, 
    tavily_api_key=os.environ["TAVILY_API_KEY"]
)


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

resultados = search_tool.invoke("What is the capital of France?")
resultados
from anthropic import Anthropic

client = Anthropic(api_key= os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "What year is it?"}]
)

for block in response.content:
    if block.type == "text":
        print(block.text)
        break

# from langchain import hub
# prompt = hub.pull("hwchase17/react")
from langchain_core.prompts import PromptTemplate

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
tools = [search_tool, get_weather_data]
prompt
# %pip install -q langchain-anthropic

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="What years it is?"
)
result = agent.invoke({"messages": [("user", "find the capital of india" "and then find its current weather.")]})
final_message = result["messages"][-1]
print(final_message.content)