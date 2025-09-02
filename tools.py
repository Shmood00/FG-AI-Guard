from dotenv import load_dotenv
import requests, os
from typing import List
import pandas as pd
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


load_dotenv()

wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
search = DuckDuckGoSearchRun(api_wrapper=wrapper)


def duckduckgo_wrapper(input: str) -> str:
    return search.run(input)

def read_sensitive_file(doc_name: str) -> str:
    """
    Reads the contents of a sensitive file dynamically based on the provided doc_name.
    The LLM should provide 'doc_name' when calling this tool.
    """
    import os

    # Ensure file exists
    if not os.path.exists(f"docs/{doc_name}"):
        raise FileNotFoundError(f"❌ The requested document '{doc_name}' does not exist.")

    with open(f"docs/{doc_name}", "r") as f:
        content = f.read()

    return content


def get_weather(location: str) -> str:
    """
    Takes a user provided location and returns the weather for the current day.
    """
    url = f"https://api.weatherapi.com/v1/current.json?key={os.getenv("WEATHER_API")}&q={location}"

    resp = requests.get(url).json()

    return f"The weather for {resp.get("location").get("name")} is\
          🌡️ {resp.get("current").get("temp_c")} for today with {resp.get("current").get("condition").get("text")} conditions."


def reverse_input(input: str) -> str:
    """
    Takes user input and returns the same input, but reversed.
    """
    return input[::-1]

def generate_barchart(x_axis: List[str], y_axis: List[int]):
    """
    Takes usr provided list of information and plots the data on an x-y axis with a barchart.
    """
    if len(x_axis) != len(y_axis):
        raise ValueError("x_axis and y_axis must be the same length")
    
    df = pd.DataFrame({"x": x_axis, "y": y_axis}).set_index("x")
    return df