from crewai import Agent
from langchain_openai import ChatOpenAI

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from crewai_tools import SerperDevTool

gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
search_tool = SerperDevTool()
class TripAgents():

  def city_selection_agent(self):
    return Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        tools=[
            search_tool,
            BrowserTools.scrape_and_summarize_website,
        ],
        llm=gpt4,
        verbose=True)

  def local_expert(self, context=None):
    return Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            search_tool,
            BrowserTools.scrape_and_summarize_website,
        ],
        llm=gpt4,
        verbose=True)

  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
        llm=gpt4,
        tools=[
            search_tool,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
        ],
        verbose=True)
