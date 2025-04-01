from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FirecrawlScrapeWebsiteTool, FirecrawlCrawlWebsiteTool
import os
from crewai import LLM
from dotenv import load_dotenv
from langfuse import Langfuse
from pydantic import BaseModel

load_dotenv()
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
class PlanState(BaseModel):
    topics: list[str] = []

llm = LLM(
    model="openrouter/deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

RESEARCH_PROMPT = """
**Initial Research**  
 - Understand the user's requirements as described in the Description: {description}.
 - Analyze {emphasizer}'s research, work areas, and latest innovations up to {datetime}.  
 - Based on your analysis, scour through multiple websites to research the latest progress, developments, innovations, and news in the context of {emphasizer}'s specific areas of focus during {datetime} which are most relevant to the user provided Description. 

**Output Requirements:** 
- Generate a **structured list of topics** covering the most relevant information from the research relevant to the user provided Description.  
- Include **source URLs** for each finding.  
- Do not provide fabricated or generalized information.
- Do not visit justdial.com for any research
- Generate a list of maximum 10 topics covering all the relevant information from the research which fulfills the user provided Description.
- Incorporate the following feedback if present to improve the research: {feedback}
"""

@CrewBase
class PlannerCrew():
	"""PlannerCrew crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def planner(self) -> Agent:
		return Agent(
			config=self.agents_config['planner'], 
			tools = [SerperDevTool(),ScrapeWebsiteTool(), FirecrawlScrapeWebsiteTool(), FirecrawlCrawlWebsiteTool()],
			multimodal=True,
			verbose=True,
			llm=llm
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			agent = self.planner(),
			description=RESEARCH_PROMPT,
			expected_output="A detailed list of topics, containing all the information from the research relevant to user's requirement mentioned in the Description: {description}.",
			output_pydantic=PlanState
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the PlannerCrew crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
	
if __name__ == "__main__":
    crew = PlannerCrew().crew()
    langfuse = Langfuse(
		secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
		public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
		host=os.getenv("LANGFUSE_HOST"),
	)
	
    response = crew.kickoff(inputs={"description": "Write a comprehensive medium article on spark tts 0.5B model , include every feature, model methodology, detailed training process, relevant mathemtical formulas whereever needed to show any working, it's breakthroughs, benchmarks, current challenges , example usage code for a particular application", "emphasizer":"SparkTTS 0.5B model", "datetime":"2025", "feedback":"None"})
    langfuse.generation(
        name="Plan Generation",
        model="deepseek-chat",    
        usage={
            "input": response.token_usage.prompt_tokens,
            "output": response.token_usage.completion_tokens,
            "input_cached_tokens": response.token_usage.cached_prompt_tokens,
        }
    )
    print("Response: ", response.pydantic)
    print("--------------------------------")
    print("Response: ", response.pydantic.topics)