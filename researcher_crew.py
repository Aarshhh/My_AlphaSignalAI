from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FirecrawlScrapeWebsiteTool, FirecrawlCrawlWebsiteTool
from dotenv import load_dotenv
import os
from crewai import LLM
from langfuse import Langfuse
load_dotenv()

llm = LLM(
    model="openrouter/deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

RESEARCH_PROMPT = """
### **Context of Research:**
Below is the chronological progression of topics being researched:  
**Total Topics:**  
{topics}  

### **Your Task:**  
-Perform in-depth research on the current topic: **{topic}**, ensuring it logically extends from the previous topics.  
-Only include upto to date information from {datetime}.
-Include the **Source URLs** for with the information you generate to verify the authenticity of the information.

### **Feedback:**
Incorporate the following feedback if present to refine the research content: {feedback}

Rules to follow:
- Every information should be latest, produced in {datetime}
- Do not generate any generic information, fabricated, or artificial information. 
- Every information should be from a authentic source. 
- Do not visit justdial.com for any research

"""


@CrewBase
class ResearcherCrew():
	"""ResearcherCrew crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools = [SerperDevTool(), ScrapeWebsiteTool(), FirecrawlScrapeWebsiteTool(), FirecrawlCrawlWebsiteTool()],
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
			agent = self.researcher(),
			description=RESEARCH_PROMPT,
			expected_output="A detailed response adding all the relevant information from research about the {topic} from {datetime}"
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the ResearcherCrew crew"""
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
    crew = ResearcherCrew().crew()
    langfuse = Langfuse(
		secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
		public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
		host=os.getenv("LANGFUSE_HOST"),
	)
	
    response = crew.kickoff(inputs={"description": "Write a comprehensive medium article on spark tts 0.5B model , include every feature, model methodology, detailed training process, relevant mathemtical formulas whereever needed to show any working, it's breakthroughs, benchmarks, current challenges , example usage code for a particular application", "topic": "Introduction to SparkTTS 0.5B model and its significance in LLM-based TTS synthesis.","datetime":"2025","feedback":"None"})
    langfuse.generation(
        name="Research Generation",
        model="deepseek-chat",    
        usage={
            "input": response.token_usage.prompt_tokens,
            "output": response.token_usage.completion_tokens,
            "input_cached_tokens": response.token_usage.cached_prompt_tokens,
        }
    )
    print("Response: ", response.raw)
    print("--------------------------------")
