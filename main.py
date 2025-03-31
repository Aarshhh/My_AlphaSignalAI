#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()  # Load environment variables first

from random import randint
from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router

from crews.planner_crew.planner_crew import PlannerCrew
from crews.researcher_crew.researcher_crew import ResearcherCrew
from crews.validator_crew.validator_crew import ValidatorCrew
from crews.writer_crew.writer_crew import WriterCrew
from typing import Optional
from langfuse import Langfuse
import os
import argparse
from concurrent.futures import ProcessPoolExecutor



langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

class ResearchState(BaseModel):
    description: str = ""
    content: str = ""
    feedback: Optional[str] = None
    score: Optional[int] = None
    retry_count : int = 0
    total_token_usage: int = 0
    emphasizer: str = ""
    datetime: str = "2025"
    topics: list[str] = []

def run_research(topic, description, datetime, feedback, topics):
    try:
        response = ResearcherCrew().crew().kickoff(inputs = {
            "topic": topic, 
            "description": description, 
            "datetime": datetime, 
            "feedback": feedback, 
            "topics": topics
        })
        langfuse.generation(
            name="Section Research",
            model="deepseek-chat",
            usage={
                "input": response.token_usage.prompt_tokens,
                "output": response.token_usage.completion_tokens,
                "input_cached_tokens": response.token_usage.cached_prompt_tokens,
            }
        )
        return response.raw
    
    except Exception as e:
        print(f"Error in Research for topic {topic}: {e}")
        return e

class DeepResearchFlow(Flow[ResearchState]):
    def __init__(self, description: str, emphasizer: str):
        super().__init__()
        self.state.description = description
        self.state.emphasizer = emphasizer

    @start()
    def generate_plan(self):
        print("Plan Generator")
        print("Description: ", self.state.description)
        print("Emphasizer: ", self.state.emphasizer)
        try:
            response = PlannerCrew().crew().kickoff(inputs = {"description":self.state.description, "feedback":self.state.feedback, "emphasizer":self.state.emphasizer, "datetime":self.state.datetime})
        except Exception as e:
            print(f"Error in Plan Generation: {e}")
            return e
        
        self.state.topics = response.pydantic.topics
        
        langfuse.generation(
            name="Plan Generation",
            model="deepseek-chat",
            usage={
            "input": response.token_usage.prompt_tokens,
            "output": response.token_usage.completion_tokens,
            "input_cached_tokens": response.token_usage.cached_prompt_tokens,
            }
        )
        self.state.total_token_usage += response.token_usage.total_tokens
        print(f"Total Token Usage Till First Kickoff: {self.state.total_token_usage}")
    

    @listen(generate_plan)
    def section_research(self):
        try:

            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(run_research, topic, self.state.description, self.state.datetime, self.state.feedback, self.state.topics): topic for topic in self.state.topics}
                for future in futures:
                    try:
                        result = future.result()
                        self.state.content += result
                    except Exception as e:
                        print(f"Error in Research: {e}")
            print(f'Consolidated Research Draft: {self.state.content}')
            with open("research_output.md", "w", encoding="utf-8") as md_file:
                md_file.write(self.state.content) 
            print("Research content successfully saved to research_output.md")
            return self.state.content
        except Exception as e:
            print(f"Error in Research: {e}")
            return e

    '''@router(section_research)
    def validate_research(self):
        print("Validator Initialized")

        if(self.state.retry_count>2):
            print("Max Retry Limit Reached")
            return "complete"
        try:
            response = ValidatorCrew().crew().kickoff(inputs = {"content":self.state.content, "description":self.state.description})
        except Exception as e:
            print(f"Error in Validation: {e}")
            return e
        
        self.state.score = response.pydantic.score
        self.state.feedback = response.pydantic.feedback
        
        print(f'Score: {self.state.score}')
        print(f'Feedback: {self.state.feedback}')
        
        langfuse.generation(
            name="Validate Research",
            model="deepseek-chat",    
            usage={
            "input": response.token_usage.prompt_tokens,
            "output": response.token_usage.completion_tokens,
            "input_cached_tokens": response.token_usage.cached_prompt_tokens,
            }
        )
            
        self.state.total_token_usage += response.token_usage.total_tokens
        print(f"Total Token Usage Till Validation Kickoff: {self.state.total_token_usage}")
        
        self.state.retry_count += 1

        if self.state.score>6:
            return self.state.content
        return "retry"'''
    
    '''@listen(section_research)
    def write_research(self):
        print("Writer Initialized")
        try:
            response = WriterCrew().crew().kickoff(inputs = {"content":self.state.content, "description":self.state.description})
        except Exception as e:
            print(f"Error in Writing: {e}")
            return e

        langfuse.generation(
            name="Write Research",
            model="deepseek-chat",    
            usage={
            "input": response.token_usage.prompt_tokens,
            "output": response.token_usage.completion_tokens,
            "input_cached_tokens": response.token_usage.cached_prompt_tokens,
            }
        )

        print(f"Final Output: {response.raw}")
        self.state.content = response.raw
        print(f"Total Token Usage Till Writer Kickoff: {self.state.total_token_usage}")
        return self.state.content'''
    
    '''@listen("max_limit_reached")
    def retry_limit_reached(self):
        print("Max Limit Reached")
        print(f"Total Token Usage: {self.state.total_token_usage}")
        return self.state.content'''

def kickoff(description: str, emphasizer: str):
    flow = DeepResearchFlow(description, emphasizer)
    flow.kickoff()
    return flow.state.content

def plot_flow(description: str, emphasizer: str):
    flow = DeepResearchFlow(description, emphasizer)
    flow.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of the research')
    parser.add_argument('--description', type=str, help='Custom Description of the research')
    parser.add_argument('--emphasizer', type=str, help='Custom Emphasizer of the research')
    args = parser.parse_args()
    
    # Default description if no argument provided
    default_description = "Generate a comprehensive medium article on the latest qwen 2.5-omni model, include every feature, model methodology, detailed training process, relevant mathematical formulas whereever needed to show any working, it's breakthroughs, benchmarks, current challenges , example usage code for a particular application and a conclusion at the end."
    default_emphasizer = "Qwen 2.5-omni model"
    description = args.description if args.description else default_description
    emphasizer = args.emphasizer if args.emphasizer else default_emphasizer
    result = kickoff(description, emphasizer)
    print(f"Result: {result}")
    plot_flow(description, emphasizer)
