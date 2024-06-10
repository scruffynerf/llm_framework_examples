import json
import time
import autogen
import instructor
from typing_extensions import Annotated
from typing import Optional, List, Union
from pydantic import BaseModel, Field, ValidationError, field_validator
from openai import OpenAI
from autogen import UserProxyAgent, ConversableAgent 
#import logging
#logging.basicConfig(level=logging.DEBUG)


# question/prompt to use

first_question = "Who is Harry Potter?"

# LLM model given once to keep it simple
my_model = 'mistral'  #the name of your running model
#my_url = "http://127.0.0.1:11434/v1"   #the local address of the api - Ollama
my_url = "http://127.0.0.1:1234/v1"   #the local address of the api - LLM Studio
my_apikey = "NA"  # just a placeholder

#    Any instructions in class will be passed to the language model.
class Person(BaseModel):
    """
    We use this format for discussing a person
    """
    name: str = Field(None, description="Name of the person")
    birthplace: Optional[str]
    school: Optional[str]
    facts: Optional[List[str]] = Field(None, description="A list of facts about the person")

    class Config:
        extra = "allow"  # Allow extra fields that are not defined in the model, LLM will make stuff up.
        # validate_assignment = True  # Validate fields on assignment

# Our actual 'use Instructor' code
class InstructorModelClient:
    """
    An Instructor client to use with AutoGen.
    """
    def __init__(self, config, **kwargs):
        print(f"InstructorClient config: {config}")
        
    def create(self, params):

        # Create a data response class
        # adhering to AutoGen's ModelClientResponseProtocol

        # Define the current timestamp
        request_time = int(time.time())

        client = instructor.from_openai(
           OpenAI(
               base_url=my_url,
               api_key=my_apikey, # required, but unused
           ),
           mode=instructor.Mode.JSON,
        )        

        # we are setting this here, for testing...        
        # response_model = None
        response_model = Person

        complete_response = client.chat.completions.create(model=my_model,messages=params["messages"],response_model=response_model, max_retries=4)
        
        if not response_model:
           return complete_response

        current_time = int(time.time())

        # Create the response object with custom values
        response = {
            "choices": [
              {
               "finish_reason": "stop",
               "index": 0,
               "message": {
                 "content": complete_response.model_dump_json(indent=2), 
                 "role": "assistant"
               },
               "logprobs": None
              }
            ],
            "created": current_time,
            "id": f"instructor{request_time}",
            "model": my_model,
            "object": "chat.completion",
            "usage": {
               "completion_tokens": 0,
               "prompt_tokens": 0,
               "total_tokens": 0
            },
            "cost": 0,
        }

        #solves the annoying bug that response.cost doesn't work in autogen, and if we use the response model, autogen dislikes the result, so we fake it all.
        class DictObj:
          def __init__(self, in_dict:dict):
            assert isinstance(in_dict, dict)
            for key, val in in_dict.items():
               if isinstance(val, (list, tuple)):
                  setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
               else:
                  setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

        return DictObj(response)
        
    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]
        
    def cost(self, response) -> float:
        return 0
        
    @staticmethod
    def get_usage(response):
        return {}
            
instructor_llm_config = {
    "config_list": [
        {"model_client_cls":"InstructorModelClient", "model": my_model}
    ],
    "cache": None,  # seed for reproducibility
    "cache_seed": None,  # seed for reproducibility
}

assistantj = autogen.AssistantAgent(
    name="assistantj",
    system_message="""
Answer only about one person, ignore any additional people asked about. 
YOU ALWAYS ANSWER ONLY IN JSON FORMAT. 
YOU ADD NO COMMENTARIES ABOUT ERRORS.
NEVER APOLOGIZE!
If an error is reported, DO NOT comment or suggest python, ONLY REVISE YOUR PROVIDED JSON.
ADD NO EXTRA TEXT, GIVE ONLY THE JSON RESULTS.""",
    llm_config=instructor_llm_config,
)
assistantj.register_model_client(model_client_cls=InstructorModelClient)

#this is stock autogen otherwise

config_list = [
    {
        "model": my_model, 
        "base_url": my_url,
        "api_key": my_apikey,
    }
]
llm_config = {"config_list": config_list, "cache": None, "cache_seed": None}

assistantq = autogen.AssistantAgent(
    name="assistantq",
    system_message = """You are AssistantQ: the questioner.
Your job is to ask ONLY "Who is" style questions.
DO NOT SAY MORE THAN 10 WORDS AT A TIME.

While you understand JSON, NEVER USE IT yourself.  YOU WILL GET JSON ANSWERS.
Volunteer NO information yourself.  

YOUR TASK: Ask about a new person.
Always ask in the form of "Who is Joe Smith?" or "Do you know Bill Jones?"
DO NOT ask "Who was the person ...." only ask "Who is Sarah Conner?" questions. 
DO NOT ASK A RIDDLE.  Use a proper name. "Who is Sam Altman?"
YOU DO NOT HAVE TO STICK TO HARRY POTTER STORIES.
DO NOT ASK AGAIN ABOUT A PERSON ALREADY ASKED ABOUT.
If you can't figure out someone to ask about, pick any famous person at random.
Ask ONLY ONE question at a time.
DO NOT POST ANY JSON, DO NOT GIVE json-like answers. Speak normal English only.
Other people besides AssistantQ CAN speak in json, only AssistantQ cannot use json.
DO NOT list any facts.
DO NOT add new information.  
Only ask about one person at a time.
DO NOT ask about multiple people like 'parents', or 'friends',

YOU ONLY ASK LIKE THIS:  

assistantq asks: 
"Who is Mickey Mouse?"
""",
    default_auto_reply="Tell me more...",
    llm_config=llm_config,
)

chat_result = assistantq.initiate_chat(
    assistantj,
    message=first_question,
    max_turns=10,
)
