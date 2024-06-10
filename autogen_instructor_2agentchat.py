# Demo of using Instructor in Autogen, via register_model_client function
# The two agents converse back and forth after a given starting question
# 
# AssistantJ only answers using a fixed model requiring json. 
#   the system prompt has no details about formatting
#   It does limit to one person (I found it would try to do multiple people otherwise)
#   It also has a tendency to not handle errors well, but I've done no error correction beyond the stock pydantic built in of Instructor
# AssistantQ attempts to ask Questions to continue the flow, but it often starts
#   making stuff up, despite the prompt, and adds new information and pseudojson.  The goal
#   of it was to push on AssistantJ, better than just human questions J could handle pretty easily.
#   It quickly exposes that J still isn't always formatting right, and gets flustered.
# Q is a nightmare compared to J, and it's the same model being used. 
#   The only real difference is Instructor guiding J to more structured data.

#  Improvements possible:
#     add more Validation
#     add a different Model to handle multiple people, other questions, etc.
#     add a better opponent/partner, including using a different Model via Instructor

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
# This uses locally hosted models but you could make it use other LLM services

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

        request_time = int(time.time())

        client = instructor.from_openai(
           OpenAI(
               base_url=my_url,
               api_key=my_apikey, # required, but unused
           ),
           mode=instructor.Mode.JSON,
        )        

        # we are setting this here, for testing... some way to pass this would be better.
        # response_model = None
        response_model = Person

        complete_response = client.chat.completions.create(model=my_model,messages=params["messages"],response_model=response_model, max_retries=4)
        
        if not response_model:
           return complete_response

        # If we don't use a response model, autogen is fine with the result
        # but with a response model, it won't work, so we have to build a
        # response Autogen won't choke on...

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

        #solves the annoying bug that response.cost doesn't work in autogen,
        #  if we pass back the response model as a dict, autogen dislikes the result, so objectify it.
        class DictObj:
          def __init__(self, in_dict:dict):
            assert isinstance(in_dict, dict)
            for key, val in in_dict.items():
               if isinstance(val, (list, tuple)):
                  setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
               else:
                  setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

        return DictObj(response)

    # these are to fill out the Class, and required by Autogen
    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]
        
    def cost(self, response) -> float:
        return 0
        
    @staticmethod
    def get_usage(response):
        return {}

# We tell Autogen our model client Class here in our config...            
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

# but despite passing that in the config, Autogen ALSO requires calling
# register_model_client()
# seems dumb: the config literally gives the same info (and could do this automagically)

assistantj.register_model_client(model_client_cls=InstructorModelClient)

# This is stock autogen otherwise

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
DO NOT SAY MORE THAN 10 WORDS AT A TIME, ONE SENTENCE THAT ENDS WITH '?'.

While you understand JSON, NEVER USE IT yourself.  YOU WILL GET JSON ANSWERS.
Other people besides AssistantQ CAN speak in json, only AssistantQ cannot use json.

YOUR TASK: Ask about a new person.
Always ask in the form of "Who is Joe Smith?" or "Tell me about Bill Jones?"
DO NOT ask "Who was the person ...." only ask "Who is Sarah Conner?" style questions. 
DO NOT ASK A RIDDLE.  Use a proper name. "Who is Sam Altman?"
YOU DO NOT HAVE TO STICK TO HARRY POTTER STORIES.
DO NOT ASK AGAIN ABOUT A PERSON ALREADY ASKED ABOUT.
If you can't figure out someone new to ask about, pick any famous person at random.
Only ask about one person at a time. DO NOT ask about multiple people, or potentially multiple like 'parents', or 'friends',

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
