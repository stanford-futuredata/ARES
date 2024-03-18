
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from datasets import Dataset
import torch
import numpy as np
import random
import os
from openai import OpenAI

#################################################

class Generation_LLM:
    def __init__(self, config):
        self.generation_llm = config.get("generation_llm")
        self.temperature = config.get("temperature")
        self.few_shot_prompt = config.get("few_shot_prompt")
        self.system_prompt = config.get("system_prompt")

        self.initialized = False

    def initialize_generation_llm(self):
        raise NotImplementedError("initialize_generation_llm method must be implemented in subclasses")

    def generate_llm_prompt(self, query, documents):
        raise NotImplementedError("generate_llm_prompt method must be implemented in subclasses")
    
    def generate_response(self, prompt):
        raise NotImplementedError("generate_response method must be implemented in subclasses")


class OpenAI_GPT(Generation_LLM):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def initialize_generation_llm(self):
        pass

    def generate_llm_prompt(self, query, documents, formatting_function):
        return formatting_function(query, documents)
        
    def generate_response(self, prompt):
        for _ in range(5):
            response = self.client.chat.completions.create(
                messages=[
                            {
                                "role": "system",
                                "content": self.system_prompt
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                model=self.generation_llm,
            ) 
            final_response = response["choices"][0]["message"]["content"]
            print("final_response")
            print(final_response)
            return final_response
