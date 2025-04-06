import os
import csv
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
import json
from datetime import datetime
import openai
import re
import argparse

# Load environment variables (for API key)
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def load_prompts(segmentation_prompt_file, essay_prompt_file, evaluation_prompt_file):
    with open(essay_prompt_file, 'r') as f:
        essay_prompt = f.read()
    with open(segmentation_prompt_file, 'r') as f:
        segmentation_prompt = f.read().format(essay_prompt=essay_prompt)

    with open(evaluation_prompt_file, 'r') as f:
        evaluation_prompt = f.read()
    
    return {
        "segmentation_prompt": segmentation_prompt, 
        "evaluation_prompt": evaluation_prompt
    }

def make_request(id, system_prompt, user_prompt):
    request = """{"custom_id": "essay-{id}", "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": {
            "model": "o3-mini", 
            "messages": [
                {"role": "system", "content": "{system_prompt}"},
                {"role": "user", "content": "{user_prompt}"}
            ],
            "reasoning": {"effort": "high"},
            "max_completion_tokens": 25000
        }
    }""".format(id=id, system_prompt=system_prompt, user_prompt=user_prompt)
    request = re.sub(r'[\t\n]+', '', request)
    return request


def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Essay Segmentation')
    parser.add_argument('--task', type=str, help='Select from [segment, evaluate]', choices=['segment', 'evaluate'])
    parser.add_argument('--essay-path', type=str, help='Path to essay')
    parser.add_argument('--prompt-path', type=str, help='Path to prompt directory')
    main() 