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

# # Load environment variables (for API key)
# load_dotenv()

# # Initialize OpenAI client
# client = openai.OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY")
# )

def load_prompts(segmentation_prompt_file, essay_prompt_file, evaluation_prompt_file):
    with open(essay_prompt_file, 'r') as f:
        essay_prompt = f.read()
    with open(segmentation_prompt_file, 'r') as f:
        segmentation_prompt = f.read().format(essay_prompt=essay_prompt)
    segmentation_prompt = re.sub(r'\s+', " ", segmentation_prompt)
    segmentation_prompt = re.sub(r'\n', r'\\n', segmentation_prompt)
    segmentation_prompt = re.sub(r'\\(?=$|[^n])', "", segmentation_prompt)
    segmentation_prompt = re.sub(r'["’]', "'", segmentation_prompt)

    with open(evaluation_prompt_file, 'r') as f:
        evaluation_prompt = f.read()
    evaluation_prompt = re.sub(r'\s+', " ", evaluation_prompt)
    evaluation_prompt = re.sub(r'\n', r'\\n', evaluation_prompt)
    evaluation_prompt = re.sub(r'\\(?=$|[^n])', "", evaluation_prompt)
    evaluation_prompt = re.sub(r'["’]', "'", evaluation_prompt)
    
    return {
        "segmentation_prompt": segmentation_prompt, 
        "evaluation_prompt": evaluation_prompt
    }

def create_request(idx, system_prompt, user_prompt):
    request = f'''{{"custom_id": "essay-{idx}", "method": "POST", "url": "/v1/chat/completions", "body": {{"model": "o3-mini", "messages": [{{"role": "system", "content": "{system_prompt}"}},{{"role": "user", "content": "{user_prompt}"}}]}}}}'''
    return request


def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Essay Segmentation')
    parser.add_argument('--task', type=str, help='Select from [segment, evaluate]', choices=['segment', 'evaluate'])
    parser.add_argument('--essay-path', type=str, help='Path to essay')
    parser.add_argument('--prompt-path', type=str, help='Path to prompt directory')
    main() 