import os
import csv
import time
import ollama
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import json
from datetime import datetime

# import essay_feedback
# from essay_feedback.data import *

model = "llama3.1"
output_dir = "base-scoring-outputs"
os.makedirs(os.path.join(output_dir, model), exist_ok=True)
date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_file = os.path.join(output_dir, model, f"{model}-scoring-output-{date_str}.tsv")


prompt_rubric_map = {}
base = "/home/kaiju/melissa_dev/ai-ed/"

prompts_path = os.path.join(base, "essay_scoring", "asap-aes", "prompts")
prompts = sorted(os.listdir(prompts_path))

rubrics_path = os.path.join(base, "essay_scoring", "asap-aes", "rubrics")
rubrics = sorted(os.listdir(rubrics_path))

for i in range(1, len(prompts) + 1):
    with open(os.path.join(prompts_path, prompts[i - 1]), "r") as f:
        prompt = f.read()
    with open(os.path.join(rubrics_path, rubrics[i - 1]), "r") as f:
        rubric = f.read()

    prompt_rubric_map[i] = {
        "prompt": prompt,
        "rubric": rubric
    }

system_prompt = """
You are a helpful essay assessement assistant that scores essays based on a rubric. Please provide a 
numerical score for the provided essay according to the specified rubric.

Some guidelines are:
- These essays were written by students ranging in grade levels from Grade 7 to Grade 10. Provide an appropriate holistic score for timed test conditions and grade level
- The essay will be anonymized by replacing revealing details with all-caps tags starting with '@', such as '@ORGANIZATION1', '@CAPS2', and '@DATE1'. 
- If information has been anonymized, do not penalize the essay if organization, coherence, clarity, and style are affected by anonymized information

The rubric or rubrics for this question is as follows:
{rubric}

The prompt is as follows:
{prompt}
"""

essay_prompt = """

The essay to score is as follows:

{essay_text}

Only provide a domain 1 score using the provided rubric.
Output the score in JSON using the following format:
{{
    "essay_id": "{essay_id}",
    "comments": "{commentary}",
    "domain_1_score": {score_1},
    "domain_2_score": None
}}
"""

essay_set_2_essay_prompt = """
This essay requires 2 scores, and you have been provided with both rubrics in the system prompt.

Please provide a score for each domain.
Domain 1: Writing Applications
Domain 2: Language Conventions

The essay to score is as follows:

{essay_text}

Output the score in JSON using the following format:
{{
    "essay_id": {essay_id},
    "comments": {commentary},
    "domain_1_score": {score_1},
    "domain_2_score": {score_2}
}}
"""

essay_data = pd.read_csv(os.path.join(base, 
                                      "essay_argument_annotation", 
                                      "processed_asap_aes_data.tsv"),
                                      sep="\t", 
                                      encoding="latin1")

data_out = []

for idx, row in enumerate(tqdm(essay_data.iterrows())):
    print(row)
    essay_id = row[1]["essay_id"]
    essay_set = row[1]["essay_set"]
    essay = row[1]["essay"]
  

    runtime_prompt = prompt_rubric_map[essay_set]["prompt"]
    runtime_rubric = prompt_rubric_map[essay_set]["rubric"]
 

    if row[1]["essay_set"] == 2:
        response = ollama.chat(model=model, messages=[
            {"role": "system", 
            "content": system_prompt.format(rubric=runtime_rubric, 
            prompt=runtime_prompt,)
            },
            {"role": "user", 
            "content": essay_set_2_essay_prompt.format(essay_text=essay,
            essay_id=essay_id,
            commentary=None,
            score_1=None,
            score_2=None)}
        ])
    else:
        response = ollama.chat(model=model, messages=[
            {"role": "system",
            "content": system_prompt.format(
                rubric=runtime_rubric, 
                prompt=runtime_prompt)},
            {"role": "user", 
            "content": essay_prompt.format(essay_text=essay,
            essay_id=essay_id, 
            commentary=None, 
            score_1=None)},
        ])

    print(response.message.content)

    data_out.append({
        "idx": idx,
        "essay_id": essay_id,
        "essay_set": essay_set,
        "essay": essay,
        "comments": response.message.content
    })

    if idx % 100 == 0:
        pd.DataFrame(data_out).to_csv(output_file, index=False, sep="\t")

pd.DataFrame(data_out).to_csv(output_file, index=True, sep="\t")








