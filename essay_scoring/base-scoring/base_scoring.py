import os
import csv
import time
import ollama
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import json
from datetime import datetime

from prompts import system_prompt, essay_prompt, essay_set_2_essay_prompt, load_prompts_rubrics

# import essay_feedback
# from essay_feedback.data import *
BASE_DIR = "/home/kaiju/melissa_dev/ai-ed/"

model = "llama3.1"
output_dir = "base-scoring-outputs"
os.makedirs(os.path.join(output_dir, model), exist_ok=True)

PROMPTS_PATH = os.path.join(BASE_DIR , "essay_scoring", "asap-aes", "prompts")
RUBRICS_PATH = os.path.join(BASE_DIR "essay_scoring", "asap-aes", "rubrics")

date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_file = os.path.join(output_dir, model, f"{model}-scoring-output-{date_str}.tsv")


prompt_rubric_map = load_prompts_rubrics(PROMPTS_PATH, RUBRICS_PATH)

essay_data = pd.read_csv(os.path.join(BASE_DIR, 
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








