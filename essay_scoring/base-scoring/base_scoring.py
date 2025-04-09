import argparse
import os
import csv
import time
import ollama
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import json
from datetime import datetime

from prompts import *

# import essay_feedback
# from essay_feedback.data import *
BASE_DIR = "/home/kaiju/melissa_dev/ai-ed/"
PROMPTS_PATH = os.path.join(BASE_DIR, "essay_scoring", "asap-aes", "prompts")
RUBRICS_PATH = os.path.join(BASE_DIR, "essay_scoring", "asap-aes", "rubrics")

default_model = "llama3.1"
default_output_dir = "base-scoring-outputs"

def cli_main():
    """
    Command line interface for the essay scoring script.
    This function parses command line arguments, sets up the model and output directory,

    """
    parser = argparse.ArgumentParser(description="Essay Scoring Script")
    parser.add_argument("--model", type=str, default=default_model, help="LLM Model name")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Output directory")
    args = parser.parse_args()

    model = args.model
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, model), exist_ok=True)

    return model, output_dir


model, output_dir = cli_main()
date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_file = os.path.join(output_dir, model, f"{model}-scoring-output-{date_str}.tsv")


prompt_rubric_map = load_prompts_rubrics(PROMPTS_PATH, RUBRICS_PATH)
input_file = os.path.join(BASE_DIR, "essay_argument_annotation", "processed_asap_aes_data.tsv")
essay_data = pd.read_csv(input_file, sep="\t", encoding="latin1")
data_out = []

for idx, row in enumerate(tqdm(essay_data.iterrows())):
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
            essay_set=essay_set,
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
            essay_set=essay_set, 
            commentary=None, 
            score_1=None)},
        ])


    data_out.append({
        "idx": idx,
        "essay_id": essay_id,
        "essay_set": essay_set,
        "essay": essay,
        "comments": response.message.content
    })

    if idx % 100 == 0:
        
        print(row)
        print(response.message.content)
        
    if idx % 1000 == 0:
        p_output_file= os.path.join(output_dir, model, f"partial_{idx}_{model}-scoring-output-{date_str}.tsv")
        print(f"Saving intermediate results... {p_output_file}")
        pd.DataFrame(data_out).to_csv(p_output_file, index=False, sep="\t")


pd.DataFrame(data_out).to_csv(os.path.join(output_dir, output_file), index=True, sep="\t")
print("Output saved to:", output_file)


if __name__ == "__main__":
    print("Running the script...")
    print({"model": model, 
           "output_dir": output_dir,
           "output file": output_file,
          "input_file": input_file,}
          )
    
    print("Essay data preview:")
    print(essay_data.head())










