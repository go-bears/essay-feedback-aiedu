import os
import re
import json

import statistics as stat
import pandas as pd

import datetime

date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

output_dir = "/home/kaiju/melissa_dev/ai-ed/base-scoring-outputs/evals"
os.makedirs(output_dir, exist_ok=True)

result_file = "/home/kaiju/melissa_dev/ai-ed/base-scoring-outputs/llama3.2/partial_1000_llama3.2-scoring-output-2025-04-09-12-19.tsv"
eval_file = "/home/kaiju/melissa_dev/ai-ed/essay_argument_annotation/processed_asap_aes_data.tsv"

def open_file(result_file, eval_file):
    result_df = pd.read_csv(result_file, sep="\t", encoding="latin1")
    eval_df = pd.read_csv(eval_file, sep="\t", encoding="latin1")[:1001]

    return result_df, eval_df

import re

def extract_json_from_comments(comments):
    # Find the first { and last } in the string
    start = comments.find('{')
    end = comments.rfind('}')
    
    if start != -1 and end != -1 and start < end:
        # Extract the JSON string
        json_str = comments[start:end+1]
        return json_str
    return None

def clean_json_string(json_str):
    # Remove any escaped quotes
    json_str = json_str.replace('\\"', '"')
    # Remove any double quotes that might be causing issues
    json_str = json_str.replace('""', '"')
    json_str = json_str.replace('None', 'null')
    return json_str

result_df, eval_data = open_file(result_file, eval_file)
result_df = result_df
eval_df = eval_data

compare_scores = []

for i, row in result_df.iterrows():
    eval_row = dict(eval_df.iloc[i])
    result_row = dict(row)

    json_data = extract_json_from_comments(row["comments"])

    if json_data:
        # Clean the JSON string
        cleaned_json = clean_json_string(json_data)
        try:
            json_dict = json.loads(cleaned_json)
            print("Successfully parsed JSON:", json_dict)

        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            print("Problematic JSON string:", cleaned_json)

    result_row["domain1_score_pred_org"] = json_dict["domain_1_score"]
    
    if eval_row["essay_set"] == 2:
        result_row["domain2_score_pred_org"] = json_dict["domain_2_score"]
    else:
        result_row["domain2_score_pred_org"] = float('nan')

    if eval_row["essay_id"] == row["essay_id"]:
        result_row["domain1_score_gt"] = eval_row["domain1_score"]
        result_row["domain2_score_gt"] = eval_row["domain2_score"]

    compare_scores.append(result_row)


compare_scores_df = pd.DataFrame(compare_scores)
compare_scores_df.to_csv(os.path.join(output_dir, f"compare_scores_all_{date_str}.tsv"), index=False, sep="\t")





