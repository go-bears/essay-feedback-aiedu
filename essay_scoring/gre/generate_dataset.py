import argparse
import json
import math
import os

# import chevron
import pandas as pd


def main():
    # Read the full training set (TSV file)
    df = pd.read_csv("gre-essay-data.tsv", sep="\t", encoding="utf-8")

    # NaN should be none for ints
    df = df.replace({float('nan'): None})

    # Open the output jsonl file for writing
    with open("gre-essay-data.jsonl", "w") as out_f:
        # Iterate over each matching essay row
        for idx, row in df.iterrows():
            prompt = row['prompt']
            score = int(row['score'])
            essay_text = row['essay-text']
            essay_feedback = row['essay-feedback']
            task_directions = row['task-directions']

            data = {
               "prompt": prompt,
               "score": score,
               "essay_text": essay_text,
               "essay_feedback": essay_feedback,
               "task_directions": task_directions
            }

            # Write out the JSON object as one line in the jsonl file.
            out_f.write(json.dumps(data) + "\n")
        print(f"Generated file")


if __name__ == "__main__":
    main()
