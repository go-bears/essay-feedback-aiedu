import argparse
import json
import math
import os

# import chevron
import pandas as pd
from datasets import Dataset
import glob

def main():
    original_data = pd.read_csv("training_set_rel3.tsv", sep="\t", encoding="ISO-8859-1")
    
    # argument_annotation_original_data = pd.read_csv(args.argument_annotation_tsv, sep="\t", encoding="ISO-8859-1")

    # NaN should be none for ints
    original_data = original_data.replace({float('nan'): None})

    trait_data = {}
    for path in glob.glob('trait-data/Prompt-*.csv'):
        # extract the essay_set number from the filename, e.g. Prompt-1.csv → 1
        fname = os.path.basename(path)
        essay_set = int(fname.split('-')[1].split('.')[0])
        
        # read and index by EssayID immediately
        df = pd.read_csv(path).set_index('EssayID')
        df.columns = df.columns.astype(str)  # Ensure column names are strings
        print(df.head())
        print(path, df.columns.tolist())
        trait_data[essay_set] = df

    current_outfile = "train.jsonl"
    data = []
    # Open the output jsonl file for writing
    with open(current_outfile, "w") as out_f:
        for idx, row in original_data.iterrows():
            essay_set = row['essay_set']
            # Skip other essay sets for now
            if essay_set not in [1, 2]:
                continue
            essay_text = row['essay']
            essay_id = row['essay_id']
            # print("essay_id", essay_id)
            domain1_score = int(row["domain1_score"])
            # Process essay set 1 by halving due to rubric mismatch
            if essay_set == 1:
                print("Changing score from", domain1_score, "to", math.ceil(domain1_score / 2))
                domain1_score = int(math.ceil(domain1_score / 2))
            domain2_score = int(row["domain2_score"]) if row["domain2_score"] is not None else None

            # Read the corresponding prompt from prompts/{essay_set}.txt
            prompt_path = os.path.join("prompts", f"{essay_set}.txt")
            with open(prompt_path, "r") as f:
                prompt_information = f.read()

            # Read the corresponding rubric from rubrics/{essay_set}.txt
            rubric_path = os.path.join("rubrics", f"{essay_set}.txt")
            with open(rubric_path, "r") as f:
                rubric = f.read()

            task_directions_path = os.path.join("task-directions", f"{essay_set}.txt")
            with open(task_directions_path, "r") as f:
                task_directions = f.read()

            try:
                trait_scores_raw = trait_data[essay_set].loc[essay_id].to_dict()
                trait_scores = {str(k): v for k, v in trait_scores_raw.items()}
            except:
                print("essay_id", essay_id, "not found in trait_data")
                continue

            
            row = {
                "essay_id": essay_id,
                "prompt": prompt_information,
                "essay_text": essay_text,
                "task_directions": task_directions,
                "essay_set": essay_set,
                "score": domain1_score,
                "trait_scores": trait_scores,
                # "argument_annotation": argument_annotation_original_data[argument_annotation_original_data["essay_id"] == essay_id]["segmented_essays"].values[0]
            }
            data.append(row)
        # print(f"Generated {current_outfile}")
    df = pd.DataFrame(data)
    print(df.head()["trait_scores"])

    # Convert all columns to string type
    # This will change the trait_scores column from dicts to string representations of dicts
    # for column in df.columns:
        # if column == "trait_scores":
            # df[column] = df[column].apply(json.dumps)
        # else:
            # df[column] = df[column].astype(str)
    # print(df["trait_scores"])
    # for value in df["trait_scores"]:
    #     assert value.keys() == ["Content", "Organization", "Word Choice", "Sentence Fluency", "Conventions"]

    # Print the trait_scores for essay_id 4335
    for idx, row in df.iterrows():
        assert row["essay_text"] != ""
        assert row["prompt"] != ""
        assert row["task_directions"] != ""
        assert list(row["trait_scores"].keys()) == ["Content", "Organization", "Word Choice", "Sentence Fluency", "Conventions"]

    ds = Dataset.from_pandas(df, preserve_index=False)
    ds.push_to_hub("jjordanoc/argumentative-asap-plus")
    print("Pushed to hub")

if __name__ == "__main__":
    main()
