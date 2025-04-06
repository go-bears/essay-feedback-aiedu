import pandas as pd
import json
import chevron
import os
import argparse

# Define the folder where your mustache templates are stored.
TEMPLATE_FOLDER = "templates"

def _get_templated(name: str, **kwargs) -> str:
    """Helper function that renders a mustache template from TEMPLATE_FOLDER."""
    template_path = os.path.join(TEMPLATE_FOLDER, f"{name}.mustache")
    with open(template_path, "r") as f:
        return chevron.render(f, data=kwargs)

def main(args):
    # Read the train IDs (should contain at least an 'essay_id' column)
    train_ids_df = pd.read_csv(args.train_ids_csv)
    val_ids_df = pd.read_csv(args.val_ids_csv)
    
    # Read the full training set (TSV file)
    df = pd.read_csv(args.training_set_tsv, sep="\t", encoding="ISO-8859-1")

    # NaN should be none for ints
    df = df.replace({float('nan'): None})
    
    # Filter the full dataset to keep only rows whose essay_id is in train_ids.csv
    train_df = df[df['essay_id'].isin(train_ids_df['essay_id'])]
    val_df = df[df['essay_id'].isin(val_ids_df['essay_id'])]
    
    # Render the system prompt once from annotate.system.mustache
    system_prompt = _get_templated("annotate.system")
    
    for current_outfile, current_df in zip([args.output_train_jsonl, args.output_val_jsonl], [train_df, val_df]):
        # Open the output jsonl file for writing
        with open(current_outfile, "w") as out_f:
            # Iterate over each matching essay row
            for idx, row in current_df.iterrows():
                essay_set = row['essay_set']
                essay_text = row['essay']
                domain1_score = int(row["domain1_score"])
                domain2_score = int(row["domain2_score"]) if row["domain2_score"] is not None else None
                
                # Read the corresponding prompt from prompts/{essay_set}.txt
                prompt_path = os.path.join("asap-aes", "prompts", f"{essay_set}.txt")
                with open(prompt_path, "r") as f:
                    prompt_information = f.read()
                
                # Read the corresponding rubric from rubrics/{essay_set}.txt
                rubric_path = os.path.join("asap-aes", "rubrics", f"{essay_set}.txt")
                with open(rubric_path, "r") as f:
                    rubric = f.read()
                
                
                # Determine the output_format based on essay_id
                if essay_set == 2:
                    output_format = args.output_format_special
                else:
                    output_format = args.output_format_default
                
                # Render the message template using annotate.message.mustache
                rendered_message = _get_templated(
                    "annotate.message",
                    prompt_information=prompt_information,
                    essay=essay_text,
                    rubric=rubric,
                    output_format=output_format
                )
                
                # Build the JSON object for this essay.
                data = {
                    "instruction": system_prompt,
                    "input": rendered_message,
                    "output" : f"{domain1_score}" if essay_set != 2 else f"{domain1_score} {domain2_score}"
                }
                
                # Write out the JSON object as one line in the jsonl file.
                out_f.write(json.dumps(data) + "\n")
    
        print(f"Generated {current_outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate argumentative-asap.jsonl from train_ids.csv and training_set_rel3.tsv"
    )
    parser.add_argument("--train_ids_csv", default="asap-aes/train-test-val-split/train_ids.csv", 
                        help="Path to the CSV file with essay IDs")
    parser.add_argument("--val_ids_csv", default="asap-aes/train-test-val-split/val_ids.csv", 
                        help="Path to the CSV file with val essay IDs")
    parser.add_argument("--training_set_tsv", default="asap-aes/training_set_rel3.tsv", 
                        help="Path to the training_set_rel3 TSV file")
    parser.add_argument("--output_train_jsonl", default="argumentative-asap-train.jsonl", 
                        help="Path for the output JSONL file")
    parser.add_argument("--output_val_jsonl", default="argumentative-asap-val.jsonl", 
                        help="Path for the val output JSONL file")
    parser.add_argument("--output_format_default", default="Your output should consist of one number, the score", 
                        help="Output format for essays (default value)")
    parser.add_argument("--output_format_special", default="Your output should consist of two scores corresponding to the two domains: Writing applications and Language Conventions, both numbers separated by one space", 
                        help="Output format for essay_set == 2")
    
    args = parser.parse_args()
    main(args)
