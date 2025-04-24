import argparse
import json
import math
import os

# import chevron
import pandas as pd


def main(args):
    # Read the train IDs (should contain at least an 'essay_id' column)
    train_ids_df = pd.read_csv(args.train_ids_csv)
    val_ids_df = pd.read_csv(args.val_ids_csv)

    # Read the full training set (TSV file)
    df = pd.read_csv(args.training_set_tsv, sep="\t", encoding="ISO-8859-1")
    argument_annotation_df = pd.read_csv(args.argument_annotation_tsv, sep="\t", encoding="ISO-8859-1")

    # NaN should be none for ints
    df = df.replace({float('nan'): None})

    # Filter the full dataset to keep only rows whose essay_id is in train_ids.csv
    train_df = df[df['essay_id'].isin(train_ids_df['essay_id'])]
    val_df = df[df['essay_id'].isin(val_ids_df['essay_id'])]

    # Render the system prompt once from annotate.system.mustache
    # system_prompt = _get_templated("annotate.system")

    for split, current_outfile, current_df in zip(["train", "validation"],
                                                  [args.output_train_jsonl, args.output_val_jsonl], [train_df, val_df]):
        # Open the output jsonl file for writing
        with open(current_outfile, "w") as out_f:
            # Iterate over each matching essay row
            for idx, row in current_df.iterrows():
                essay_set = row['essay_set']
                essay_text = row['essay']
                essay_id = row['essay_id']
                domain1_score = int(row["domain1_score"])
                # Process essay set 1 by halving due to rubric mismatch
                if essay_set == 1:
                    print("Changing score from", domain1_score, "to", math.ceil(domain1_score / 2))
                    domain1_score = int(math.ceil(domain1_score / 2))
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

                # # Render the message template using annotate.message.mustache
                # rendered_message = _get_templated(
                #     "annotate.message",
                #     essay_id=int(essay_id),
                #     prompt_information=prompt_information,
                #     essay=essay_text,
                #     rubric=rubric,
                #     output_format=output_format
                # )

                # Build the JSON object for this essay.
                # data = {
                #     "instruction": system_prompt,
                #     "input": rendered_message,
                #     "output" : f"{domain1_score}" if essay_set != 2 else f"{domain1_score} {domain2_score}"
                # }
                data = {
                    "essay_id": essay_id,
                    "essay_prompt": prompt_information,
                    "essay_text": essay_text,
                    "rubric": rubric,
                    "essay_set": essay_set,
                    "grader_score": f"{domain1_score}" if essay_set != 2 else f"{domain1_score} {domain2_score}",
                    "argument_annotation": argument_annotation_df[argument_annotation_df["essay_id"] == essay_id]["segmented_essays"].values[0]
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
    parser.add_argument("--output_train_jsonl", default="train.jsonl",
                        help="Path for the output JSONL file")
    parser.add_argument("--output_val_jsonl", default="validation.jsonl",
                        help="Path for the val output JSONL file")
    parser.add_argument("--output_format_default", default="Your output should consist of one number, the score",
                        help="Output format for essays (default value)")
    parser.add_argument("--output_format_special",
                        default="Your output should consist of two scores corresponding to the two domains: Writing applications and Language Conventions, both numbers separated by one space",
                        help="Output format for essay_set == 2")
    parser.add_argument("--argument_annotation_tsv", default="../essay_argument_annotation/asap_aes_data_segmented.tsv",
                        help="Path to the argument annotation TSV file")
    args = parser.parse_args()
    main(args)
