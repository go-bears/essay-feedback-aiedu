import os
import pandas as pd

def load_prompts_rubrics(prompts_path: str, rubrics_path: str) -> dict: 
    """
    Loads text files from prompts and rubrics folders, and 
    returns a dictionary mapping essay_set to the corresponding prompt and rubric.

    Args:
        prompts_path (str): Path to the folder containing prompt text files.
        rubrics_path (str): Path to the folder containing rubric text files.
    Returns:
        prompt_rubric_mapping (dict): A dictionary where keys are essay_set categories
          and values are dictionaries containing prompt and rubric text.
    """
    prompt_rubric_map = {}

    prompts = sorted(os.listdir(prompts_path))
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

    return prompt_rubric_map

system_prompt = """
You are a helpful essay assessement assistant that scores essays based on a rubric. Please provide a 
numerical score and commentary for the provided essay according to the specified rubric.

Some guidelines are:
- These essays were written by students ranging in grade levels from Grade 7 to Grade 10.
- Provide an appropriate holistic score for limited timed test conditions where there is litte to no time for revision
- The essay has been anonymized by replacing revealing details with tags that start with '@' and all letters are capitalized, such as '@ORGANIZATION1', '@CAPS2', '@DATE1', and etc. 
- If information has been anonymized, do not penalize the essay if organization, coherence, clarity, specificity, and style are affected.
- You may make a reasonable substitution or interpolation for the anonymized information to preserve minimal coherence and readability, 
  but do not change or edit the original essay with the substitution.

The rubric or rubrics for this essay is as follows:
{rubric}

The prompt is as follows:
{prompt}
"""

essay_prompt = """

Review the given rubric and prompt carefully. 
The essay that requires a holistic score from the rubric is as follows:

{essay_text}

Provide a numerical domain_1_score by using the provided rubric's guidance.
You may also provide your reasoning on the essay's strengths and weaknesses in regards to the provided rubric.
Output the score in JSON using the following format:
{{
    "essay_set": "{essay_set}",
    "comments": "{commentary}",
    "domain_1_score": {score_1},
    "domain_2_score": None
}}
"""

essay_set_2_essay_prompt = """
This essay requires 2 scores, and you have been provided with both rubrics in the system prompt.

Please provide a numerical score for each domain based on the appropriate rubric.
Domain 1: Writing Applications
Domain 2: Language Conventions

Review the given rubrics and prompt carefully. The essay that requires a holistic score from the rubric is as follows:

{essay_text}

Output the scores in JSON using the following format:
{{
    "essay_set": {essay_set},
    "comments": {commentary},
    "domain_1_score": {score_1},
    "domain_2_score": {score_2}
}}
"""