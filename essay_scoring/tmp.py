import copy
import json

import numpy
from sklearn.metrics import cohen_kappa_score

# res = """{"system_prompt": "\nYou are an expert middle school teacher (ages 11-16) who scores essays based on a rubric. \nPlease provide a numerical score for the provided essay according to the specified rubric.\n\n- These essays were written by students ranging in grade levels from Grade 7 to Grade 10 (ages 11-16).\n- Provide an appropriate holistic score for limited timed test conditions where there is litte to no time for revision\n- The essay has been anonymized by replacing revealing details with tags that start with '@' and all letters are capitalized, such as '@ORGANIZATION1', '@CAPS2', '@DATE1', and etc. Do not penalize this. \n- You will carefully read the rubric and prompt, as many times as needed.\n- You will provide a detailed explanation to your decisions as to why you chose this score following the rubric and guidelines.\n- You will also make sure to be fair knowing your students are still in school.\n\nThe rubric or rubrics for this essay is as follows:\n{rubric}\n\nThe prompt is as follows:\n{prompt}\n", "essay_prompt": "\n\nReview the given rubric and prompt carefully. The essay that requires a holistic score from the rubric is as follows:\n\n{essay_text}\n\nProvide a numerical domain_1_score by using the provided rubric's guidance.\nOutput the score in JSON using the following format:\n{{\n    \"domain_1_score\": {{essay_score}}\n}}\n", "essay_prompt_set_2": "\nThis essay requires 2 scores, and you have been provided with both rubrics in the system prompt.\n\nPlease provide a numerical score for each domain based on the appropriate rubric.\nDomain 1: Writing Applications\nDomain 2: Language Conventions\n\nReview the given rubrics and prompt carefully. The essay that requires a holistic score from the rubric is as follows:\n\n{essay_text}\n\nOutput the scores in JSON using the following format:\n{{\n    \"domain_1_score\": {{domain_score_1}},\n    \"domain_2_score\": {{domain_score_2}}\n}}\n", "qwk_summary": {"2": [0.09406623735050601, 0.24689312344656167], "avg": 0.028413280066422308}, "predicted_labels": {"2": [[5.0, 4.0], [5.0, 4.0], [5.0, 4.0], [4.0, 3.0], [5.0, 4.0], [4.0, 3.0], [5.0, 3.0], [4.0, 3.0], [5.0, 4.0], [5.0, 4.0], [5.0, 4.0], [4.0, 3.0], [5.0, 3.0], [4.0, 3.0], [5.0, 4.0], [5.0, 4.0], [6.0, 4.0], [5.0, 4.0], [6.0, 4.0], [4.0, 4.0], [5.0, 4.0], [5.0, 4.0], [4.0, 3.0], [5.0, 4.0], [4.0, 4.0], [6.0, 3.0], [5.0, 3.0], [5.0, 4.0], [5.0, 4.0], [4.0, 3.0], [4.0, 3.0], [5.0, 4.0], [5.0, 4.0], [5.0, 3.0], [4.0, 4.0], [5.0, 4.0], [5.0, 4.0], [5.0, 3.0], [6.0, 4.0], [5.0, 3.0], [4.0, 3.0], [5.0, 3.0], [4.0, 2.0], [6.0, 6.0], [4.0, 4.0], [5.0, 3.0], [5.0, 3.0], [5.0, 3.0], [5.0, 3.0], [4.0, 3.0], [4.0, 2.0], [4.0, 3.0], [4.0, 3.0], [5.0, 3.0], [5.0, 4.0], [4.0, 3.0], [5.0, 3.0], [4.0, 3.0], [4.0, 3.0], [5.0, 4.0], [5.0, 3.0], [4.0, 3.0], [4.0, 3.0], [5.0, 4.0], [6.0, 4.0], [4.0, 3.0], [4.0, 3.0], [5.0, 3.0], [5.0, 3.0], [4.0, 3.0], [6.0, 4.0], [5.0, 4.0], [4.0, 3.0], [5.0, 4.0], [5.0, 4.0], [6.0, 4.0], [4.0, 3.0], [5.0, 3.0], [5.0, 4.0], [4.0, 3.0], [5.0, 3.0], [5.0, 4.0], [5.0, 3.0], [4.0, 4.0], [6.0, 4.0], [5.0, 4.0], [5.0, 3.0], [5.0, 3.0], [5.0, 4.0], [4.0, 4.0], [5.0, 3.0], [5.0, 3.0], [5.0, 3.0], [5.0, 4.0], [6.0, 4.0], [5.0, 3.0], [6.0, 4.0], [4.0, 3.0], [5.0, 3.0], [5.0, 3.0], [5.0, 4.0]]}, "ground_truths": {"2": [[5.0, 4.0], [4.0, 4.0], [3.0, 4.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0], [5.0, 4.0], [3.0, 3.0], [3.0, 3.0], [1.0, 2.0], [4.0, 4.0], [3.0, 3.0], [4.0, 4.0], [3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0], [4.0, 4.0], [4.0, 3.0], [4.0, 4.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [2.0, 2.0], [4.0, 4.0], [3.0, 4.0], [3.0, 4.0], [3.0, 3.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [3.0, 2.0], [3.0, 3.0], [4.0, 4.0], [3.0, 2.0], [2.0, 3.0], [3.0, 2.0], [2.0, 2.0], [2.0, 2.0], [5.0, 4.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [3.0, 3.0], [3.0, 4.0], [3.0, 3.0], [3.0, 3.0], [2.0, 3.0], [4.0, 4.0], [3.0, 4.0], [3.0, 3.0], [4.0, 4.0], [1.0, 1.0], [3.0, 3.0], [3.0, 2.0], [3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0], [5.0, 4.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [4.0, 3.0], [3.0, 3.0], [4.0, 4.0], [3.0, 3.0], [3.0, 2.0], [4.0, 3.0], [3.0, 4.0], [3.0, 3.0], [3.0, 4.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0], [3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 3.0], [4.0, 4.0], [3.0, 3.0], [4.0, 3.0], [4.0, 3.0], [4.0, 3.0], [3.0, 3.0], [4.0, 3.0], [3.0, 2.0], [3.0, 3.0], [4.0, 4.0], [3.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [3.0, 4.0], [4.0, 3.0], [4.0, 4.0]]}, "avg_time_ms": 7.947631342820617e+28, "sample_size": 100}"""
# res.replace("\n", "\\n")
# print("weird", res[18:21])
file = open("tmp.json", "r")
data = json.load(file)

# pred = data["predicted_labels"]["orchestrated_scores"]
# gt = data["ground_truths"]
feedbacks = data["predicted_labels"]["feedbacks"]
print(feedbacks[1])

# print("pred:", pred)
# print("gt:", gt)

# # # for tup
# # pred_1 = [int(tup[0]) for tup in pred]
# # gt_1 = [int(tup[0]) for tup in gt]


# # # # gt_1[8] = 1
# # # pred_1.append(0)
# # # gt_1.append(0)
# # print("pred 1:", pred_1)
# # print("gt 1:", gt_1)

# score = cohen_kappa_score(gt, pred, weights="quadratic")

# print(score)

# file.close()
