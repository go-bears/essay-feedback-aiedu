import json
import logging
import os
import re
import secrets
from typing import Optional
from collections import defaultdict

import modal

# from .common import VOLUME_CONFIG, MINUTES, ALLOW_WANDB, HOURS
import re
import json
import numpy as np

INFERENCE_GPU_CONFIG = "A100:1"

N_CLASSES = 6
# LIMIT = np.inf
LIMIT = 20
ASAPResults = dict[int, list[tuple[int, int]]]


def compute_kappa_summary(truth_dict: ASAPResults, pred_dict: ASAPResults) -> dict:
    from sklearn.metrics import cohen_kappa_score
    results: dict[str, float | tuple[float, float]] = dict()
    avg_qwk = 0
    for essay_set in truth_dict:
        try:
            if essay_set == 2:
                truth_1 = [tup[0] for tup in truth_dict[essay_set]]
                pred_1 = [tup[0] for tup in pred_dict[essay_set]]
                truth_2 = [tup[1] for tup in truth_dict[essay_set]]
                pred_2 = [tup[1] for tup in pred_dict[essay_set]]
                qwk_1 = cohen_kappa_score(truth_1, pred_1, weights="quadratic")
                qwk_2 = cohen_kappa_score(truth_2, pred_2, weights="quadratic")
                results[str(essay_set)] = (qwk_1, qwk_2)
                avg_qwk += ((qwk_1 + qwk_2) / 2)
            else:
                truth_1 = [tup[0] for tup in truth_dict[essay_set]]
                pred_1 = [tup[0] for tup in pred_dict[essay_set]]
                qwk = cohen_kappa_score(truth_1, pred_1, weights="quadratic")
                results[str(essay_set)] = qwk
                avg_qwk += qwk
        except Exception as e:
            logging.error("Error computing Kappa")
            logging.error(e)
    avg_qwk /= N_CLASSES
    results["avg"] = avg_qwk
    return results

def extract_domain_score(text: str, domain: int) -> Optional[int]:
    # Step 1: Find JSON objects in the string
    # This pattern looks for text that starts with { and ends with }
    json_pattern = r'{(?:[^{}]|(?:{[^{}]*}))*}'

    # Step 2: Extract the value for domain_1_score key
    domain_score_pattern = rf'"domain_{domain}_score"\s*:\s*([0-9.]+)'

    # Find all potential JSON objects
    json_matches = re.findall(json_pattern, text)

    for potential_json in json_matches:
        try:
            # Try to parse as JSON to validate
            json_obj = json.loads(potential_json)

            # Check if our key exists directly
            if f"domain_{domain}_score" in json_obj:
                return int(json_obj[f"domain_{domain}_score"])

            # Alternative: use regex to extract the value
            match = re.search(domain_score_pattern, potential_json)
            if match:
                return int(match.group(1))
        except Exception as e:
            # Not valid JSON, continue to next match
            logging.error(e)
            continue

    return None

df = pd.read_csv("final_llama3.2-scoring-output-2025-04-09-12-19.tsv", sep="\t", encoding="ISO-8859-1")

val_ids = pd.read_csv("asap-aes/train-test-val-split/val_ids.csv")

print(val_ids)

eval_dataset = load_dataset("jjordanoc/argumentative-asap", split="validation")


