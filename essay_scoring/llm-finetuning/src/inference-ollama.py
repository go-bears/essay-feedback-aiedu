import json
import logging
import os
import re
import secrets
from typing import Optional

import modal

from .common import VOLUME_CONFIG, MINUTES, ALLOW_WANDB, HOURS

INFERENCE_GPU_CONFIG = "A10G:2"
import re
import json

N_CLASSES = 6


def compute_kappa_summary(truth_dict, pred_dict) -> dict:
    from sklearn.metrics import cohen_kappa_score
    results: dict[str, float] = dict()
    avg_qwk = 0
    for essay_set in truth_dict:
        qwk = cohen_kappa_score(truth_dict[essay_set], pred_dict[essay_set], weights="quadratic")
        results[str(essay_set)] = qwk
        avg_qwk += qwk
    avg_qwk /= N_CLASSES
    results["avg"] = avg_qwk
    return results


def extract_domain_score(text: str, domain: int) -> Optional[str]:
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
                return str(json_obj[f"domain_{domain}_score"])

            # Alternative: use regex to extract the value
            match = re.search(domain_score_pattern, potential_json)
            if match:
                return str(match.group(1))
        except json.JSONDecodeError:
            # Not valid JSON, continue to next match
            continue

    return None


system_prompt = """
You are a helpful essay assessement assistant that scores essays based on a rubric. Please provide a 
numerical score for the provided essay according to the specified rubric.

Some guidelines are:
- These essays were written by students ranging in grade levels from Grade 7 to Grade 10 (ages 11-16).
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

Review the given rubric and prompt carefully. The essay that requires a holistic score from the rubric is as follows:

{essay_text}

Provide a numerical domain_1_score by using the provided rubric's guidance.
Output the score in JSON using the following format:
{{
    "domain_1_score": {{essay_score}},
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
    "domain_1_score": {{domain_score_1}},
    "domain_2_score": {{domain_score_2}}
}}
"""


def init_ollama():
    import httpx
    import subprocess
    import time
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    subprocess.Popen(["ollama", "serve"])

    start_time = time.time()
    timeout = 30
    interval = 2

    while True:
        try:
            # subprocess.Popen(["ollama", "serve"])
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                print("Ollama service is ready")

                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            print(
                f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
            )
            time.sleep(interval)


ollama_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(  # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HUGGINGFACE_HUB_CACHE": "/pretrained",
    })  # faster model transfers
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("ollama",
                 "httpx",
                 "loguru",
                 "huggingface_hub[hf_transfer]==0.30.1",
                 "fastapi==0.110.0",
                 "pydantic",
                 "transformers==4.51.0",
                 "datasets",
                 "unsloth",
                 "numpy",
                 "scikit-learn"
                 )
    .entrypoint([])
)

inference_app = modal.App(
    "inference",
    secrets=[
        modal.Secret.from_name("huggingface-rw-joaquin"),
        modal.Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
)


# @inference_app.function(image=ollama_image, timeout=60 * MINUTES, volumes=VOLUME_CONFIG)
# def setup_job(model_handle: str):
#
#     inference_handle = inference_job.spawn(model_handle, run_folder)
#     return inference_handle


@inference_app.function(image=ollama_image, timeout=15 * MINUTES, volumes=VOLUME_CONFIG, gpu=INFERENCE_GPU_CONFIG)
def inference_job(model_handle: str, ):
    import ollama
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    import pandas as pd
    import subprocess
    from datetime import datetime
    import numpy as np

    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = (
        f"ollama-{model_handle}-{time_string}-{secrets.token_hex(2)}"
    )
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)
    print(f"Prepared training run in {run_folder}.")

    init_ollama()

    EXPORT_PATH = os.path.join(run_folder, "results.csv")

    VOLUME_CONFIG["/pretrained"].reload()
    os.environ["OLLAMA_MODELS"] = os.path.join("pretrained", "ollama")
    subprocess.run(["ollama", "pull", "gemma3:12b"], stdout=subprocess.PIPE)
    VOLUME_CONFIG["/pretrained"].commit()

    #     modelfile_content = """
    # FROM hf.co/{repo}
    # TEMPLATE \"""
    # Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.
    #
    # ### Instruction:
    # {{{{.Instruction}}}}
    #
    # ### Input:
    # {{{{.Input}}}}
    #
    # ### Response:
    # {{{{.Response}}}}
    # \"""
    # PARAMETER stop "### Response:"
    # """.format(repo=model_handle)
    #
    #     # Write the Modelfile to disk
    #     with open("Modelfile", "w") as f:
    #         f.write(modelfile_content)
    #
    #     subprocess.run(["ollama", "create", "llama31-ft-asap", "-f", "Modelfile"], stdout=subprocess.PIPE)

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=model_handle,
    #     max_seq_length=16384,
    #     dtype=None,
    #     load_in_4bit=True,
    # )
    # FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    # EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    # EOS_TOKEN = "<|end_of_text|>"
    # def formatting_prompts_func(examples):
    #     instruction = examples["instruction"]
    #     input = examples["input"]
    #     output = "" # MASK OUTPUTS
    #     # texts = []
    #     # for instruction, input, output in zip(instructions, inputs, outputs):
    #     #     # Must add EOS_TOKEN, otherwise your generation will go on forever!
    #     text = alpaca_prompt.format(instruction, input, output)
    #     return {"text": text, }

    eval_dataset = load_dataset("jjordanoc/argumentative-asap", split="validation")
    # eval_dataset = eval_dataset.map(formatting_prompts_func)

    LIMIT = 20
    N = min(len(eval_dataset), LIMIT)
    results: dict[int, list[str]] = dict()
    ground_truths: dict[int, list[str]] = dict()
    times = np.empty((N + 1, 1), dtype=np.float32)
    for idx, grading_instruction in enumerate(eval_dataset):
        logging.debug("*" * 120)
        logging.debug("Processing essay", idx)
        logging.debug("=" * 80)
        logging.debug("Prompt:")
        logging.debug(grading_instruction)

        essay_set = int(grading_instruction["essay_set"])

        response = ollama.chat(
            model="gemma3:12b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt.format(rubric=grading_instruction["rubric"],
                                                    prompt=grading_instruction["essay_prompt"])
                },
                {
                    "role": "user",
                    "content": essay_set_2_essay_prompt.format(
                        essay_text=grading_instruction["essay_text"]) if essay_set == 2 else essay_prompt.format(
                        essay_text=grading_instruction["essay_text"])
                }
            ])
        logging.debug("=" * 80)
        logging.debug("Answer:")
        score_1 = extract_domain_score(response.message.content, 1)
        logging.debug(score_1)

        score = score_1
        if essay_set == 2:
            score_2 = extract_domain_score(response.message.content, 2)
            logging.debug(score_2)
            score = f"{score_1} {score_2}"

        if essay_set not in results:
            results[essay_set] = [score]
            ground_truths[essay_set] = [str(grading_instruction["grader_score"])]
        else:
            results[essay_set].append(score)
            ground_truths[essay_set].append(str(grading_instruction["grader_score"]))

        times[idx] = response.total_duration
        logging.debug("*" * 120)
        if idx == N:
            break

    # qwk =
    VOLUME_CONFIG["/runs"].reload()
    # Store data in a traceable format
    output = {
        "system_prompt": system_prompt,
        "essay_prompt": essay_prompt,
        "essay_prompt_set_2": essay_set_2_essay_prompt,
        "qwk_summary": compute_kappa_summary(ground_truths, results),
        "predicted_labels": results,
        "ground_truths": ground_truths,
        "avg_time_ms": float(np.average(times) / (10 ** 6)),
    }
    outfile = open(os.path.join(run_folder, "run.json"), "w")
    # outfile.write("\n".join(output))
    json.dump(output, outfile)
    outfile.close()

    # pd.DataFrame(results).to_csv(EXPORT_PATH, sep="\t")
    VOLUME_CONFIG["/runs"].commit()


"""
Llama 3.1 FT:
modal run --detach -m src.inference-ollama --model=jjordanoc/llama31-ft-asap
"""


@inference_app.local_entrypoint()
def inference_main(model: str):
    handle = inference_job.spawn(model)
    handle.get()
