import json
import os
import re
import re
import secrets
from collections import defaultdict
from typing import Optional, Literal, Any

import modal
import numpy as np
from .common import (
    VOLUME_CONFIG,
    MINUTES,
    ALLOW_WANDB,
    HOURS,
    Colors,
    SUPPORTED_MODELS,
    format_prompt_inference_iter1,
    format_prompt_inference_ft,
    GREGeneralGraderPrompts,
    GREAgentPrompts,
    format_prompt_inference_gre,
    GREOrchestratorPrompts,
)

INFERENCE_GPU_CONFIG = "A100-80GB:2"

if len(INFERENCE_GPU_CONFIG.split(":")) <= 1:
    N_INFERENCE_GPUS = int(os.environ.get("N_INFERENCE_GPUS", 2))
    INFERENCE_GPU_CONFIG = f"{INFERENCE_GPU_CONFIG}:{N_INFERENCE_GPUS}"
else:
    N_INFERENCE_GPUS = int(INFERENCE_GPU_CONFIG.split(":")[-1])

N_CLASSES = 6

ASAPResults = dict[int, list[tuple[int, int]]]
GREResults = list[int]


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
                avg_qwk += (qwk_1 + qwk_2) / 2
            else:
                truth_1 = [tup[0] for tup in truth_dict[essay_set]]
                pred_1 = [tup[0] for tup in pred_dict[essay_set]]
                qwk = cohen_kappa_score(truth_1, pred_1, weights="quadratic")
                results[str(essay_set)] = qwk
                avg_qwk += qwk
        except Exception as e:
            print("Error computing Kappa")
            print(e)
    avg_qwk /= N_CLASSES
    results["avg"] = avg_qwk
    return results


def try_extract_key(key: str, text: str, dtype: Optional[type] = str) -> Optional[Any]:
    # Step 1: Find JSON objects in the string
    # This pattern looks for text that starts with { and ends with }
    json_pattern = r"{(?:[^{}]|(?:{[^{}]*}))*}"

    domain_score_pattern = rf'(?:"{key}"|\'{key}\'|""{key}"")\s*:\s*([0-9.]+)'

    # Find all potential JSON objects
    json_matches = re.findall(json_pattern, text)

    for potential_json in json_matches:
        try:
            # Alternative: use regex to extract the value
            match = re.search(domain_score_pattern, potential_json)
            if match:
                try:
                    return dtype(match.group(1))
                except Exception as e:
                    pass

            # Try to parse as JSON to validate
            json_obj = json.loads(potential_json)
            # Check if our key exists directly
            if key in json_obj:
                return dtype(json_obj[key])

        except Exception as e:
            # Not valid JSON, continue to next match
            # print("Error extracting domain score")
            # print(e)
            continue
    return None

def extract_domain_score(text: str, domain: int) -> Optional[int]:
    # Step 1: Find JSON objects in the string
    # This pattern looks for text that starts with { and ends with }
    json_pattern = r"{(?:[^{}]|(?:{[^{}]*}))*}"

    # Step 2: Extract the value for domain_1_score key
    domain_score_key = f"domain_{domain}_score"

    domain_score_pattern = rf'(?:"{domain_score_key}"|\'{domain_score_key}\'|""{domain_score_key}"")\s*:\s*([0-9.]+)'

    # Find all potential JSON objects
    json_matches = re.findall(json_pattern, text)

    for potential_json in json_matches:
        try:
            # Alternative: use regex to extract the value
            match = re.search(domain_score_pattern, potential_json)
            if match:
                try:
                    return int(match.group(1))
                except Exception as e:
                    pass

            # Try to parse as JSON to validate
            json_obj = json.loads(potential_json)
            # Check if our key exists directly
            if f"domain_{domain}_score" in json_obj:
                return int(json_obj[f"{domain_score_key}"])

        except Exception as e:
            # Not valid JSON, continue to next match
            # print("Error extracting domain score")
            # print(e)
            continue
    return None


def init_ollama():
    import httpx
    import subprocess
    import time
    import os

    # os.environ["OLLAMA_MODELS"] = "/pretrained/ollama"
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
            print(f"Waiting for Ollama service... ({int(time.time() - start_time)}s)")
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
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HUGGINGFACE_HUB_CACHE": "/pretrained",
        }
    )  # faster model transfers
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install(
        "ollama",
        "httpx",
        "huggingface_hub[hf_transfer]==0.30.1",
        "fastapi==0.110.0",
        "pydantic",
        "transformers==4.51.0",
        "datasets",
        "unsloth",
        "numpy",
        "scikit-learn",
    )
    .entrypoint([])
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    .run_commands("apt-get update && apt-get install -y build-essential")
    .pip_install(
        "vllm==0.8.2",
        "torch==2.6.0",
        "modal",
        "huggingface_hub[hf_transfer]==0.30.1",
        "pydantic",
        "transformers==4.51.0",
        "datasets",
        "unsloth",
        "numpy",
        "scikit-learn",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
    .env({"VLLM_USE_V1": "0"})
    .env({"CC": "/usr/bin/gcc"})
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


# Unified Inference Class
class UnifiedInference:
    def __init__(
        self,
        backend: Literal["ollama", "vllm"],
        model_name: str = "",
        adapters_repo: str = "",
    ):
        self.backend = backend
        self.model_name = model_name
        self.engine = None
        if adapters_repo != "":
            from huggingface_hub import snapshot_download

            self.adapters_path = snapshot_download(adapters_repo)
        else:
            self.adapters_path = None
        if self.backend == "vllm":
            with vllm_image.imports():
                from vllm import LLM

            print(
                Colors.GREEN,
                Colors.BOLD,
                f"🧠: Initializing vLLM engine for model {self.model_name}",
                Colors.END,
                sep="",
            )
            VOLUME_CONFIG["/pretrained"].reload()

            self.engine = LLM(
                model=self.model_name,
                # TODO: bring back
                tensor_parallel_size=N_INFERENCE_GPUS,
                pipeline_parallel_size=1,
                gpu_memory_utilization=0.98,
                block_size=128,
                cpu_offload_gb=0,
                max_model_len=64000,
                max_num_seqs=8,
                disable_custom_all_reduce=False,
                enable_chunked_prefill=True,
                # TODO: refactor
                # tensor_parallel_size=1,
                # pipeline_parallel_size=N_INFERENCE_GPUS,
                quantization="bitsandbytes",
                enable_lora=True,
                max_lora_rank=32,
                qlora_adapter_name_or_path=self.adapters_path,
            )
            VOLUME_CONFIG["/pretrained"].commit()
        else:  # ollama
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
                    response = httpx.get("http://localhost:11434/api/version")
                    if response.status_code == 200:
                        print("Ollama service is ready")
                        break
                except httpx.ConnectError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Ollama service failed to start")
                    print(
                        f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
                    )
                    time.sleep(interval)

    def generate(self, prompt: str) -> str:
        if self.backend == "vllm":
            from vllm.sampling_params import SamplingParams
            from vllm.utils import random_uuid
            from vllm.lora.request import LoRARequest

            sampling_params = SamplingParams(
                seed=903,
                repetition_penalty=1.1,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                max_tokens=1024,
            )
            if self.adapters_path:
                outputs = self.engine.generate(
                    prompt,
                    sampling_params,
                    lora_request=LoRARequest("asap-lora", 1, self.adapters_path),
                )
            else:
                outputs = self.engine.generate(prompt, sampling_params)
            full_response = outputs[0].outputs[0].text
            print(f"Full response: {full_response}")
            return full_response
        else:  # ollama
            import httpx

            response = httpx.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
            )
            return response.json()["response"]
        

def prompt_processing_asap(content: str, essay_set: int):
    # try:
    #     score_1 = extract_domain_score(content, 1)
    #     score_2 = -1
    #     print(score_1)

    #     if essay_set == 2:
    #         score_2 = extract_domain_score(content, 2)
    #         print(score_2)
    #     score = (score_1, score_2)
    # except Exception as e:
    #     # skip this essay
    #     print(e)
    #     return None

    # # Discard from analysis
    # if score_1 is None or score_2 is None:
    #     none_count += 1
    #     return None

    # grader_score_1 = -1
    # grader_score_2 = -1
    # if essay_set == 2:
    #     split = grading_instruction["grader_score"].split(" ")
    #     grader_score_1 = int(split[0])
    #     grader_score_2 = int(split[1])
    # else:
    #     grader_score_1 = int(grading_instruction["grader_score"])
    # grader_score = (grader_score_1, grader_score_2)

    # if essay_set not in results:
    #     results[essay_set] = [score]
    #     ground_truths[essay_set] = [grader_score]
    # else:
    #     results[essay_set].append(score)
    #     ground_truths[essay_set].append(grader_score)
    pass

def compute_kappa(ground_truths: GREResults, predicted_results: GREResults) -> float:
    from sklearn.metrics import cohen_kappa_score
    # Drop na's
    ground_truths = np.array(ground_truths)
    predicted_results = np.array(predicted_results)
    mask_complete = ~np.isnan(predicted_results)
    ground_truths = ground_truths[mask_complete]
    predicted_results = predicted_results[mask_complete]
    return round(cohen_kappa_score(ground_truths, predicted_results, weights="quadratic"), 4)

def inference_loop(
    run_folder: str,
    inference: UnifiedInference,
    few_shot_n: int = 0,
    limit: Optional[int] = None,
    add_argument_annotation: bool = False,
    adapters_repo: str = "",
    agent_prompts: Optional[Any] = None,
    agent_rubric_item: int = -1,
) -> dict:
    from datasets import load_dataset
    import time
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    orchestrated_results: GREResults = []
    results_per_domain: list[GREResults] = []
    orchestrated_feedbacks: list[str] = []
    feedbacks_per_domain: list[list[str]] = []
    averaged_scores = []

    ground_truths: GREResults = []
    raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
    none_count = 0

    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")
    # train_dataset = load_dataset("jjordanoc/argumentative-asap", split="train")
    # train_df = pd.DataFrame(train_dataset)

    n = min(len(eval_dataset), limit) if limit is not None else len(eval_dataset)
    times = np.zeros((n + 1, 1), dtype=np.float32)

    # for rubric_item in [1, 2, 3, 4, 5]:
    #     # print(Colors.BOLD + Colors.BLUE, f"Folder for aspect {rubric_item+1}", Colors.END)
    #     rubric_item_folder = os.path.join(run_folder, f"aspect_{rubric_item}")
    #     os.makedirs(rubric_item_folder, exist_ok=True)

    # ground_truths = np.array([int(row["score"]) for row in eval_dataset])

    for idx, grading_instruction in enumerate(eval_dataset):
        start_time = time.time()
        # Has to be here to match length of orchestrated_results
        ground_truths.append(int(grading_instruction["score"]))
        print("*" * 120)
        print("Processing essay", idx)
        print("=" * 80)

        domain_scores = []
        domain_feedbacks = []
        domain_responses = []
        
        for rubric_item in [1, 2, 3, 4, 5]:
            if adapters_repo != "":
                full_prompt = format_prompt_inference_ft(grading_instruction)
            else:
                # full_prompt = format_prompt_inference_iter1(
                #     grading_instruction, few_shot_n, add_argument_annotation, train_df
                # )
                full_prompt = GREAgentPrompts.format_prompt_inference(grading_instruction, rubric_item)
            
            # Use the unified inference interface
            content = inference.generate(full_prompt)

            out_str = (
                "=" * 30
                + f"Domain {rubric_item}"
                + "=" * 30
                + "\nPrompt:\n"
                + full_prompt
                + "\n\n"
                + "Response:\n"
                + content
                + "\n\n"
            )

            # Log the output
            raw_outputs.write(out_str)

            # Prompt processing
            score = try_extract_key("score", content, dtype=int)
            feedback = try_extract_key("feedback", content, dtype=str)
            if score is None:
                domain_scores.append(np.nan)
                domain_feedbacks.append(None)
                none_count += 1
                continue
            domain_scores.append(score)
            domain_feedbacks.append(feedback)
            domain_responses.append(content)

        print(Colors.GREEN + Colors.BOLD + f"Scores per domain (5):  {domain_scores}" + Colors.END)
        results_per_domain.append(domain_scores)
        feedbacks_per_domain.append(domain_feedbacks)
        avg_domain_score = np.nan if np.isnan(np.nanmean(domain_scores)) else round(np.nanmean(domain_scores))
        # Orchestration
        full_prompt = GREOrchestratorPrompts.format_prompt_inference(grading_instruction, domain_scores, domain_feedbacks)
        content = inference.generate(full_prompt)
        out_str = (
                "=" * 30
                + f"Interaction {idx}"
                + "=" * 30
                + "\nPrompt:\n"
                + full_prompt
                + "\n\n"
                + "Response:\n"
                + content
                + "\n\n"
            )
        # Log the output
        raw_outputs.write(out_str)
        # Prompt processing
        score = try_extract_key("score", content, dtype=int)
        feedback = try_extract_key("feedback", content, dtype=str)
        orchestrated_feedbacks.append(feedback)
        averaged_scores.append(avg_domain_score)
        print(Colors.GREEN + Colors.BOLD + f"Averaged scores: {averaged_scores}" + Colors.END)
        if score is None:
            orchestrated_results.append(np.nan)
            none_count += 1
            continue
        orchestrated_results.append(score)
        print(Colors.GREEN + Colors.BOLD + f"Orchestrated scores: {orchestrated_results}" + Colors.END)

        times[idx] = (time.time() - start_time) * 1000  # Convert to milliseconds
        # Periodic writes
        if idx % 10 == 0:
            with open(os.path.join(run_folder, "tmp.json"), "w") as tmp_outs:
                output = {
                    "orchestrator_prompts": GREOrchestratorPrompts.dump_prompts(),
                    "agent_prompts": GREAgentPrompts.dump_prompts(),
                    # "few_shot_n": few_shot_n,
                    # "add_argument_annotation": add_argument_annotation,
                    # "essay_prompt_set_2": essay_set_2_essay_prompt_instruction,
                    "qwk_orchestrator": compute_kappa(ground_truths, orchestrated_results),
                    "qwk_average": compute_kappa(ground_truths, averaged_scores),
                    "predicted_labels": {
                        "scores_per_domain": results_per_domain,
                        "feedbacks_per_domain": feedbacks_per_domain,
                        "orchestrated_scores": orchestrated_results,
                        "orchestrated_feedbacks": orchestrated_feedbacks,
                    },
                    "ground_truths": ground_truths,
                    "avg_time_ms": float(np.average(times)),
                    "sample_size": idx,
                    "none_count": none_count,
                }
                json.dump(output, tmp_outs)
                VOLUME_CONFIG["/runs"].commit()

        # if idx == n:
        #     break
    # Store data in a traceable format
    output = {
        "orchestrator_prompts": GREOrchestratorPrompts.dump_prompts(),
        "agent_prompts": GREAgentPrompts.dump_prompts(),
        # "few_shot_n": few_shot_n,
        # "add_argument_annotation": add_argument_annotation,
        # "essay_prompt_set_2": essay_set_2_essay_prompt_instruction,
        "qwk_orchestrator": compute_kappa(ground_truths, orchestrated_results),
        "qwk_average": compute_kappa(ground_truths, averaged_scores),
        "predicted_labels": {
            "scores_per_domain": results_per_domain,
            "feedbacks_per_domain": feedbacks_per_domain,
            "orchestrated_scores": orchestrated_results,
            "orchestrated_feedbacks": orchestrated_feedbacks,
        },
        "ground_truths": ground_truths,
        "avg_time_ms": float(np.average(times)),
        "sample_size": idx,
        "none_count": none_count,
    }
    outfile = open(os.path.join(run_folder, "run.json"), "w")
    json.dump(output, outfile)
    outfile.close()
    raw_outputs.close()
    tmp_outs.close()
    VOLUME_CONFIG["/runs"].commit()
    return output


# @inference_app.function(image=ollama_image, timeout=24 * HOURS, volumes=VOLUME_CONFIG, gpu=INFERENCE_GPU_CONFIG)
# def inference_ollama(ollama_handle: str):
#     import ollama
#     from datasets import load_dataset
#     from unsloth import FastLanguageModel
#     import subprocess
#     from datetime import datetime

#     time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#     run_name = (
#         f"ollama-{ollama_handle.replace(':', '-')}-{time_string}-{secrets.token_hex(2)}"
#     )
#     run_folder = f"/runs/{run_name}"
#     os.makedirs(run_folder, exist_ok=True)
#     print(f"Prepared training run in {run_folder}.")
#     init_ollama()

#     VOLUME_CONFIG["/pretrained"].reload()
#     subprocess.run(["ollama", "pull", ollama_handle], stdout=subprocess.PIPE)
#     VOLUME_CONFIG["/pretrained"].commit()

#     inference_loop(run_folder, ollama_handle=ollama_handle)

#     VOLUME_CONFIG["/runs"].commit()


def linear_regression_analysis(run_folder: str, results_per_domain: list[GREResults], ground_truths: list[int]):
    """
    Merge results from different domains into a single regression model.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import cohen_kappa_score
    # Convert from matrix (domain, essay) to matrix (essay, domain)
    X = np.array(results_per_domain).T
    # Result per essay (essay, 1)
    y = np.array(ground_truths)
    # Discard incomplete essays (columns with 1 or more NaNs)
    mask_complete = ~np.isnan(X).any(axis=1)
    X = X[mask_complete]
    y = y[mask_complete]
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=903)
    # Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Round to nearest integer
    y_pred = np.round(y_pred).astype(int)
    y_test = y_test.astype(int)
    # Compute metrics
    qwk = round(cohen_kappa_score(y_test, y_pred, weights="quadratic"), 4)
    print(Colors.GREEN + Colors.BOLD + f"Final QWK: {qwk}" + Colors.END)
    aggregated_output = {
        "regression_coefficients": model.coef_.tolist(),
        "regression_intercept": model.intercept_,
        "predicted_labels": y_pred.tolist(),
        "ground_truths": y_test.tolist(),
        "qwk_summary": qwk,
    }
    with open(os.path.join(run_folder, "regression_output.json"), "w") as f:
        json.dump(aggregated_output, f)

@inference_app.function(
    image=vllm_image,
    timeout=24 * HOURS,
    volumes=VOLUME_CONFIG,
    gpu=INFERENCE_GPU_CONFIG,
)
def inference_vllm(
    model_handle: str,
    few_shot_n: int = 0,
    limit: Optional[int] = None,
    add_argument_annotation: bool = False,
    adapters_repo: str = "",
):
    from datetime import datetime
    import numpy as np
    from sklearn.metrics import cohen_kappa_score


    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"vllm-{model_handle.replace(':', '-').replace('/', '-')}-{time_string}-{secrets.token_hex(2)}"
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)
    print(
        Colors.BLUE,
        Colors.BOLD,
        "https://modal.com/storage/ai-in-education-essay/main/example-runs-vol/"
        + run_name,
        Colors.END,
        sep="",
    )
    # Initialize the inference engine
    inference_engine = UnifiedInference(
        backend="vllm", model_name=model_handle, adapters_repo=adapters_repo
    )
    inference_loop(
        run_folder=run_folder,
        inference=inference_engine,
        few_shot_n=few_shot_n,
        limit=limit,
        add_argument_annotation=add_argument_annotation,
        adapters_repo=adapters_repo,
    )
    # Accumulate results lists (rows will be domains, columns will be essays)
    # results_per_domain = []
    # feedbacks_per_domain = []
    # ground_truths = []
    # # Argumentative scoring
    # for rubric_item in [1, 2, 3, 4, 5]:
    #     print(Colors.BOLD + Colors.BLUE, f"Running aspect {rubric_item+1}", Colors.END)
    #     rubric_item_folder = os.path.join(run_folder, f"aspect_{rubric_item}")
    #     os.makedirs(rubric_item_folder, exist_ok=True)
    #     output = inference_loop(
    #         run_folder=rubric_item_folder,
    #         inference=inference_engine,
    #         few_shot_n=few_shot_n,
    #         limit=limit,
    #         add_argument_annotation=add_argument_annotation,
    #         adapters_repo=adapters_repo,
    #         agent_prompts=GREAgentPrompts,
    #         agent_rubric_item=rubric_item,
    #     )
    #     feedbacks_per_domain.append(output["feedbacks"])
    #     results_per_domain.append(output["predicted_labels"])
    #     ground_truths = output["ground_truths"]
        
    
    

    # # Merge with orchestration
    # orchestration_folder = os.path.join(run_folder, "orchestration")
    # orchestration_output = inference_loop(
    #     run_folder=orchestration_folder,
    #     inference=inference_engine,
    #     few_shot_n=few_shot_n,
    #     limit=limit,
    #     add_argument_annotation=add_argument_annotation,
    #     adapters_repo=adapters_repo,
    #     agent_prompts=GREAgentPrompts,
    # )
    # ground_truths = np.array(orchestration_output["ground_truths"])
    # orchestrated_scores = np.array(orchestration_output["predicted_labels"])
    # # Filter out NaNs
    # mask_complete = ~np.isnan(orchestrated_scores)
    # orchestrated_scores = orchestrated_scores[mask_complete]
    # ground_truths = ground_truths[mask_complete]
    # final_qwk = round(cohen_kappa_score(ground_truths, orchestrated_scores, weights="quadratic"), 4)
    # print(Colors.GREEN + Colors.BOLD + f"Final QWK: {final_qwk}" + Colors.END)
    # orchestration_output["final_qwk"] = final_qwk
    # # Save results
    # with open(os.path.join(run_folder, "orchestration_output.json"), "w") as f:
    #     json.dump(orchestration_output, f)

    # linear_regression_analysis(run_folder=run_folder, results_per_domain=results_per_domain, ground_truths=ground_truths)

    VOLUME_CONFIG["/runs"].commit()

    print(
        Colors.GREEN,
        "https://modal.com/storage/ai-in-education-essay/main/example-runs-vol/"
        + run_name,
        Colors.END,
        sep="",
    )


@inference_app.local_entrypoint()
def inference_main(
    backend: str,
    model: str = "",
    shots: int = 0,
    limit: Optional[int] = None,
    arguments: bool = False,
    adapters_repo: str = "",
):
    """
    Run using vllm handle:
        modal run --detach -m src.inference --model=google/gemma-3-12b-it --backend=vllm --shots=1 --arguments

    Deepseek With arguments:
        modal run --detach -m src.inference --model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B --backend=vllm --shots=1 --arguments

    Run using lora adapters:
        modal run --detach -m src.inference --model=llama-4bit --backend=vllm --adapters_repo=jjordanoc/llama31-ft-asap

    Run using ollama handle:
        modal run --detach -m src.inference-ollama --model=gemma3:12b --backend=ollama

    Run using multiple models:
        modal run --detach -m src.inference --backend=vllm
    """
    if backend == "ollama":
        # handle = inference_ollama.spawn(model)
        pass
    else:
        if model in SUPPORTED_MODELS:
            if adapters_repo == "":
                print(
                    f"{Colors.BOLD + Colors.RED}Need valid adapters repo for {model}{Colors.END}"
                )
                return
            model = SUPPORTED_MODELS[model]
            shots = 0
            print(
                f"{Colors.BOLD + Colors.BLUE}Fine-tune defaults to 0-shot{Colors.END}"
            )
        if model == "":
            # gre_models_small = ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-12b-it", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
            gre_models_small = ["google/gemma-3-12b-it"]
            for gre_model in gre_models_small:
                inference_vllm.spawn(
                    gre_model,
                    few_shot_n=shots,   
                    limit=limit,
                    add_argument_annotation=arguments,
                    adapters_repo=adapters_repo,
                )
        else:
            handle = inference_vllm.spawn(
                model,
                few_shot_n=shots,
                limit=limit,
                add_argument_annotation=arguments,
                adapters_repo=adapters_repo,
            )
            handle.get()