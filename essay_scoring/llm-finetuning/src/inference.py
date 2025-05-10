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
    GREGeneralGraderPrompts,
    GREAgentPrompts,
    GREOrchestratorPrompts,
    RubricJudgePrompts,
    ASAPGeneralGraderPrompts,
    ASAPAgentAlphaPrompts,
    ASAPOrchestratorPrompts,
)

INFERENCE_GPU_CONFIG = "A100-80GB:4"

if len(INFERENCE_GPU_CONFIG.split(":")) <= 1:
    N_INFERENCE_GPUS = int(os.environ.get("N_INFERENCE_GPUS", 2))
    INFERENCE_GPU_CONFIG = f"{INFERENCE_GPU_CONFIG}:{N_INFERENCE_GPUS}"
else:
    N_INFERENCE_GPUS = int(INFERENCE_GPU_CONFIG.split(":")[-1])

N_CLASSES = 6




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
        "pydantic",
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
        modal.Secret.from_name("openai-secret-xavier"),
    ],
)


# Unified Inference Class
class UnifiedInference:
    def __init__(
        self,
        model_name: str = "",
        adapters_repo: str = "",
    ):
        self.model_name = model_name
        self.engine = None
        if adapters_repo != "":
            from huggingface_hub import snapshot_download

            self.adapters_path = snapshot_download(adapters_repo)
        else:
            self.adapters_path = None
        
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

    def generate(self, prompt: str) -> str:
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
        return full_response
        

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

def compute_kappa(ground_truths: list[int], predicted_results: list[int]) -> float:
    from sklearn.metrics import cohen_kappa_score
    # Drop na's
    ground_truths = np.array(ground_truths)
    predicted_results = np.array(predicted_results)
    mask_complete = ~np.isnan(predicted_results)
    ground_truths = ground_truths[mask_complete]
    predicted_results = predicted_results[mask_complete]
    return round(cohen_kappa_score(ground_truths, predicted_results, weights="quadratic"), 4)

def compute_kappa_summary_per_essay_set(truth_dict: defaultdict[int, list[int]], pred_dict: defaultdict[int, list[int]]) -> dict[str, float]:
    results: dict[str, float] = dict()
    avg_qwk = 0
    for essay_set in truth_dict:
        qwk = compute_kappa(truth_dict[essay_set], pred_dict[essay_set])
        results[str(essay_set)] = qwk
        avg_qwk += qwk
    avg_qwk /= N_CLASSES
    results["avg"] = avg_qwk
    return results

def compute_kappa_summary_per_domain(truth_dict: defaultdict[str, list[int]], pred_dict: defaultdict[str, list[int]]) -> dict[str, float]:
    assert len(truth_dict) == len(pred_dict) == 5
    results: dict[str, float] = dict()
    avg_qwk = 0
    for domain in truth_dict:
        qwk = compute_kappa(truth_dict[domain], pred_dict[domain])
        results[str(domain)] = qwk
        avg_qwk += qwk
    avg_qwk /= len(truth_dict)
    results["avg"] = avg_qwk
    return results


def inference_loop_baseline(
    run_folder: str,
    inference: UnifiedInference,
    few_shot_n: int = 0,
    limit: Optional[int] = None,
    add_argument_annotation: bool = False,
    adapters_repo: str = "",
    judge: bool = False,
) -> dict:
    from datasets import load_dataset
    import time
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    
    if add_argument_annotation:
        run_folder = run_folder + "/with-argument-annotation"
    else:
        run_folder = run_folder + "/no-argument-annotation"
    os.makedirs(run_folder, exist_ok=True)

    results: list[int] = []
    ground_truths: list[int] = []
    results_feedback: list[str] = []
    raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
    none_count = 0

    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")
    # train_dataset = load_dataset("jjordanoc/argumentative-asap", split="train")
    # train_df = pd.DataFrame(train_dataset)

    n = min(len(eval_dataset), limit) if limit is not None else len(eval_dataset)
    times = np.zeros((n + 1, 1), dtype=np.float32)

    for idx, grading_instruction in enumerate(eval_dataset):
        start_time = time.time()
        ground_truths.append(int(grading_instruction["score"]))

        full_prompt = GREGeneralGraderPrompts.format_prompt_inference(grading_instruction, add_argument_annotation=add_argument_annotation)
        
        # Use the unified inference interface
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
        print(out_str)
        raw_outputs.write(out_str)

        # Prompt processing
        score = try_extract_key("score", content, dtype=int)
        feedback = try_extract_key("feedback", content, dtype=str)
        results_feedback.append(feedback)
        if score is None:
            results.append(np.nan)
            none_count += 1
            continue
        results.append(score)
        
        times[idx] = (time.time() - start_time) * 1000  # Convert to milliseconds
        # Periodic writes
        if idx % 10 == 0:
            with open(os.path.join(run_folder, "tmp.json"), "w") as tmp_outs:
                output = {
                    "grader_prompts": GREGeneralGraderPrompts.dump_prompts(),
                    "qwk_grader": compute_kappa(ground_truths, results),
                    "predicted_labels": {
                        "scores": results,
                        "feedbacks": results_feedback,
                    },
                    "ground_truths": ground_truths,
                    "avg_time_ms": float(np.average(times)),
                    "sample_size": idx,
                    "none_count": none_count,
                }
                json.dump(output, tmp_outs)
                VOLUME_CONFIG["/runs"].commit()

    # Store data in a traceable format
    output = {
        "grader_prompts": GREGeneralGraderPrompts.dump_prompts(),
        "qwk_grader": compute_kappa(ground_truths, results),
        "predicted_labels": {
            "scores": results,
            "feedbacks": results_feedback,
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
    if judge:
        llm_as_judge.spawn(run_folder=run_folder, feedbacks=output["predicted_labels"]["feedbacks"])
    return output

def inference_loop_baseline_asap(
    run_folder: str,
    inference: UnifiedInference,
    few_shot_n: int = 0,
    limit: Optional[int] = None,
    add_argument_annotation: bool = False,
    adapters_repo: str = "",
) -> dict:
    from datasets import load_dataset
    import time
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    
    if add_argument_annotation:
        run_folder = run_folder + "/with-argument-annotation"
    else:
        run_folder = run_folder + "/no-argument-annotation"
    os.makedirs(run_folder, exist_ok=True)

    raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
    none_count = 0

    eval_dataset = load_dataset("jjordanoc/argumentative-asap-plus", split="train")
    n = min(len(eval_dataset), limit) if limit is not None else len(eval_dataset)
    times = np.zeros((n + 1, 1), dtype=np.float32)

    results_per_essay_set: defaultdict[int, list[int]] = defaultdict(list)
    ground_truths_per_essay_set: defaultdict[int, list[int]] = defaultdict(list)

    for idx, grading_instruction in enumerate(eval_dataset):
        start_time = time.time()
        essay_set = grading_instruction["essay_set"]
        ground_truths_per_essay_set[essay_set].append(int(grading_instruction["score"]))
        

        full_prompt = ASAPGeneralGraderPrompts.format_prompt_inference(grading_instruction, add_argument_annotation=add_argument_annotation)
        
        # Use the unified inference interface
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
        print(out_str)
        raw_outputs.write(out_str)

        # Prompt processing
        score = try_extract_key("score", content, dtype=int)
        if score is None:
            results_per_essay_set[essay_set].append(score)
            none_count += 1
            continue
        results_per_essay_set[essay_set].append(score)
        
        times[idx] = (time.time() - start_time) * 1000  # Convert to milliseconds
        # Periodic writes
        if idx % 10 == 0:
            with open(os.path.join(run_folder, "tmp.json"), "w") as tmp_outs:
                output = {
                    "grader_prompts": ASAPGeneralGraderPrompts.dump_prompts(),
                    "qwk_summary": compute_kappa_summary(ground_truths_per_essay_set, results_per_essay_set),
                    "predicted_labels": {
                        "scores": results_per_essay_set,
                    },
                    "ground_truths": ground_truths_per_essay_set,
                    "avg_time_ms": float(np.average(times)),
                    "sample_size": idx,
                    "none_count": none_count,
                }
                json.dump(output, tmp_outs)
                VOLUME_CONFIG["/runs"].commit()

    # Store data in a traceable format
    output = {
        "grader_prompts": ASAPGeneralGraderPrompts.dump_prompts(),
        "qwk_summary": compute_kappa_summary(ground_truths_per_essay_set, results_per_essay_set),
        "predicted_labels": {
            "scores": results_per_essay_set,
        },
        "ground_truths": ground_truths_per_essay_set,
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


def inference_loop_orchestration_asap(
    run_folder: str,
    inference: UnifiedInference,
    few_shot_n: int = 0,
    limit: Optional[int] = None,
    add_argument_annotation: bool = False,
    adapters_repo: str = "",
) -> dict:
    from datasets import load_dataset
    import time
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    import json

    if add_argument_annotation:
        run_folder = run_folder + "/with-argument-annotation"
    else:
        run_folder = run_folder + "/no-argument-annotation"
    os.makedirs(run_folder, exist_ok=True)

    orchestrated_results: list[int] = []
   
    ground_truths: list[int] = []
    raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
    none_count = 0

    eval_dataset = load_dataset("jjordanoc/argumentative-asap-plus", split="train")

    # Per essay set to compute orchestrated score
    results_per_essay_set: defaultdict[int, list[list]] = defaultdict(list)
    ground_truths_per_essay_set: defaultdict[int, list[int]] = defaultdict(list)

    # Per essay set per domain just in case
    results_per_essay_set_per_domain: defaultdict[int, list[list[int]]] = defaultdict(list)
    ground_truths_per_essay_set_per_domain: defaultdict[int, list[list[int]]] = defaultdict(list)

    # Per domain to compute domain scores
    results_per_domain: defaultdict[str, list[int]] = defaultdict(list)
    ground_truths_per_domain: defaultdict[str, list[int]] = defaultdict(list)

    averaged_scores_per_essay_set: defaultdict[int, list[int]] = defaultdict(list)

    n = min(len(eval_dataset), limit) if limit is not None else len(eval_dataset)
    times = np.zeros((n + 1, 1), dtype=np.float32)

    for idx, grading_instruction in enumerate(eval_dataset):
        start_time = time.time()
        # Has to be here to match length of orchestrated_results
        essay_set = grading_instruction["essay_set"]
        ground_truths_per_essay_set[essay_set].append(int(grading_instruction["score"]))

        trait_scores = grading_instruction["trait_scores"]
        ground_truths_current_essay_set = []
        for aspect in trait_scores:
            ground_truths_per_domain[aspect].append(trait_scores[aspect])
            ground_truths_current_essay_set.append(trait_scores[aspect])
        ground_truths_per_essay_set_per_domain[essay_set].append(ground_truths_current_essay_set)

        domain_scores_current_essay_set: list[int] = []
        domain_responses: list[str] = []
        
        for rubric_item, aspect_name in zip([1, 2, 3, 4, 5], ASAPAgentAlphaPrompts.aspect_names):
            full_prompt = ASAPAgentAlphaPrompts.format_prompt_inference(grading_instruction, rubric_item, add_argument_annotation=add_argument_annotation)
            
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
            if score is None:
                results_per_domain[aspect_name].append(np.nan)
                domain_scores_current_essay_set.append(np.nan)
                none_count += 1
                continue
            results_per_domain[aspect_name].append(score)
            domain_scores_current_essay_set.append(score)
            domain_responses.append(content)

        print(Colors.GREEN + Colors.BOLD + f"Scores per domain:  {domain_scores_current_essay_set}" + Colors.END)
        results_per_essay_set_per_domain[essay_set].append(domain_scores_current_essay_set)
        avg_domain_score = np.nan if np.isnan(np.nanmean(list(domain_scores_current_essay_set))) else round(np.nanmean(list(domain_scores_current_essay_set)))
        # Orchestration
        full_prompt = ASAPOrchestratorPrompts.format_prompt_inference(grading_instruction, domain_responses)
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
        averaged_scores_per_essay_set[essay_set].append(avg_domain_score)
        print(Colors.GREEN + Colors.BOLD + f"Averaged scores: {averaged_scores_per_essay_set}" + Colors.END)
        if score is None:
            orchestrated_results.append(np.nan)
            none_count += 1
            continue
        results_per_essay_set[essay_set].append(score)
        print(Colors.GREEN + Colors.BOLD + f"Orchestrated scores: {results_per_essay_set}" + Colors.END)

        times[idx] = (time.time() - start_time) * 1000  # Convert to milliseconds
        # Periodic writes
        if (idx+1) % 10 == 0:
            with open(os.path.join(run_folder, "tmp.json"), "w") as tmp_outs:
                output = {
                    "orchestrator_prompts": ASAPOrchestratorPrompts.dump_prompts(),
                    "agent_prompts": ASAPAgentAlphaPrompts.dump_prompts(),
                    "qwk_orchestrator_summary": compute_kappa_summary_per_essay_set(ground_truths_per_essay_set, results_per_essay_set),
                    "qwk_average_summary": compute_kappa_summary_per_essay_set(ground_truths_per_essay_set, averaged_scores_per_essay_set),
                    "qwk_per_domain_summary": compute_kappa_summary_per_domain(ground_truths_per_domain, results_per_domain),
                    "predicted_labels": {
                        "results_per_essay_set_per_domain": results_per_essay_set_per_domain,
                        "results_per_essay_set": results_per_essay_set,
                        "results_per_domain": results_per_domain,
                    },
                    "ground_truths": ground_truths_per_essay_set,
                    "avg_time_ms": float(np.average(times)),
                    "sample_size": idx,
                    "none_count": none_count,
                }
                json.dump(output, tmp_outs)
                VOLUME_CONFIG["/runs"].commit()

    # Store data in a traceable format
    output = {
        "orchestrator_prompts": ASAPOrchestratorPrompts.dump_prompts(),
        "agent_prompts": ASAPAgentAlphaPrompts.dump_prompts(),
        "qwk_orchestrator_summary": compute_kappa_summary_per_essay_set(ground_truths_per_essay_set, results_per_essay_set),
        "qwk_average_summary": compute_kappa_summary_per_essay_set(ground_truths_per_essay_set, averaged_scores_per_essay_set),
        "qwk_per_domain_summary": compute_kappa_summary_per_domain(ground_truths_per_domain, results_per_domain),
        "predicted_labels": {
            "results_per_essay_set_per_domain": results_per_essay_set_per_domain,
            "results_per_essay_set": results_per_essay_set,
            "results_per_domain": results_per_domain,
        },
        "ground_truths": ground_truths_per_essay_set,
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

def inference_loop_orchestration(
    run_folder: str,
    inference: UnifiedInference,
    few_shot_n: int = 0,
    limit: Optional[int] = None,
    add_argument_annotation: bool = False,
    adapters_repo: str = "",
    judge: bool = False,
) -> dict:
    from datasets import load_dataset
    import time
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score

    if add_argument_annotation:
        run_folder = run_folder + "/with-argument-annotation"
    else:
        run_folder = run_folder + "/no-argument-annotation"
    os.makedirs(run_folder, exist_ok=True)

    orchestrated_results: list[int] = []
    results_per_domain: list[list[int]] = []
    orchestrated_feedbacks: list[str] = []
    feedbacks_per_domain: list[list[str]] = []
    averaged_scores = []

    ground_truths: list[int] = []
    raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
    none_count = 0

    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")

    n = min(len(eval_dataset), limit) if limit is not None else len(eval_dataset)
    times = np.zeros((n + 1, 1), dtype=np.float32)

    for idx, grading_instruction in enumerate(eval_dataset):
        start_time = time.time()
        # Has to be here to match length of orchestrated_results
        ground_truths.append(int(grading_instruction["score"]))

        domain_scores = []
        domain_feedbacks = []
        domain_responses = []
        
        for rubric_item in [1, 2, 3, 4, 5]:
            full_prompt = GREAgentPrompts.format_prompt_inference(grading_instruction, rubric_item, add_argument_annotation=add_argument_annotation)
            
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
                    "qwk_orchestrator": compute_kappa(ground_truths, orchestrated_results),
                    "qwk_average": compute_kappa(ground_truths, averaged_scores),
                    "predicted_labels": {
                        "scores_per_domain": results_per_domain,
                        "feedbacks_per_domain": feedbacks_per_domain,
                        "scores": orchestrated_results,
                        "feedbacks": orchestrated_feedbacks,
                    },
                    "ground_truths": ground_truths,
                    "avg_time_ms": float(np.average(times)),
                    "sample_size": idx,
                    "none_count": none_count,
                }
                json.dump(output, tmp_outs)
                VOLUME_CONFIG["/runs"].commit()

    # Store data in a traceable format
    output = {
        "orchestrator_prompts": GREOrchestratorPrompts.dump_prompts(),
        "agent_prompts": GREAgentPrompts.dump_prompts(),
        "qwk_orchestrator": compute_kappa(ground_truths, orchestrated_results),
        "qwk_average": compute_kappa(ground_truths, averaged_scores),
        "predicted_labels": {
            "scores_per_domain": results_per_domain,
            "feedbacks_per_domain": feedbacks_per_domain,
            "scores": orchestrated_results,
            "feedbacks": orchestrated_feedbacks,
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
    if judge:
        llm_as_judge.spawn(run_folder=run_folder, feedbacks=output["predicted_labels"]["feedbacks"])
    return output

def linear_regression_analysis(run_folder: str, results_per_domain: list[list[int]], ground_truths: list[int]):
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
    cpu=2.0,
)
def llm_as_judge(run_folder: str, feedbacks: list[str] = [], judge_run: str = ""):
    """
    Use an LLM as a judge to score a set of essays.
    """
    from datasets import load_dataset
    from openai import OpenAI
    import pandas as pd
    os.makedirs(run_folder, exist_ok=True)
    if judge_run != "":
        with open(f"/runs/{judge_run}/run.json", "r") as f:
            output = json.load(f) 
            feedbacks=output["predicted_labels"]["feedbacks"]
    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")
    client = OpenAI()
    results = []
    for idx, grading_instruction in enumerate(eval_dataset):
        # Human will be 1, LLM will be 2
        prompt = RubricJudgePrompts.format_prompt_judge(feedback_1=grading_instruction["essay_feedback"], 
                                                  feedback_2=feedbacks[idx], 
                                                  student_essay=grading_instruction["essay_text"], 
                                                  task_directions=grading_instruction["task_directions"],
                                                  prompt=grading_instruction["prompt"]
                                                  )
        response = client.beta.chat.completions.parse(
            model="o4-mini",
            reasoning_effort="high",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format=RubricJudgePrompts.ResponseModel
        )
        decision = response.choices[0].message.content
        # decision = try_extract_key("feedback_choice", response.choices[0].message.content, dtype=int)
        # reasoning = try_extract_key("explanation", response.choices[0].message.content, dtype=str)
        # if decision == 1:
        #     print(Colors.RED + Colors.BOLD + f"Human is better" + Colors.END)
        #     print(Colors.RED + Colors.BOLD + f"Reasoning: {reasoning}" + Colors.END)
        #     human_wins += 1
        # elif decision == 2:
        #     print(Colors.GREEN + Colors.BOLD + f"LLM is better" + Colors.END)
        #     print(Colors.GREEN + Colors.BOLD + f"Reasoning: {reasoning}" + Colors.END)
        #     llm_wins += 1
        try:
            decision = json.loads(decision)
            results.append(decision)
            print(Colors.BOLD + f"Decision: {decision}" + Colors.END)
        except:
            print(Colors.RED + Colors.BOLD + f"Error parsing decision: {decision}" + Colors.END)

    # print(Colors.GREEN + Colors.BOLD + f"LLM wins: {llm_wins}, Human wins: {human_wins}" + Colors.END)
    output = {
        "results": results,
    }
    with open(os.path.join(run_folder, "judge_output.json"), "w") as f:
        json.dump(output, f)

    # Analysis
    df = pd.DataFrame(results)
    counts = (
        df
        .apply(pd.Series.value_counts)   # rows = values (1,2), cols = c1…c5
        .fillna(0)
        .astype(int)
        .T                                # now rows = c1…c5, cols = 1,2
    )

    # Compute win_rate = (# of 2s) / (total picks)
    counts['win_rate'] = counts[2] / (counts[1] + counts[2])
    counts = counts.rename(columns={1: "Human", 2: "LLM"})
    counts.to_csv(os.path.join(run_folder, "criteria_counts_with_win_rate.csv"))
    VOLUME_CONFIG["/runs"].commit()

    return results


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
    baseline: bool = False,
    orchestration: bool = False,
    judge: bool = False,
    run_folder: str = "",
    asap: bool = False,
):
    from datetime import datetime
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    
    os.makedirs(run_folder, exist_ok=True)
    baseline_folder = run_folder + "/baseline"
    orchestration_folder = run_folder + "/orchestration"
    os.makedirs(baseline_folder, exist_ok=True)
    os.makedirs(orchestration_folder, exist_ok=True)
    # Initialize the inference engine
    inference_engine = UnifiedInference(
        model_name=model_handle, adapters_repo=adapters_repo
    )
    if asap:
        # inference_loop_baseline_asap(
        #     run_folder=baseline_folder,
        #     inference=inference_engine,
        #     few_shot_n=few_shot_n,
        #     limit=limit,
        #     add_argument_annotation=False,
        #     adapters_repo=adapters_repo,
        # ) 
        inference_loop_orchestration_asap(
            run_folder=orchestration_folder,
            inference=inference_engine,
            few_shot_n=few_shot_n,
            limit=limit,
            add_argument_annotation=False,
            adapters_repo=adapters_repo,
        )

    if baseline:
        inference_loop_baseline(
            run_folder=baseline_folder,
            inference=inference_engine,
            few_shot_n=few_shot_n,
            limit=limit,
            add_argument_annotation=False,
            adapters_repo=adapters_repo,
            judge=judge,
        )   
        if add_argument_annotation:
            inference_loop_baseline(
                run_folder=baseline_folder,
                inference=inference_engine,
                few_shot_n=few_shot_n,
                limit=limit,
                add_argument_annotation=True,
                adapters_repo=adapters_repo,
                judge=judge,
            )
    if orchestration:
        inference_loop_orchestration(
            run_folder=orchestration_folder,
            inference=inference_engine,
            few_shot_n=few_shot_n,
            limit=limit,
            add_argument_annotation=False,
            adapters_repo=adapters_repo,
            judge=judge,
        )
        if add_argument_annotation:
            inference_loop_orchestration(
                run_folder=orchestration_folder,
                inference=inference_engine,
                few_shot_n=few_shot_n,
                limit=limit,
                add_argument_annotation=True,
                adapters_repo=adapters_repo,
                judge=judge,
            )
       
    VOLUME_CONFIG["/runs"].commit()


@inference_app.local_entrypoint()
def inference_main(
    model: str = "",
    shots: int = 0,
    limit: Optional[int] = None,
    arguments: bool = False,
    adapters_repo: str = "",
    baseline: bool = False,
    orchestration: bool = False,
    judge: bool = False,
    judge_run: str = "",
    asap: bool = False,
):
    from datetime import datetime
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
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"vllm-{model.replace(':', '-').replace('/', '-')}-{time_string}-{secrets.token_hex(2)}"
    run_folder = f"/runs/{run_name}"
    print(
        Colors.BLUE,
        Colors.BOLD,
        "https://modal.com/storage/ai-in-education-essay/main/example-runs-vol/"
        + run_name,
        Colors.END,
        sep="",
    )
    if judge and judge_run != "":
        judge_handle = llm_as_judge.spawn(run_folder=run_folder, judge_run=judge_run)
        judge_handle.get()
    else:
        # gre_models_small = ["google/gemma-3-12b-it"]
        gre_models_small = ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-12b-it", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "google/gemma-3-27b-it"]
        # gre_models_small = ["meta-llama/Llama-3.1-8B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
        if model != "":
            gre_models_small = [model]
        for gre_model in gre_models_small:
            inference_vllm.spawn(
                gre_model,
                run_folder=run_folder,
                few_shot_n=shots,   
                limit=limit,
                add_argument_annotation=arguments,
                adapters_repo=adapters_repo,    
                baseline=baseline,
                orchestration=orchestration,
                judge=judge,
                asap=asap,
            )