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

INFERENCE_GPU_CONFIG = "A100-80GB:1"

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
        

def compute_kappa(ground_truths: list[int], predicted_results: list[int]) -> float:
    from sklearn.metrics import cohen_kappa_score
    # Convert None to np.nan
    ground_truths = [np.nan if x is None else x for x in ground_truths]
    predicted_results = [np.nan if x is None else x for x in predicted_results]
    # Drop na's
    ground_truths = np.array(ground_truths)
    predicted_results = np.array(predicted_results)
    mask_complete = ~np.isnan(predicted_results)
    mask_complete_ground_truths = ~np.isnan(ground_truths)  # ignore missing ground truths
    print( Colors.RED + Colors.BOLD + "{} incomplete ground truths, {} incomplete predictions".format(np.sum(~mask_complete_ground_truths), np.sum(~mask_complete)) + Colors.END)
    ground_truths = ground_truths[mask_complete & mask_complete_ground_truths]
    predicted_results = predicted_results[mask_complete & mask_complete_ground_truths]
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
    import csv
    
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

    # Write to a csv file
    csv_file = open(os.path.join(run_folder, "baseline_results.tsv"), "w")
    csv_writer = csv.writer(csv_file, delimiter="\t")
    # csv will have:
    # essay_id (idx), score, human_score, feedback, human_feedback, raw_response
    csv_writer.writerow(["essay_id", "score", "human_score", "feedback", "human_feedback", "raw_response"])



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

        csv_writer.writerow([idx, score, grading_instruction["score"], feedback, grading_instruction["essay_feedback"], content])
        
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
        if limit is not None and idx >= limit:
            break

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
    csv_file.close()
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
    # few_shot_examples = load_dataset("jjordanoc/argumentative-asap-plus", split="shots")

    # test_dataset = load_dataset("jjordanoc/argumentative-asap", split="test")

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
        # TODO: Remove, only held out data
        # if grading_instruction["essay_id"] not in test_dataset["essay_id"]:
        #     continue
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
            full_prompt = ASAPAgentAlphaPrompts.format_prompt_inference(grading_instruction, rubric_item, add_argument_annotation=add_argument_annotation, calibration_examples=few_shot_examples)

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
            print(out_str)
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
        print(out_str)
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
    ablations: list[int] = [1,2,3,4,5],
) -> dict:
    from datasets import load_dataset
    import time
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    import csv
    if ablations == [1,2,3,4,5]:
        ablation_name = "ablation_all"
    else:
        ablation_name = "ablation_" + "_".join([f"domain_{ablation}" for ablation in ablations])
    run_folder = run_folder + f"/{ablation_name}"
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

    # Write to a csv file
    csv_file = open(os.path.join(run_folder, "ablation_results.tsv"), "w")
    csv_writer = csv.writer(csv_file, delimiter="\t")
    # csv will have:
    # essay_id (idx), orchestrated_score, human_score, orchestrated_feedback, human_feedback, domain_1_score,..., domain_5_score, human_domain_1_score,..., human_domain_5_score, domain_1_feedback,..., domain_5_feedback, raw_response, avg_score
    domain_scores_names = [f"domain_{i}_score" for i in range(1, 6)]
    domain_feedbacks_names = [f"domain_{i}_feedback" for i in range(1, 6)]
    human_scores_names = [f"human_domain_{i}_score" for i in range(1, 6)]
    csv_writer.writerow(["essay_id", "orchestrated_score", "human_score", "orchestrated_feedback", "human_feedback", *domain_scores_names, *human_scores_names, *domain_feedbacks_names, "raw_response", "avg_score"])


    for idx, grading_instruction in enumerate(eval_dataset):
        start_time = time.time()
        # Has to be here to match length of orchestrated_results
        ground_truths.append(int(grading_instruction["score"]))

        domain_scores = []
        domain_feedbacks = []
        domain_responses = []
        
        for rubric_item in [1, 2, 3, 4, 5]:
            # Perform ablation
            if rubric_item not in ablations:
                domain_scores.append(np.nan)
                domain_feedbacks.append("")
                continue
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
                domain_feedbacks.append("")
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
        # essay_id (idx), orchestrated_score, human_score, orchestrated_feedback, human_feedback, domain_1_score,..., domain_5_score, human_domain_1_score,..., human_domain_5_score, domain_1_feedback,..., domain_5_feedback, raw_response, avg_score
        human_domain_scores = [grading_instruction["aspect_1"], grading_instruction["aspect_2"], grading_instruction["aspect_3"], grading_instruction["aspect_4"], grading_instruction["aspect_5"]]
        csv_writer.writerow([idx, score, grading_instruction["score"], feedback, grading_instruction["essay_feedback"], *domain_scores, *human_domain_scores, *domain_feedbacks, content, avg_domain_score])

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
        if limit is not None and idx >= limit:
            break

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
    csv_file.close()
    VOLUME_CONFIG["/runs"].commit()
    if judge:
        llm_as_judge.spawn(run_folder=run_folder, feedbacks=output["predicted_labels"]["feedbacks"])
    return output



@inference_app.function(
    image=vllm_image,
    timeout=24 * HOURS,
    volumes=VOLUME_CONFIG,
    cpu=2.0,
)

def llm_as_judge(run_folder: str):
    """
    Use an LLM as a judge to score a set of essays.
    run_folder: folder where results were saved
    """
    from datasets import load_dataset
    from openai import OpenAI
    import pandas as pd
    os.makedirs(run_folder, exist_ok=True)
    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")
    for model_path,_, filenames in os.walk(run_folder):
        if "run.json" in filenames:
            with open(os.path.join(model_path, "run.json"), "r") as f:
                output = json.load(f) 
                feedbacks=output["predicted_labels"]["feedbacks"]
                llm_as_judge_helper("Human", "LLM", eval_dataset["essay_feedback"], feedbacks, model_path)

def llm_as_judge_helper(tag1: str, tag2: str, feedbacks1: list[str], feedbacks2: list[str], run_folder: str, fixed: bool = False):
    """
    Use an LLM as a judge to score a set of essays.
    """
    from datasets import load_dataset
    from openai import OpenAI
    import pandas as pd
    run_folder = f"{run_folder}/{tag1}_vs_{tag2}"
    os.makedirs(run_folder, exist_ok=True)
   
    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")
    client = OpenAI()
    results = []
    total_tag1_wins = 0
    total_tag2_wins = 0
    none_count1 = 0
    none_count2 = 0
    assert len(feedbacks1) == len(feedbacks2)
    reasoning_outputs = []
    summary_outputs = []
    for idx, grading_instruction in enumerate(eval_dataset):
        if feedbacks1[idx] is None:
            none_count1 += 1
            continue
        if feedbacks2[idx] is None:
            none_count2 += 1
            continue
        # Human will be 1, LLM will be 2
        prompt = RubricJudgePrompts.format_prompt_judge(feedback_1=feedbacks1[idx], 
                                                  feedback_2=feedbacks2[idx], 
                                                  student_essay=grading_instruction["essay_text"], 
                                                  task_directions=grading_instruction["task_directions"],
                                                  prompt=grading_instruction["prompt"]
                                                  )
        print(Colors.BLUE + f"Essay 1: {feedbacks1[idx]}" + Colors.END)
        print(Colors.RED + f"Essay 2: {feedbacks2[idx]}" + Colors.END)
        response = client.responses.parse(
            model="o4-mini",
            input=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            text_format=RubricJudgePrompts.ResponseModel,
            reasoning={
                "effort" : "medium",
                "summary" : "auto"
            }
        )
        print(Colors.BOLD + f"Response: {response}" + Colors.END)
        decision = response.output_parsed
        try:
            reasoning_outputs.append(response.output[0].summary[0].text)
        except:
            reasoning_outputs.append("")
        tag1_wins = 0
        tag2_wins = 0
        print(Colors.GREEN, reasoning_outputs, Colors.END)
        summary_outputs.append(decision.summary)
        try:
            # We do winner takes all
            decision = dict(decision)
            del decision["summary"]
            for i in range(1, 6):
                if decision[f"c{i}"] == 1:
                    tag1_wins += 1
                else:
                    tag2_wins += 1
            # Works because it's odd
            if tag1_wins > tag2_wins:
                total_tag1_wins += 1
            elif tag2_wins > tag1_wins:
                total_tag2_wins += 1
            results.append(decision)
            print(Colors.BOLD + f"Decision: {decision}" + Colors.END)
        except:
            print(Colors.RED + Colors.BOLD + f"Error parsing decision: {decision}" + Colors.END)

    print(Colors.GREEN + Colors.BOLD + f"{tag1} wins: {total_tag1_wins}, {tag2} wins: {total_tag2_wins}" + Colors.END)

    win_rate_tag1 = total_tag1_wins / (total_tag1_wins + total_tag2_wins)
    win_rate_tag2 = total_tag2_wins / (total_tag1_wins + total_tag2_wins)

    output = {
        "results": results,
        "none_count1": none_count1,
        "none_count2": none_count2,
        "win_rate_tag1": win_rate_tag1,
        "win_rate_tag2": win_rate_tag2,
        "reasoning_outputs": reasoning_outputs,
        "summary_outputs": summary_outputs
    }
    with open(os.path.join(run_folder, f"judge_output.json"), "w") as f:
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
    counts['win_rate'] = round(counts[2] / (counts[1] + counts[2]), 3)
    counts = counts.rename(columns={1: tag1, 2: tag2})
    counts.to_csv(os.path.join(run_folder, f"criteria_counts_with_win_rate.csv"))
    VOLUME_CONFIG["/runs"].commit()


    return results, df, win_rate_tag1, win_rate_tag2, counts


@inference_app.function(
    image=vllm_image,
    timeout=24 * HOURS,
    volumes=VOLUME_CONFIG,
    cpu=1.0,
)
def gre_qwk_analysis(run_folder: str):
    from datasets import load_dataset
    os.makedirs(run_folder, exist_ok=True)
    csv_file = f"{run_folder}/gre_analysis.csv"
    domain_names = ",".join(str(i) for i in range(1, 6))
    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")
    with open(csv_file, "w") as f:
        f.write(f"model,{domain_names}\n")
    for model_path,_, filenames in os.walk(run_folder):
        if "run.json" in filenames:
            with open(os.path.join(model_path, "run.json"), "r") as f:
                output = json.load(f)
                try:
                    scores_per_domain_list = output["predicted_labels"]["scores_per_domain"]
                except:
                    continue
                # ground_truths = output["ground_truths"]
                # qwk_matrix will have rows = domains, cols = scores per domain
                # initialize with 5 empty lists
                qwk_matrix = [[] for _ in range(5)]
                for domain_scores in scores_per_domain_list:
                    for i in range(5):
                        qwk_matrix[i].append(domain_scores[i])
                print(Colors.GREEN + Colors.BOLD + f"QWK Matrix: {qwk_matrix}" + Colors.END)
                domain_qwks = []
                for domain_idx in range(len(qwk_matrix)):
                    ground_truths = eval_dataset["aspect_" + str(domain_idx + 1)]
                    domain_qwk = compute_kappa(ground_truths, qwk_matrix[domain_idx])
                    domain_qwks.append(domain_qwk)
                domain_qwks_str = ",".join([str(round(qwk, 3)) for qwk in domain_qwks])   
                print(Colors.BLUE + Colors.BOLD + f"Domain QWK: {domain_qwks_str}" + Colors.END)
                with open(csv_file, "a") as f:
                    full_model_alias = "-".join(model_path.split("/")[1:])
                    f.write(f"{full_model_alias},{domain_qwks_str}\n")
                

@inference_app.function(
    image=vllm_image,
    timeout=24 * HOURS,
    volumes=VOLUME_CONFIG,
    cpu=2.0,
)
def llm_as_judge_matrix(run_folder: str):
    """
    Use an LLM as a judge to score a set of essays.
    """
    from datasets import load_dataset
    from openai import OpenAI
    import pandas as pd
    import numpy as np

    os.makedirs(run_folder, exist_ok=True)

    matrix_run_folder = f"{run_folder}/matrix-fix"
    os.makedirs(matrix_run_folder, exist_ok=True)

    eval_dataset = load_dataset("jjordanoc/gre-scoring-dataset", split="train")

    eval_list = [
        ("human", eval_dataset["essay_feedback"])
    ]

    # Ablations
    for model_path,_, filenames in os.walk(run_folder):
        if "run.json" in filenames:
            with open(os.path.join(model_path, "run.json"), "r") as f:
                output = json.load(f) 
                feedbacks=output["predicted_labels"]["feedbacks"]
                model_alias = "-".join(model_path.split("/")[1:])
                eval_list.append((model_alias, feedbacks))
    labels = [alias for alias, _ in eval_list]
    print(Colors.GREEN + Colors.BOLD + f"Recovered feedback from labels: {labels}" + Colors.END)

    # Results
    average_matrix = np.full((len(eval_list), len(eval_list)), 0.5)
    majority_matrix = np.full((len(eval_list), len(eval_list)), 0.5)
    for idx_1, (alias_1, feedback_1) in enumerate(eval_list):
        for idx_2, (alias_2, feedback_2) in enumerate(eval_list):
            # Triangular matrix
            if idx_1 <= idx_2:
                continue
            _, _, majority_win_rate_tag1, majority_win_rate_tag2, win_rate_counts_per_tag = llm_as_judge_helper(alias_1, alias_2, feedback_1, feedback_2, matrix_run_folder)
            # helper returns winrate for tag2 over tag1
            # for tag 1 is 1-winrate
            win_rate_tag2_per_criteria = win_rate_counts_per_tag["win_rate"]
            win_rate_tag1_per_criteria = 1 - win_rate_tag2_per_criteria
            # use average winrate per criteria
            win_rate_tag2 = np.average(win_rate_tag2_per_criteria)
            win_rate_tag1 = np.average(win_rate_tag1_per_criteria)
            # (i,j) is the win-rate of i over j
            average_matrix[idx_1, idx_2] = win_rate_tag1
            average_matrix[idx_2, idx_1] = win_rate_tag2
            majority_matrix[idx_1, idx_2] = majority_win_rate_tag1
            majority_matrix[idx_2, idx_1] = majority_win_rate_tag2
            print(Colors.BOLD, "Average win rate: ", win_rate_counts_per_tag, win_rate_tag1, win_rate_tag2, Colors.END)
    print(Colors.BLUE + Colors.BOLD + f"Average Matrix: {average_matrix}" + Colors.END)
    print(Colors.BLUE + Colors.BOLD + f"Majority Matrix: {majority_matrix}" + Colors.END)

    output = {
        "labels": labels,
        "average_matrix": average_matrix.tolist(),
        "majority_matrix": majority_matrix.tolist(),
    }
    with open(os.path.join(matrix_run_folder, "judge_matrix.json"), "w") as f:
        json.dump(output, f)
    VOLUME_CONFIG["/runs"].commit()
    return output

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
    adapters_repo: str = "",
    baseline: bool = False,
    orchestration: bool = False,
    judge: bool = False,
    run_folder: str = "",
    asap: bool = False,
    ablations: Optional[list[int]] = None,
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
    # if asap and baseline:
    #     inference_loop_baseline_asap(
    #         run_folder=baseline_folder,
    #         inference=inference_engine,
    #         few_shot_n=few_shot_n,
    #         limit=limit,
    #         add_argument_annotation=False,
    #         adapters_repo=adapters_repo,
    #     ) 
    # if asap and orchestration:
    #     inference_loop_orchestration_asap(
    #         run_folder=orchestration_folder,
    #         inference=inference_engine,
    #         few_shot_n=few_shot_n,
    #         limit=limit,
    #         add_argument_annotation=False,
    #         adapters_repo=adapters_repo,
    #     )

    if baseline and ablations is None:
        inference_loop_baseline(
            run_folder=baseline_folder,
            inference=inference_engine,
            few_shot_n=few_shot_n,
            limit=limit,
            add_argument_annotation=False,
            adapters_repo=adapters_repo,
            judge=judge
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
            ablations=ablations
        )

    VOLUME_CONFIG["/runs"].commit()


@inference_app.local_entrypoint()
def inference_main(
    model: str = "",
    shots: int = 0,
    limit: Optional[int] = None,
    adapters_repo: str = "",
    baseline: bool = False,
    orchestration: bool = False,
    judge: bool = False,
    judge_run: str = "",
    asap: bool = False,
    judge_matrix: bool = False,
    gre_analysis: bool = False,
    final_ablation_agents: int = 0,
):
    from datetime import datetime
    from itertools import combinations
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
    # run_name = f"vllm-{model.replace(':', '-').replace('/', '-')}-{time_string}-{secrets.token_hex(2)}"
    run_name = time_string
    run_folder = f"/runs/{run_name}"
    print(
        Colors.BLUE,
        Colors.BOLD,
        "https://modal.com/storage/ai-in-education-essay/main/example-runs-vol/"
        + run_name,
        Colors.END,
        sep="",
    )
    if gre_analysis and judge_run != "":
        gre_analysis_handle = gre_qwk_analysis.spawn(run_folder="/runs/"+judge_run)
        gre_analysis_handle.get()
    elif judge and judge_run != "":
        judge_handle = llm_as_judge.spawn(run_folder="/runs/"+judge_run)
        judge_handle.get()
    elif judge_matrix and judge_run != "":
        judge_matrix_handle = llm_as_judge_matrix.spawn(run_folder="/runs/" + judge_run)
        judge_matrix_handle.get()
    elif final_ablation_agents > 0:
        gre_model = "google/gemma-3-12b-it"
        # 2 agents
        for combination in combinations(range(1,6), final_ablation_agents):
            inference_vllm.spawn(
                gre_model,
                run_folder=run_folder + f"/{gre_model.replace('/', '-').replace(':', '-')}",
                few_shot_n=shots,   
                limit=limit,
                adapters_repo=adapters_repo,    
                baseline=baseline,
                orchestration=orchestration,
                judge=judge,
                asap=asap,
                ablations=list(combination)
            )
    else:
        # gre_models_small = ["google/gemma-3-12b-it"]
        gre_models_small = ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-12b-it", "google/gemma-3-27b-it", "meta-llama/Llama-3.3-70B-Instruct"]
        # gre_models_small = ["meta-llama/Llama-3.1-8B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
        if model != "":
            gre_models_small = [model]
        for gre_model in gre_models_small:
            inference_vllm.spawn(
                gre_model,
                run_folder=run_folder + f"/{gre_model.replace('/', '-').replace(':', '-')}",
                few_shot_n=shots,   
                limit=limit,
                adapters_repo=adapters_repo,    
                baseline=baseline,
                orchestration=orchestration,
                judge=judge,
                asap=asap,
            )