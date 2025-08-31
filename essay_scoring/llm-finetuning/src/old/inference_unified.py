import json
import logging
import os
import re
from collections import defaultdict
from typing import Optional, Literal

import modal
import numpy as np
from .common import VOLUME_CONFIG, MINUTES, ALLOW_WANDB, HOURS, vllm_image, Colors

# Constants and types
N_CLASSES = 6
LIMIT = np.inf
ASAPResults = dict[int, list[tuple[int, int]]]

# Inference configuration
INFERENCE_GPU_CONFIG = os.environ.get("INFERENCE_GPU_CONFIG", "a10g:2")
if len(INFERENCE_GPU_CONFIG.split(":")) <= 1:
    N_INFERENCE_GPUS = int(os.environ.get("N_INFERENCE_GPUS", 2))
    INFERENCE_GPU_CONFIG = f"{INFERENCE_GPU_CONFIG}:{N_INFERENCE_GPUS}"
else:
    N_INFERENCE_GPUS = int(INFERENCE_GPU_CONFIG.split(":")[-1])


# Helper functions
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
            logging.error("Error computing Kappa")
            logging.error(e)
    avg_qwk /= N_CLASSES
    results["avg"] = avg_qwk
    return results


def extract_domain_score(text: str, domain: int) -> Optional[int]:
    # Step 1: Find JSON objects in the string
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
            if f"domain_{domain}_score" in json_obj:
                return int(json_obj[f"{domain_score_key}"])

        except Exception as e:
            continue
    logging.warning(f"This text is none: {text}")
    return None


# Prompts
system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
Please provide a numerical score for the provided essay according to the specified rubric.

- The essay has been anonymized by replacing revealing details with tags that start with '@' and all letters are capitalized, such as '@ORGANIZATION1', '@CAPS2', '@DATE1', and etc. Do not penalize this. 
- Provide an appropriate holistic score for limited timed test conditions where there is little to no time for revision.
- You will carefully read the rubric (<rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will provide a detailed explanation to your decisions as to why you chose this score following the rubric and guidelines.
- Essay length matters. A good essay is generally comprised of at least 5 well-developed sentences.

The rubric or rubrics for this essay is as follows:
<rubric>
{rubric}
</rubric>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>
"""

essay_prompt = """
Review the given rubric and prompt carefully. The essay that requires a holistic score from the rubric is as follows:

<student_essay>
{essay_text}
</student_essay>

Provide a numerical domain_1_score by using the provided rubric's guidance.
Output the score in JSON using the following format:
{{
    "domain_1_score": {{essay_score}}
}}
"""

essay_set_2_essay_prompt = """
This essay requires 2 scores, and you have been provided with both rubrics in the system prompt.

Please provide a numerical score for each domain based on the appropriate rubric.
Domain 1: Writing Applications
Domain 2: Language Conventions

- Be sure to Review the given rubrics and prompt carefully, reasoning through your decision for each domain.

The essay that requires two scores from the rubric is as follows:

<student_essay>
{essay_text}
</student_essay>

Output the scores in JSON using the following format:
{{
    "domain_1_score": {{domain_score_1}},
    "domain_2_score": {{domain_score_2}}
}}
"""

# Modal app setup
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
    def __init__(self, backend: Literal["ollama", "vllm"], model_name: str = ""):
        self.backend = backend
        self.model_name = model_name
        self.engine = None
        print("here")
        if self.backend == "vllm":
            print("here2")
            with vllm_image.imports():
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine
                from vllm.sampling_params import SamplingParams
                from vllm.utils import random_uuid

            print(
                Colors.GREEN,
                Colors.BOLD,
                f"🧠: Initializing vLLM engine for model {self.model_name}",
                Colors.END,
                sep="",
            )

            engine_args = AsyncEngineArgs(
                model=self.model_name,
                gpu_memory_utilization=0.95,
                tensor_parallel_size=N_INFERENCE_GPUS,
                disable_custom_all_reduce=True,
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
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
                        logging.info("Ollama service is ready")
                        break
                except httpx.ConnectError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Ollama service failed to start")
                    logging.info(
                        f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
                    )
                    time.sleep(interval)

    # @modal.enter()
    # def init(self):

    async def generate(self, prompt: str) -> str:
        if self.backend == "vllm":
            from vllm.sampling_params import SamplingParams
            from vllm.utils import random_uuid

            sampling_params = SamplingParams(
                repetition_penalty=1.1,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                max_tokens=1024,
            )
            request_id = random_uuid()
            results_generator = self.engine.generate(
                prompt, sampling_params, request_id
            )

            full_response = ""
            async for request_output in results_generator:
                if (
                    request_output.outputs[0].text
                    and "\ufffd" != request_output.outputs[0].text[-1]
                ):
                    full_response += request_output.outputs[0].text
            return full_response
        else:  # ollama
            import httpx

            response = httpx.post(
                "http://localhost:11434/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
            )
            return response.json()["response"]

    # @modal.exit()
    # def cleanup(self):
    #     if self.backend == "vllm" and N_INFERENCE_GPUS > 1:
    #         import ray
    #         ray.shutdown()
    #         if hasattr(self.engine, '_background_loop_unshielded'):
    #             self.engine._background_loop_unshielded.cancel()


# Inference loop
async def inference_loop(
    run_folder: str,
    model_name: str,
    backend: Literal["ollama", "vllm"] = "ollama",
    remote_job: bool = True,
    local_dataset_path: str = "",
):
    from datasets import load_dataset
    import time

    results: ASAPResults = {}
    ground_truths: ASAPResults = {}
    raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
    tmp_outs = open(os.path.join(run_folder, "tmp.json"), "w")
    none_count = 0

    eval_dataset = load_dataset("jjordanoc/argumentative-asap", split="validation")
    n = min(len(eval_dataset), LIMIT)
    times = np.zeros((n + 1, 1), dtype=np.float32)

    if not remote_job:
        import pandas as pd

        df = pd.read_csv(local_dataset_path, sep="\t", encoding="ISO-8859-1", dtype=str)

    # Initialize the inference engine
    inference = UnifiedInference(backend=backend, model_name=model_name)

    for idx, grading_instruction in enumerate(eval_dataset):
        logging.info("*" * 120)
        logging.info("Processing essay", idx)
        logging.info("=" * 80)
        logging.info("Prompt:")
        logging.info(grading_instruction)

        essay_set = int(grading_instruction["essay_set"])

        content: Optional[str] = None
        if remote_job:
            system_prompt_formatted = system_prompt.format(
                rubric=grading_instruction["rubric"],
                prompt=grading_instruction["essay_prompt"],
            )

            essay_set_prompt_formatted = (
                essay_set_2_essay_prompt.format(
                    essay_text=grading_instruction["essay_text"]
                )
                if essay_set == 2
                else essay_prompt.format(essay_text=grading_instruction["essay_text"])
            )

            full_prompt = system_prompt_formatted + "\n\n" + essay_set_prompt_formatted

            # Use the unified inference interface
            start_time = time.time()
            content = await inference.generate(full_prompt)
            times[idx] = (time.time() - start_time) * 1000  # Convert to milliseconds
        else:
            content = (
                df[df["essay_id"] == (grading_instruction["essay_id"])]["comments"]
            ).values[0]

        out_str = "=" * 30 + f"Interaction {idx}" + "=" * 30 + content + "\n\n"
        raw_outputs.write(out_str)
        logging.info("=" * 80)
        logging.info("Answer:")

        try:
            score_1 = extract_domain_score(content, 1)
            score_2 = -1
            logging.info(score_1)

            if essay_set == 2:
                score_2 = extract_domain_score(content, 2)
                logging.info(score_2)
            score = (score_1, score_2)
        except Exception as e:
            # skip this essay
            logging.error(e)
            continue

        # Discard from analysis
        if score_1 is None or score_2 is None:
            none_count += 1
            continue

        grader_score_1 = -1
        grader_score_2 = -1
        if essay_set == 2:
            split = grading_instruction["grader_score"].split(" ")
            grader_score_1 = int(split[0])
            grader_score_2 = int(split[1])
        else:
            grader_score_1 = int(grading_instruction["grader_score"])
        grader_score = (grader_score_1, grader_score_2)

        if essay_set not in results:
            results[essay_set] = [score]
            ground_truths[essay_set] = [grader_score]
        else:
            results[essay_set].append(score)
            ground_truths[essay_set].append(grader_score)

        logging.info("*" * 120)

        # Periodic writes
        if idx % 100 == 0 and remote_job:
            output = {
                "system_prompt": system_prompt,
                "essay_prompt": essay_prompt,
                "essay_prompt_set_2": essay_set_2_essay_prompt,
                "qwk_summary": compute_kappa_summary(ground_truths, results),
                "predicted_labels": results,
                "ground_truths": ground_truths,
                "avg_time_ms": float(np.average(times) / (10**6)),
                "sample_size": idx,
                "none_count": none_count,
            }
            json.dump(output, tmp_outs)
            VOLUME_CONFIG["/runs"].commit()

        if idx == n:
            break

    # Store data in a traceable format
    output = {
        "system_prompt": system_prompt,
        "essay_prompt": essay_prompt,
        "essay_prompt_set_2": essay_set_2_essay_prompt,
        "qwk_summary": compute_kappa_summary(ground_truths, results),
        "predicted_labels": results,
        "ground_truths": ground_truths,
        "avg_time_ms": float(np.average(times) / (10**6)),
        "sample_size": n,
        "none_count": none_count,
    }
    outfile = open(os.path.join(run_folder, "run.json"), "w")
    json.dump(output, outfile)
    outfile.close()
    raw_outputs.close()
    tmp_outs.close()


# Modal function for inference job
@inference_app.function(
    image=(
        vllm_image
        if os.environ.get("INFERENCE_BACKEND", "ollama") == "vllm"
        else ollama_image
    ),
    timeout=24 * HOURS,
    volumes=VOLUME_CONFIG,
    gpu=INFERENCE_GPU_CONFIG,
)
def inference_job(model_name: str, backend: str = "ollama"):
    import asyncio

    asyncio.run(inference_loop("", model_name=model_name, backend=backend))


# Entry points
@inference_app.local_entrypoint()
def inference_main(model: str, backend: str = "ollama"):
    import asyncio

    asyncio.run(inference_loop("", model_name=model, backend=backend))


def local_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to use for inference"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "vllm"],
        help="Backend to use for inference",
    )
    args = parser.parse_args()

    import asyncio

    asyncio.run(inference_loop("", model_name=args.model, backend=args.backend))


if __name__ == "__main__":
    local_main()
