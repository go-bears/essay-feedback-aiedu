import json
import os
import re
import re
import secrets
from collections import defaultdict
from typing import Optional, Literal

import modal
import numpy as np
from .common import VOLUME_CONFIG, MINUTES, ALLOW_WANDB, HOURS, Colors

INFERENCE_GPU_CONFIG = "A100:2"

if len(INFERENCE_GPU_CONFIG.split(":")) <= 1:
    N_INFERENCE_GPUS = int(os.environ.get("N_INFERENCE_GPUS", 2))
    INFERENCE_GPU_CONFIG = f"{INFERENCE_GPU_CONFIG}:{N_INFERENCE_GPUS}"
else:
    N_INFERENCE_GPUS = int(INFERENCE_GPU_CONFIG.split(":")[-1])

N_CLASSES = 6

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
        # "transformers==4.50.3",
        "modal",
        "huggingface_hub[hf_transfer]==0.30.1",
        # "fastapi==0.110.0",
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
    def __init__(self, backend: Literal["ollama", "vllm"], model_name: str = ""):
        self.backend = backend
        self.model_name = model_name
        self.engine = None
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
                tensor_parallel_size=N_INFERENCE_GPUS,
                pipeline_parallel_size=1,
                gpu_memory_utilization=0.98,
                block_size=128,
                cpu_offload_gb=0,
                max_model_len=128000,
                max_num_seqs=8,
                disable_custom_all_reduce=False,
                enable_chunked_prefill=True,
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

            sampling_params = SamplingParams(
                repetition_penalty=1.1,
                temperature=0.2,
                top_p=0.95,
                top_k=50,
                max_tokens=1024,
            )
            request_id = random_uuid()
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

    # @modal.exit()
    # def cleanup(self):
    #     if self.backend == "vllm" and N_INFERENCE_GPUS > 1:
    #         import ray
    #         ray.shutdown()
    #         if hasattr(self.engine, '_background_loop_unshielded'):
    #             self.engine._background_loop_unshielded.cancel()


# @inference_app.function(image=ollama_image, timeout=60 * MINUTES, volumes=VOLUME_CONFIG)
# def setup_job(model_handle: str):
#
#     inference_handle = inference_job.spawn(model_handle, run_folder)
#     return inference_handle




def get_examples(train_df, essay_set: int, num: int) -> str:
    few_shot_examples = "Here are some examples of essays and their scores:\n"

    for idx, row in enumerate(train_df[train_df["essay_set"] == str(essay_set)].itertuples()):
        # print(idx, row, num)
        if idx >= num:
            break
        few_shot_examples += f"""
<example_essay>
{row.essay_text}
</example_essay>
        """
        if int(row.essay_set) == 2:
            few_shot_examples += f"""
<example_output>
{{
    "domain_1_score": {row.grader_score.split(" ")[0]},
    "domain_2_score": {row.grader_score.split(" ")[1]}
}}
</example_output>
            """
        else:
            few_shot_examples += f"""
<example_output>
{{
    "domain_1_score": {row.grader_score}
}}
</example_output>
            """
    return few_shot_examples

def inference_loop(
    run_folder: str,
    model_name: str,
    backend: Literal["ollama", "vllm"] = "ollama",
    remote_job: bool = True,
    local_dataset_path: str = "",
    few_shot_n: int = 0,
    limit: Optional[int] = None,
    add_argument_annotation: bool = False,
):
    from datasets import load_dataset
    import time
    import pandas as pd

    results: ASAPResults = {}
    ground_truths: ASAPResults = {}
    raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
    none_count = 0

    eval_dataset = load_dataset("jjordanoc/argumentative-asap", split="validation")
    
    train_dataset = load_dataset("jjordanoc/argumentative-asap", split="train")
    train_df = pd.DataFrame(train_dataset)

    n = min(len(eval_dataset), limit) if limit is not None else len(eval_dataset)
    times = np.zeros((n + 1, 1), dtype=np.float32)

    if not remote_job:
        import pandas as pd

        df = pd.read_csv(local_dataset_path, sep="\t", encoding="ISO-8859-1", dtype=str)

    # Initialize the inference engine
    inference = UnifiedInference(backend=backend, model_name=model_name)

    for idx, grading_instruction in enumerate(eval_dataset):
        print("*" * 120)
        print("Processing essay", idx)
        print("=" * 80)
        # print("Prompt:")
        # print(grading_instruction)

        essay_set = int(grading_instruction["essay_set"])
        essay_id = grading_instruction["essay_id"]

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

            full_prompt = system_prompt_formatted + "\n\n"
            if few_shot_n > 0:
                full_prompt += get_examples(train_df, essay_set=essay_set, num=few_shot_n) + "\n\n"
            full_prompt += essay_set_prompt_formatted 
            if add_argument_annotation:
                full_prompt += "Here is an annotation of the essay's argument components to assist you in scoring:\n" + "<student_essay_argument_annotation>\n" + grading_instruction["argument_annotation"] + "\n</student_essay_argument_annotation>\n\n"
            full_prompt += "<output>" + "\n" + "<explanation>"

            # Use the unified inference interface
            start_time = time.time()
            content = inference.generate(full_prompt)
            times[idx] = (time.time() - start_time) * 1000  # Convert to milliseconds
        else:
            full_prompt = ""
            content = (
                df[df["essay_id"] == (grading_instruction["essay_id"])]["comments"]
            ).values[0]

        out_str = "=" * 30 + f"Interaction {idx}" + "=" * 30 + "\nPrompt:\n" + full_prompt + "\n\n" + "Response:\n" + content + "\n\n"
        raw_outputs.write(out_str)
        print("=" * 80)
        print("Answer:")

        try:
            score_1 = extract_domain_score(content, 1)
            score_2 = -1
            print(score_1)

            if essay_set == 2:
                score_2 = extract_domain_score(content, 2)
                print(score_2)
            score = (score_1, score_2)
        except Exception as e:
            # skip this essay
            print(e)
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

        print("*" * 120)

        # Periodic writes
        if idx % 100 == 0 and remote_job:
            with open(os.path.join(run_folder, "tmp.json"), "w") as tmp_outs:
                output = {
                    "system_prompt": system_prompt,
                    "essay_prompt": essay_prompt,
                    "few_shot_n": few_shot_n,
                    "add_argument_annotation": add_argument_annotation,
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
        "few_shot_n": few_shot_n,
        "add_argument_annotation": add_argument_annotation,
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


@inference_app.function(
    image=vllm_image,
    timeout=24 * HOURS,
    volumes=VOLUME_CONFIG,
    gpu=INFERENCE_GPU_CONFIG,
)
def inference_vllm(model_handle: str, few_shot_n: int = 0, limit: Optional[int] = None, add_argument_annotation: bool = False):
    from datetime import datetime

    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = (
        f"vllm-{model_handle.replace(':', '-').replace('/', '-')}-{time_string}-{secrets.token_hex(2)}"
    )
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)
    print( Colors.BLUE, Colors.BOLD, "https://modal.com/storage/ai-in-education-essay/main/example-runs-vol/" + run_name, Colors.END, sep="", )
    # VOLUME_CONFIG["/runs"].reload()
    inference_loop(run_folder, model_name=model_handle, backend="vllm", few_shot_n=few_shot_n, limit=limit, add_argument_annotation=add_argument_annotation)
    VOLUME_CONFIG["/runs"].commit()
    print( Colors.GREEN, Colors.BOLD, "https://modal.com/storage/ai-in-education-essay/main/example-runs-vol/" + run_name, Colors.END, sep="", )


"""
Run using vllm handle:
    modal run --detach -m src.inference --model=google/gemma-3-12b-it --backend=vllm --shots=1 --arguments 

Deepseek With arguments:
    modal run --detach -m src.inference --model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B --backend=vllm --shots=1 --arguments
    
Run using ollama handle:x
    modal run --detach -m src.inference-ollama --model=gemma3:12b --backend=ollama
"""


@inference_app.local_entrypoint()
def inference_main(model: str, backend: str, shots: int = 0, limit: Optional[int] = None, arguments: bool = False):
    if backend == "ollama":
        # handle = inference_ollama.spawn(model)
        pass
    else:
        handle = inference_vllm.spawn(model, few_shot_n=shots, limit=limit, add_argument_annotation=arguments)
    handle.get()


def local_main():
    run_folder = "../local_runs"
    inference_loop(
        run_folder,
        remote_job=False,
        local_dataset_path="/Users/joaquin/Desktop/ai_education/essay-feedback-aiedu/essay_scoring/final_llama3.2-scoring-output-2025-04-09-12-19.tsv",
    )


if __name__ == "__main__":
    local_main()
