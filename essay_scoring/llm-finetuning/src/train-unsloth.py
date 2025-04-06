import os
from datetime import datetime
from pathlib import Path
import secrets

from .common import (
    app,
    axolotl_image,
    HOURS,
    MINUTES,
    VOLUME_CONFIG,
)

GPU_CONFIG = "A100-80GB:1"
SINGLE_GPU_CONFIG = os.environ.get("GPU_CONFIG", "a10g:1")


def model_setup(model_handle: str):
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model_name = model_handle.split("/")[-1]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_handle,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer


@app.function( 
    image=axolotl_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def train_unsloth(run_folder: str, model_handle: str):
    model, tokenizer = model_setup()
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    from datasets import load_dataset
    dataset = load_dataset("jjordanoc/argumentative-asap", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 1,
            warmup_steps = 5,
            # num_train_epochs = 4, # Set this for 1 full training run.
            max_steps = 30,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )
    trainer.train()

     # Ensure volumes contain latest files.
    VOLUME_CONFIG["/pretrained"].reload()
    VOLUME_CONFIG["/runs"].reload()

    trainer.save_model(os.path.join(run_folder,model_name))

    # Commit writes to volume.
    VOLUME_CONFIG["/runs"].commit()



@app.function(image=axolotl_image, timeout=30 * MINUTES, volumes=VOLUME_CONFIG)
def launch_unsloth(model_handle: str):
    import yaml
    from huggingface_hub import snapshot_download

    model_name = model_handle.split("/")[-1]

    try:
        snapshot_download(model_handle, local_files_only=True)
        print(f"Volume contains {model_handle}.")
    except FileNotFoundError:
        print(f"Downloading {model_handle} ...")
        snapshot_download(model_handle)

        print("Committing /pretrained directory (no progress bar) ...")
        VOLUME_CONFIG["/pretrained"].commit()

    # Write config and data into a training subfolder.
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = (
        f"unsloth-{time_string}-{secrets.token_hex(2)}"
    )
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)

    # Start training run.
    print(f"Spawning container for training {run_folder}.")
    train_handle = train_unsloth.spawn(run_folder, model_handle)

    with open(f"{run_folder}/logs.txt", "w") as f:
        lbl = "train"
        f.write(f"{lbl}: https://modal.com/logs/call/{train_handle.object_id}")
    VOLUME_CONFIG["/runs"].commit()

    return run_name, train_handle


@app.local_entrypoint()
def main_unsloth(
   model: str
):
    # Read config and data source files and pass their contents to the remote function.
    # with open(config, "r") as cfg, open(train, "r") as train_dat, open(val, "r") as val_dat:
    run_name, launch_handle = launch_unsloth.remote(
        model
    )

    # Write a local reference to the location on the remote volume with the run
    with open(".last_run_name", "w") as f:
        f.write(run_name)

    # Wait for the launch run to finish.
    train_handle = launch_handle.get()
    # if merge_lora and not preproc_only:
    #     merge_handle.get()
    
    # Wait for the training run to finish.
    train_handle.get()

    print(f"Run complete. Tag: {run_name}")
    print(f"To inspect outputs, run `modal volume ls example-runs-vol {run_name}`")


def run_cmd(cmd: str, run_folder: str):
    """Run a command inside a folder, with Modal Volume reloading before and commit on success."""
    import subprocess

    # Ensure volumes contain latest files.
    VOLUME_CONFIG["/pretrained"].reload()
    VOLUME_CONFIG["/runs"].reload()

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):
        exit(exit_code)

    # Commit writes to volume.
    VOLUME_CONFIG["/runs"].commit()
