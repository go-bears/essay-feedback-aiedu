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
    SUPPORTED_MODELS,
    format_prompt_training,
    Colors,
)


GPU_CONFIG = "A100-80GB:2"
SINGLE_GPU_CONFIG = os.environ.get("GPU_CONFIG", "a10g:1")


max_seq_length = 64000  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.


def model_setup(model_handle: str):
    from unsloth import FastLanguageModel, FastModel
    import torch

    # LoRa adapters
    # if model_handle == SUPPORTED_MODELS["llama-31-8b-it"]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_handle,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    # elif model_handle == SUPPORTED_MODELS["gemma"]:
    #     model, tokenizer = FastModel.from_pretrained(
    #         model_name=model_handle,
    #         max_seq_length=max_seq_length,  # Choose any for long context!
    #         dtype=dtype,
    #         load_in_4bit=load_in_4bit,
    #     )
    #     model = FastModel.get_peft_model(
    #         model,
    #         finetune_vision_layers=False,  # Turn off for just text!
    #         finetune_language_layers=True,  # Should leave on!
    #         finetune_attention_modules=True,  # Attention good for GRPO
    #         finetune_mlp_modules=True,  # SHould leave on always!
    #         r=8,  # Larger = higher accuracy, but might overfit
    #         lora_alpha=8,  # Recommended alpha == r at least
    #         lora_dropout=0,
    #         bias="none",
    #         random_state=3407,
    #     )
    return model, tokenizer


def dataset_setup(tokenizer):
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    formatting_fun = lambda grading_instruction: format_prompt_training(
        grading_instruction, EOS_TOKEN
    )

    from datasets import load_dataset

    train_dataset = load_dataset("jjordanoc/argumentative-asap", split="train")
    train_dataset = train_dataset.map(formatting_fun)

    # Take a subset of the validation set for evaluation
    eval_dataset = load_dataset("jjordanoc/argumentative-asap", split="validation")
    eval_dataset = eval_dataset.map(formatting_fun)
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(100))
    return train_dataset, eval_dataset


def train_loop(
    model_name: str, model, tokenizer, train_dataset, eval_dataset, max_seq_length: int
):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    import wandb
    import os

    wandb.init(project="asap-ft")
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            # Checkpoint in hub
            push_to_hub=True,
            per_device_eval_batch_size=1,  # ← minimize this
            eval_accumulation_steps=4,
            hub_model_id=model_name,
            save_strategy="epoch",
            # save_strategy="steps",
            warmup_steps=5,
            num_train_epochs=2,
            # max_steps=10,
            do_eval=True,
            learning_rate=2e-4,
            eval_strategy="steps",  # maybe not the best?
            eval_steps=50,
            # eval_steps=150,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="wandb",  # Use this for WandB etc
        ),
    )
    trainer_stats = trainer.train()
    print(trainer_stats)
    return model, trainer


@app.function(
    image=axolotl_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def train_unsloth(run_folder: str, model_handle: str, output_model_name: str):
    import numpy as np

    np.float_ = np.float64

    model, tokenizer = model_setup(model_handle)
    train_dataset, eval_dataset = dataset_setup(tokenizer)
    print(f"{Colors.BLUE + Colors.BOLD}Setup model and datasets{Colors.END}")
    model, trainer = train_loop(
        output_model_name, model, tokenizer, train_dataset, eval_dataset, max_seq_length
    )
    print(
        f"{Colors.GREEN + Colors.BOLD}Adapters pushed to hub as {output_model_name}{Colors.END}"
    )

    # # Push adapters to hub
    # model.push_to_hub(f"jjordanoc/{output_model_name}")
    # tokenizer.push_to_hub(f"jjordanoc/{output_model_name}")

    # # Local save to then load
    # model.save_pretrained(output_model_name)
    # tokenizer.save_pretrained(output_model_name)

    # # Load model from adapters
    # from unsloth import FastLanguageModel
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = output_model_name, # YOUR MODEL YOU USED FOR TRAINING
    #     max_seq_length = max_seq_length,
    #     dtype = dtype,
    #     load_in_4bit = load_in_4bit,
    # )
    # FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    # model.push_to_hub_gguf(output_model_name, tokenizer, quantization_method = "q4_k_m")


@app.function(image=axolotl_image, timeout=30 * MINUTES, volumes=VOLUME_CONFIG)
def launch_unsloth(model_handle: str, output_model_name: str):
    from huggingface_hub import snapshot_download

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
    run_name = f"unsloth-{time_string}-{secrets.token_hex(2)}"
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)

    # Start training run.
    print(f"Spawning container for training {run_folder}.")
    train_handle = train_unsloth.spawn(run_folder, model_handle, output_model_name)

    with open(f"{run_folder}/logs.txt", "w") as f:
        lbl = "train"
        f.write(f"{lbl}: https://modal.com/logs/call/{train_handle.object_id}")
    VOLUME_CONFIG["/runs"].commit()

    return run_name, train_handle


"""
Example use:

Llama 3.1:
modal run --detach -m src.train-unsloth --model=llama-31-8b-it --output-name=llama31-ft-asap
"""


@app.local_entrypoint()
def main_unsloth(model: str, output_name: str):
    if model not in SUPPORTED_MODELS:
        print(f"{Colors.BOLD} Model not supported {Colors.END}")
        print(
            f"{Colors.BOLD} Supported models: {', '.join(SUPPORTED_MODELS.keys())} {Colors.END}"
        )
        return

    run_name, launch_handle = launch_unsloth.remote(
        SUPPORTED_MODELS[model], output_name
    )

    # Write a local reference to the location on the remote volume with the run
    with open(".last_run_name", "w") as f:
        f.write(run_name)

    # Wait for the launch run to finish.
    launch_handle.get()

    print(f"Run complete. Tag: {run_name}")
