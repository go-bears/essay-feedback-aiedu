import os
from pathlib import PurePosixPath
from typing import Union

import modal

SUPPORTED_MODELS = {
    "llama-31-8b-it": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "deepseek-r1-distill-llama-8b": "unsloth/DeepSeek-R1-Distill-Llama-8B",
}

APP_NAME = "ft-gemma-3"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"

LLAMA_CPP_RELEASE = "b4568"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

axolotl_image = (
    # modal.Image.from_registry(f"axolotlai/axolotl:main-latest")
    # modal.Image.from_registry("nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10")
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .pip_install(
        "huggingface_hub[hf_transfer]==0.30.1",
        "hf-transfer==0.1.5",
        "fastapi==0.110.0",
        "pydantic",
        "transformers==4.51.0",
        "unsloth",
        "wandb",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="false",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
)

# Extracted from https://ericmjl.github.io/blog/2024/11/14/deploying-ollama-on-modal/
# Configure Modal image with Ollama dependencies

app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("huggingface-rw-joaquin"),
        modal.Secret.from_name("wandb-secret-joaquin"),
        modal.Secret.from_dict({"ALLOW_WANDB": "true"}),
    ],
)

# Volumes for pre-trained models and training runs.
pretrained_volume = modal.Volume.from_name(
    "example-pretrained-vol", create_if_missing=True
)
runs_volume = modal.Volume.from_name("example-runs-vol", create_if_missing=True)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}

class GREGeneralGraderPrompts:
    system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
Please provide a numerical score for the provided essay according to the specified rubric.

- Provide an appropriate holistic score for limited timed test conditions where there is little to no time for revision.
- You will carefully read the rubric (<rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve their essay.

The rubric or rubrics for this essay is as follows:
<rubric>
Score 6
In addressing the specific task directions, a
6 response presents a cogent, well­articulated
analysis of the issue and conveys meaning skillfully.
A typical response in this category exhibits the
following characteristics:
1. It articulates a clear and insightful position on
the issue in accordance with the assigned task.
2. It develops the position fully, with compelling
reasons and/or persuasive examples.
3. It sustains a well­focused, well­organized
analysis, connecting ideas logically.
4. It conveys ideas fluently and precisely, using
effective vocabulary and sentence variety.
5. It demonstrates superior facility with the
conventions of standard written English
(i.e., grammar, usage, and mechanics) but
may have minor errors.

Score 5
In addressing the specific task directions, a
5 response presents a generally thoughtful,
well­developed analysis of the issue and conveys
meaning clearly.
A typical response in this category exhibits the
following characteristics:
1. It presents a clear and well­considered position
on the issue in accordance with the assigned
task.
2. It develops the position with logically sound
reasons and/or well­chosen examples.
3. It is focused and generally well organized,
connecting ideas appropriately.
4. It conveys ideas clearly and well, using
appropriate vocabulary and sentence variety.
5. It demonstrates facility with the conventions of
standard written English but may have minor
errors.

Score 4
In addressing the specific task directions, a
4 response presents a competent analysis of the issue
and conveys meaning with acceptable clarity.
A typical response in this category exhibits the
following characteristics:
1. It presents a clear position on the issue in
accordance with the assigned task.
2. It develops the position with relevant reasons
and/or examples.
3. It is adequately focused and organized.
4. It demonstrates sufficient control of language
to express ideas with acceptable clarity.
5. It generally demonstrates control of the
conventions of standard written English but
may have some errors.

Score 3
A 3 response demonstrates some competence in
addressing the specific task directions, in analyzing
the issue, and in conveying meaning but is obviously
flawed.
A typical response in this category exhibits ONE OR
MORE of the following characteristics:
1. It is vague or limited in addressing the specific
task directions and/or in presenting or
developing a position on the issue.
2. It is weak in the use of relevant reasons or
examples, or relies largely on unsupported
claims.
3. It is limited in focus and/or organization.
4. It has problems in language and sentence
structure that result in a lack of clarity.
5. It contains occasional major errors or frequent
minor errors in grammar, usage, or mechanics
that can interfere with meaning.

Score 2
A 2 response largely disregards the specific task
directions and/or demonstrates serious weaknesses in
analytical writing.
A typical response in this category exhibits ONE OR
MORE of the following characteristics:
1. It is unclear or seriously limited in addressing
the specific task directions and/or in presenting
or developing a position on the issue.
2. It provides few, if any, relevant reasons or
examples in support of its claims.
3. It is poorly focused and/or poorly organized.
4. It has serious problems in language and
sentence structure that frequently interfere
with meaning.
5. It contains serious errors in grammar, usage,
or mechanics that frequently obscure meaning.

Score 1
A 1 response demonstrates fundamental deficiencies
in analytical writing.
A typical response in this category exhibits ONE OR
MORE of the following characteristics:
1. It provides little or no evidence of
understanding the issue.
2. It provides little or no evidence of the ability to
develop an organized response (e.g., is
disorganized and/or extremely brief).
3. It has severe problems in language and
sentence structure that persistently interfere
with meaning.
4. It contains pervasive errors in grammar,
usage, or mechanics that result in incoherence.

Score 0
A 0 response is off topic (i.e., provides no evidence of
an attempt to respond to the assigned topic), written
in a foreign language, merely copies the topic,
consists of only keystroke characters, or is illegible or
nonverbal.
</rubric>

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance.

Output the score in JSON using the following format:
{{
    "score": {{essay_score}},
    "feedback": {{student_feedback}}
}}
"""

    input_prompt = """
This is the student essay you have to score.
<student_essay>
{essay_text}
</student_essay>
"""

# output_prompt = """
# {{
#     "domain_1_score": {domain_1_score}
# }}
# """

# output_prompt_set_2 = """
# {{
#     "domain_1_score": {domain_1_score},
#     "domain_2_score": {domain_2_score}
# }}
# """

# essay_prompt_instruction = """

# Review the given rubric and prompt carefully and score the <student_essay>.

# Provide a numerical domain_1_score by using the provided rubric's guidance.
# Output the score in JSON using the following format:
# {
#     "domain_1_score": {essay_score}
# }
# """

# essay_set_2_essay_prompt_instruction = """
# This essay requires 2 scores, and you have been provided with both rubrics in the system prompt.

# Please provide a numerical score for each domain based on the appropriate rubric.
# Domain 1: Writing Applications
# Domain 2: Language Conventions

# - Be sure to Review the given rubrics and prompt carefully, reasoning through your decision for each domain.

# Output the scores in JSON using the following format:
# {
#     "domain_1_score": {domain_1_score},
#     "domain_2_score": {domain_2_score}
# }
# """

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""




class GREArgumentativeAgentPrompts:
    system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring the argumentative qualities of an essay.
Please provide a numerical score for the provided essay considering all aspects of the specified rubric.

- Provide an appropriate holistic argumentative score for limited timed test conditions where there is little to no time for revision.
- You will carefully read the rubric (<argumentative_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve the argumentative qualities of their essay.

The rubric or rubrics for this essay is as follows:
<argumentative_rubric>
Aspect 1: Quality of the response to the prompt instructions
Score 6: The essay articulates a clear and insightful position on the issue in accordance with the assigned task.
Score 5: The essay presents a clear and well-considered position on the issue in accordance with the assigned task.
Score 4: The essay presents a clear position on the issue in accordance with the assigned task.
Score 3: The essay is vague or limited in addressing the specific task directions and/or in presenting or developing a position on the issue.
Score 2: The essay is unclear or seriously limited in addressing the specific task directions and/or in presenting or developing a position on the issue.
Score 1: The essay presents little or no understanding of how to respond to the prompt.

Aspect 2: Considering the complexities of the issue
Score 6: The essay develops the position fully, with compelling reasons and/or persuasive examples.
Score 5: The essay develops the position with logically sound reasons and/or well-chosen examples.
Score 4: The essay develops the position with relevant reasons and/or examples.
Score 3: The essay is weak in the use of relevant reasons or examples, or relies largely on unsupported claims.
Score 2: The essay provides few, if any, relevant reasons or examples in support of its claims.
Score 1: The essay provides little or no evidence of understanding the issue.

Aspect 3: Organizing, developing, and expressing your ideas
Score 6: The essay sustains a well-focused, well-organized analysis, connecting ideas logically.
Score 5: The essay is focused and generally well organized, connecting ideas appropriately.
Score 4: The essay's ideas are adequately focused and organized.
Score 3: The essay is limited in focus and/or organization.
Score 2: The essay is poorly focused and/or poorly organized.
Score 1: The essay provides little or no evidence of the ability to develop an organized response (e.g., is disorganized and/or extremely brief).
</argumentative_rubric>

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance. The score should be a number between 1 and 6.

Output the score in JSON using the following format:
{{
    "score": {{essay_score}},
    "feedback": {{student_feedback}}
}}
"""

    input_prompt = GREGeneralGraderPrompts.input_prompt

    def format_prompt_inference(grading_instruction) -> str:
        system_prompt_formatted = GREArgumentativeAgentPrompts.system_prompt.format(
            prompt=grading_instruction["prompt"],
            task_directions=grading_instruction["task_directions"],
        )

        input_prompt_formatted = GREArgumentativeAgentPrompts.input_prompt.format(
            essay_text=grading_instruction["essay_text"]
        )

        return alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")


def format_prompt_inference_gre(grading_instruction) -> str:
    system_prompt_formatted = GREGeneralGraderPrompts.system_prompt.format(
        prompt=grading_instruction["prompt"],
        task_directions=grading_instruction["task_directions"],
    )

    input_prompt_formatted = GREGeneralGraderPrompts.input_prompt.format(
        essay_text=grading_instruction["essay_text"]
    )

    return alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")




def format_prompt_training(grading_instruction, eos_token):
    essay_set = int(grading_instruction["essay_set"])
    system_prompt_formatted = system_prompt.format(
        rubric=grading_instruction["rubric"],
        prompt=grading_instruction["essay_prompt"],
    )

    essay_set_prompt_formatted = (
        essay_set_2_essay_prompt_instruction
        if essay_set == 2
        else essay_prompt_instruction
    )
    output_formatted = (
        output_prompt_set_2.format(
            domain_1_score=grading_instruction["grader_score"].split(" ")[0],
            domain_2_score=grading_instruction["grader_score"].split(" ")[1],
        )
        if essay_set == 2
        else output_prompt.format(domain_1_score=grading_instruction["grader_score"])
    )
    instructions = system_prompt_formatted + "\n" + essay_set_prompt_formatted
    inputs = grading_instruction["essay_text"]
    outputs = output_formatted
    # Must add EOS_TOKEN, otherwise your generation will go on forever!
    text = alpaca_prompt.format(instructions, inputs, outputs) + eos_token
    return {
        "text": text,
    }


def get_few_shot_examples(train_df, essay_set: int, num: int) -> str:
    few_shot_examples = "Here are some examples of essays and their scores:\n"

    for idx, row in enumerate(
        train_df[train_df["essay_set"] == str(essay_set)].itertuples()
    ):
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


def format_prompt_inference_iter1(
    grading_instruction, few_shot_n: int, add_argument_annotation: bool, train_df
):
    essay_set = int(grading_instruction["essay_set"])
    system_prompt_formatted = system_prompt.format(
        rubric=grading_instruction["rubric"],
        prompt=grading_instruction["essay_prompt"],
    )

    essay_set_prompt_formatted = (
        essay_set_2_essay_prompt_instruction.format(
            essay_text=grading_instruction["essay_text"]
        )
        if essay_set == 2
        else essay_prompt_instruction.format(
            essay_text=grading_instruction["essay_text"]
        )
    )
    full_prompt = system_prompt_formatted + "\n\n"
    if few_shot_n > 0:
        full_prompt += (
            get_few_shot_examples(train_df, essay_set=essay_set, num=few_shot_n)
            + "\n\n"
        )
    full_prompt += essay_set_prompt_formatted
    if add_argument_annotation:
        full_prompt += (
            "Here is an annotation of the essay's argument components to assist you in scoring:\n"
            + "<student_essay_argument_annotation>\n"
            + grading_instruction["argument_annotation"]
            + "\n</student_essay_argument_annotation>\n\n"
        )
    full_prompt += "<output>" + "\n" + "<explanation>"

    return full_prompt


def format_prompt_inference_ft(grading_instruction):
    essay_set = int(grading_instruction["essay_set"])
    system_prompt_formatted = system_prompt.format(
        rubric=grading_instruction["rubric"],
        prompt=grading_instruction["essay_prompt"],
    )

    essay_set_prompt_formatted = (
        essay_set_2_essay_prompt_instruction
        if essay_set == 2
        else essay_prompt_instruction
    )
    instructions = system_prompt_formatted + "\n" + essay_set_prompt_formatted
    inputs = grading_instruction["essay_text"]
    outputs = ""  # for inference, we don't have outputs
    # Must add EOS_TOKEN, otherwise your generation will go on forever!
    return alpaca_prompt.format(instructions, inputs, outputs)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
