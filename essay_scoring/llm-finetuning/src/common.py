import os
from pathlib import PurePosixPath
from typing import Union
from pydantic import BaseModel

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

class GREBasePrompts:
    input_prompt = """
This is the student essay you have to score.
<student_essay>
{essay_text}
</student_essay>
"""
    output_format = """
Your output should be a JSON in the following format:
{
    "score": (your generated score),
    "feedback": (your generated feedback)
}
"""
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    rubric = """
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
"""



class BasicJudgePrompts(GREBasePrompts):
    system_prompt = """
You are an expert professional grader who specializes in evaluating feedback from expert graders.
You will be given two feedbacks for an essay crafted by two expert graders, and will choose the better feedback.

- You will carefully read the two feedbacks (<feedback_1> and <feedback_2>), the essay (<student_essay>), and the rubric (<rubric>).
- You will reason carefully as to which feedback is more accurate and helpful to the student and why.
- You will choose the better feedback (<feedback_1> or <feedback_2>) and provide a detailed explanation of your reasoning for the choice.

The rubric for the essay is as follows:
<rubric>
{rubric}
</rubric>

The two feedbacks are as follows:
<feedback_1>
{feedback_1}
</feedback_1>

<feedback_2>
{feedback_2}
</feedback_2>

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

The student essay is as follows:
<student_essay>
{student_essay}
</student_essay>

Provide a number (1 or 2) representing the feedback that you choose.

{output_format}
"""
    output_format = """
Your output should be a JSON in the following format:
{
    "feedback_choice": (1 or 2)
}
"""

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "system_prompt": cls.system_prompt,
            "output_format": cls.output_format,
        }
    
    @classmethod
    def format_prompt_judge(cls, feedback_1, feedback_2, student_essay, task_directions, prompt) -> str:
        return cls.system_prompt.format(
            feedback_1=feedback_1,
            feedback_2=feedback_2,
            student_essay=student_essay,
            rubric=cls.rubric,
            task_directions=task_directions,
            prompt=prompt,
            output_format=cls.output_format,
        )


class RubricJudgePrompts(BasicJudgePrompts):
    system_prompt = """
You are an expert professional grader who specializes in evaluating feedback from expert graders.
You will be given two feedbacks for an essay crafted by two expert graders. 
You will choose the better feedback (<feedback_1> or <feedback_2>) for each of the criteria specified in <criteria>.

<criteria>
- C1: Which feedback is more relevant to the essay content?
- C2: Which feedback is better at highlighting weakness?
- C3: Which feedback is better at highlighting strengths?
- C4: Which feedback is more specific and actionable?
- C5: Which feedback is overall more helpful for a student?
</criteria>

The rubric for the essay is as follows:
<rubric>
{rubric}
</rubric>

The two feedbacks are as follows:
<feedback_1>
{feedback_1}
</feedback_1>

<feedback_2>
{feedback_2}
</feedback_2>

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

The student essay is as follows:
<student_essay>
{student_essay}
</student_essay>

Provide a number (1 or 2) representing the feedback that you choose for each of the criteria.

{output_format}
"""
    output_format = """
Your output should be a JSON in the following format:
{
    "c1": (1 or 2),
    "c2": (1 or 2),
    "c3": (1 or 2),
    "c4": (1 or 2),
    "c5": (1 or 2)
}
"""
    class ResponseModel(BaseModel):
        c1: int
        c2: int
        c3: int
        c4: int
        c5: int

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "system_prompt": cls.system_prompt,
            "output_format": cls.output_format,
        }
    
    @classmethod
    def format_prompt_judge(cls, feedback_1, feedback_2, student_essay, task_directions, prompt) -> str:
        return cls.system_prompt.format(
            feedback_1=feedback_1,
            feedback_2=feedback_2,
            student_essay=student_essay,
            rubric=cls.rubric,
            task_directions=task_directions,
            prompt=prompt,
            output_format=cls.output_format,
        )



class GREGeneralGraderPrompts(GREBasePrompts):
    system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
Please provide a numerical score for the provided essay according to the specified rubric.

- Provide an appropriate holistic score.
- You will carefully read the rubric (<rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve their essay.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The rubric or rubrics for this essay is as follows:
<rubric>
{rubric}
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
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""
    

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "system_prompt": cls.system_prompt,
            "input_prompt": cls.input_prompt,
        }
    
    @classmethod
    def format_prompt_inference(cls, grading_instruction, add_argument_annotation: bool = False) -> str:
        # By default, use the essay text
        essay_text = grading_instruction["essay_text"]
        if add_argument_annotation:
            essay_text = grading_instruction["argument_annotation"]

        system_prompt_formatted = cls.system_prompt.format(
            prompt=grading_instruction["prompt"],
            task_directions=grading_instruction["task_directions"],
            rubric=cls.rubric,
            output_format=cls.output_format,
        )

        input_prompt_formatted = cls.input_prompt.format(
            essay_text=essay_text
        )
        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")
    

class GREOrchestratorPrompts(GREBasePrompts):
    system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on other expert grader's scores and reasoning.
Please provide a numerical score for the provided essay according to the opinions of the other expert grader's scores and reasoning.
Each expert grader is an expert grader for a specific aspect of the essay.

- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
- You will carefully read each expert grader's score and reasoning, prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score balancing the opinions of the other expert grader's scores and reasoning.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve their essay, balancing the opinions of the other expert grader's feedback.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The expert grader's scores and reasoning are as follows:
{expert_grader_scores_and_reasoning}

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given expert grader's scores and reasoning, prompt and student essay carefully and score the <student_essay>.
Provide an integer score between 0 and 6 by balancing the provided expert grader's scores and reasoning.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "system_prompt": cls.system_prompt,
            "input_prompt": cls.input_prompt,
        }

    @classmethod
    def format_prompt_inference(cls, grading_instruction, domain_scores, domain_feedbacks) -> str:
        expert_grader_scores_and_reasoning = ""
        for domain_score, domain_feedback, aspect_rubric in zip(domain_scores, domain_feedbacks, GREAgentPrompts.aspect_rubrics):
            aspect_name = aspect_rubric.strip().split(".", 1)[0] + "."
            expert_grader_scores_and_reasoning += f"""
<expert_grader_judgement>
{aspect_name}
{{
    "score": {domain_score},
    "feedback": {domain_feedback}
}}
</expert_grader_judgement>\n
"""

        system_prompt_formatted = cls.system_prompt.format(
            expert_grader_scores_and_reasoning=expert_grader_scores_and_reasoning,
            prompt=grading_instruction["prompt"],
            task_directions=grading_instruction["task_directions"],
            output_format=cls.output_format,
        )
        input_prompt_formatted = cls.input_prompt.format(
            essay_text=grading_instruction["essay_text"]
        )

        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")






class GREAgentPrompts(GREBasePrompts):
    aspect_1_rubric = """
Aspect 1: Quality of the response to the prompt instructions
Score 6: The essay articulates a clear and insightful position on the issue in accordance with the assigned task.
Score 5: The essay presents a clear and well-considered position on the issue in accordance with the assigned task.
Score 4: The essay presents a clear position on the issue in accordance with the assigned task.
Score 3: The essay is vague or limited in addressing the specific task directions and/or in presenting or developing a position on the issue.
Score 2: The essay is unclear or seriously limited in addressing the specific task directions and/or in presenting or developing a position on the issue.
Score 1: The essay presents little or no understanding of how to respond to the prompt.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""
    aspect_2_rubric = """
Aspect 2: Considering the complexities of the issue
Score 6: The essay develops the position fully, with compelling reasons and/or persuasive examples.
Score 5: The essay develops the position with logically sound reasons and/or well-chosen examples.
Score 4: The essay develops the position with relevant reasons and/or examples.
Score 3: The essay is weak in the use of relevant reasons or examples, or relies largely on unsupported claims.
Score 2: The essay provides few, if any, relevant reasons or examples in support of its claims.
Score 1: The essay provides little or no evidence of understanding the issue.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""
    aspect_3_rubric = """
Aspect 3: Organizing, developing, and expressing ideas
Score 6: The essay sustains a well-focused, well-organized analysis, connecting ideas logically.
Score 5: The essay is focused and generally well organized, connecting ideas appropriately.
Score 4: The essay's ideas are adequately focused and organized.
Score 3: The essay is limited in focus and/or organization.
Score 2: The essay is poorly focused and/or poorly organized.
Score 1: The essay provides little or no evidence of the ability to develop an organized response (e.g., is disorganized and/or extremely brief).
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""

    aspect_4_rubric = """
Aspect 4: Vocabulary and sentence variety
Score 6: The essay conveys ideas fluently and precisely, using effective vocabulary and sentence variety.
Score 5: The essay conveys ideas clearly and well, using appropriate vocabulary and sentence variety.
Score 4: The essay conveys ideas with acceptable clarity, demonstrating sufficient control of language.
Score 3: The essay has problems in language and sentence structure that result in a lack of clarity.
Score 2: The essay has serious problems in language and sentence structure that frequently interfere with meaning.
Score 1: The essay has severe problems in language and sentence structure that persistently interfere with meaning.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""

    aspect_5_rubric = """
Aspect 5: Grammar and mechanics
Score 6: The essay demonstrates superior facility with the conventions of standard written English (i.e., grammar, usage, and mechanics) but may have minor errors.
Score 5: The essay demonstrates facility with the conventions of standard written English but may have minor errors.
Score 4: The essay generally demonstrates control of the conventions of standard written English but may have some errors.
Score 3: The essay contains contains occasional major errors or frequent minor errors in grammar, usage, or mechanics that can interfere with meaning.
Score 2: The essay contains serious errors in grammar, usage, or mechanics that frequently obscure meaning.
Score 1: The essay contains pervasive errors in grammar, usage, or mechanics that result in incoherence.
Score 0: The essay is off topic (i.e., provides no evidence of an attempt to respond to the assigned topic), written in a foreign language, merely copies the topic, consists of only keystroke characters, or is illegible or nonverbal.
"""

    aspect_rubrics = [aspect_1_rubric, aspect_2_rubric, aspect_3_rubric, aspect_4_rubric, aspect_5_rubric]

    argumentative_system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring the argumentative qualities of an essay.
Please provide a numerical score for the provided essay considering all aspects of the specified rubric.


- Provide an appropriate holistic argumentative score.
- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
- You will carefully read the rubric (<argumentative_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve the argumentative qualities of their essay.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The rubric or rubrics for this essay is as follows:
<argumentative_rubric>
{argumentative_rubric}
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
Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""
    vocabulary_system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring the vocabulary and sentence variety of an essay.
Please provide a numerical score for the provided essay considering all aspects of the specified rubric.

- Provide an appropriate holistic vocabulary score.
- You will carefully read the rubric (<vocabulary_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve the vocabulary and sentence variety of their essay.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The rubric or rubrics for this essay is as follows:
<vocabulary_rubric>
{vocabulary_rubric}
</vocabulary_rubric>

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""

    grammar_system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring the grammar and mechanics of an essay.
Please provide a numerical score for the provided essay considering all aspects of the specified rubric.

- Provide an appropriate holistic grammar score.
- You will carefully read the rubric (<grammar_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed explanation of your reasoning for the score.
- You will provide feedback for the student on how to improve the grammar and mechanics of their essay.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

The rubric or rubrics for this essay is as follows:
<grammar_rubric>
{grammar_rubric}
</grammar_rubric>

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance. The score should be a number between 0 and 6.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""
    

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "argumentative_system_prompt": cls.argumentative_system_prompt,
            "vocabulary_system_prompt": cls.vocabulary_system_prompt,
            "grammar_system_prompt": cls.grammar_system_prompt,
            "input_prompt": cls.input_prompt,
            "aspect_rubrics": cls.aspect_rubrics,
        }

    @classmethod
    def format_prompt_inference(cls, grading_instruction, agent_rubric_item: int, add_argument_annotation: bool = False) -> str:
        """
        Aspects 1-3 are argumentative, 4 is vocabulary, 5 is grammar
        """
        # By default, use the essay text
        essay_text = grading_instruction["essay_text"]
        assert agent_rubric_item in [1, 2, 3, 4, 5]
        if agent_rubric_item == 4:
            system_prompt_formatted = cls.vocabulary_system_prompt.format(
                vocabulary_rubric=cls.aspect_rubrics[agent_rubric_item-1],
                prompt=grading_instruction["prompt"],
                task_directions=grading_instruction["task_directions"],
                output_format=cls.output_format,
            )
        elif agent_rubric_item == 5:
            system_prompt_formatted = cls.grammar_system_prompt.format(
                grammar_rubric=cls.aspect_rubrics[agent_rubric_item-1],
                prompt=grading_instruction["prompt"],
                task_directions=grading_instruction["task_directions"],
                output_format=cls.output_format,
            )
        else:
            system_prompt_formatted = cls.argumentative_system_prompt.format(
                argumentative_rubric=cls.aspect_rubrics[agent_rubric_item-1],
                prompt=grading_instruction["prompt"],
                task_directions=grading_instruction["task_directions"],
                output_format=cls.output_format,
            )
            # Use argument annotations if provided (only for argumentative aspects)
            if add_argument_annotation:
                essay_text = grading_instruction["argument_annotation"]
        
        input_prompt_formatted = cls.input_prompt.format(
            essay_text=essay_text
        )

        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")







# def format_prompt_training(grading_instruction, eos_token):
#     essay_set = int(grading_instruction["essay_set"])
#     system_prompt_formatted = system_prompt.format(
#         rubric=grading_instruction["rubric"],
#         prompt=grading_instruction["essay_prompt"],
#     )

#     essay_set_prompt_formatted = (
#         essay_set_2_essay_prompt_instruction
#         if essay_set == 2
#         else essay_prompt_instruction
#     )
#     output_formatted = (
#         output_prompt_set_2.format(
#             domain_1_score=grading_instruction["grader_score"].split(" ")[0],
#             domain_2_score=grading_instruction["grader_score"].split(" ")[1],
#         )
#         if essay_set == 2
#         else output_prompt.format(domain_1_score=grading_instruction["grader_score"])
#     )
#     instructions = system_prompt_formatted + "\n" + essay_set_prompt_formatted
#     inputs = grading_instruction["essay_text"]
#     outputs = output_formatted
#     # Must add EOS_TOKEN, otherwise your generation will go on forever!
#     text = alpaca_prompt.format(instructions, inputs, outputs) + eos_token
#     return {
#         "text": text,
#     }


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
    RED = "\033[0;31m"
