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

class BasePrompts:
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



class BasicJudgePrompts(BasePrompts):
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
        summary: list[str]

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



class GREGeneralGraderPrompts(BasePrompts):
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
    

class GREOrchestratorPrompts(BasePrompts):
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




class ASAPGeneralGraderPrompts(BasePrompts):
    dataset_specific_instructions = """
- The essay has been anonymized by replacing revealing details with tags that start with '@' and all letters are capitalized, such as '@ORGANIZATION1', '@CAPS2', '@DATE1', and etc. Do not penalize this. 
- Provide an appropriate score for limited timed test conditions where there is little to no time for revision.
- These essays were written by students ranging in grade levels from Grade 7 to Grade 10 (ages 11-16).
- Use your understanding of the capabilities of 7th to 10th graders to score the essay appropriately.
- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
"""

    system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
Please provide a numerical score for the provided essay according to the specified rubric.

- Provide an appropriate holistic score.
- You will carefully read the rubric (<rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed step-by-step explanation of your reasoning for the score.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.
{dataset_specific_instructions}

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
Provide a numerical score by using the provided rubric's guidance. The score should be a number between {score_range}.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""

    rubric_prompt_1 = """
Score Point 1: An undeveloped response that may take a position but offers no more than very minimal support. Typical elements:
•	Contains few or vague details.
•	Is awkward and fragmented.
•	May be difficult to read and understand.
•	May show no awareness of audience.

Score Point 2: An under-developed response that may or may not take a position. Typical elements:
•	Contains only general reasons with unelaborated and/or list-like details.
•	Shows little or no evidence of organization.
•	May be awkward and confused or simplistic.
•	May show little awareness of audience.

Score Point 3: A minimally-developed response that may take a position, but with inadequate support and details. Typical elements:
•	Has reasons with minimal elaboration and more general than specific details.
•	Shows some organization.
•	May be awkward in parts with few transitions.
•	Shows some awareness of audience.

Score Point 4: A somewhat-developed response that takes a position and provides adequate support. Typical elements:
•	Has adequately elaborated reasons with a mix of general and specific details.
•	Shows satisfactory organization.
•	May be somewhat fluent with some transitional language.
•	Shows adequate awareness of audience.

Score Point 5: A developed response that takes a clear position and provides reasonably persuasive support. Typical elements:
•	Has moderately well elaborated reasons with mostly specific details.
•	Exhibits generally strong organization.
•	May be moderately fluent with transitional language throughout.
•	May show a consistent awareness of audience.

Score Point 6: A well-developed response that takes a clear and thoughtful position and provides persuasive support. Typical elements:
•	Has fully elaborated reasons with specific details.
•	Exhibits strong organization.
•	Is fluent and uses sophisticated transitional language.
•	May show a heightened awareness of audience.
"""
    rubric_prompt_2 = """
Score Point 6: A Score Point 6 paper is rare. It fully accomplishes the task in a thorough and insightful manner and has a distinctive quality that sets it apart as an outstanding performance.
Ideas and Content
Does the writing sample fully accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
•	present a unifying theme or main idea without going off on tangents?
•	stay completely focused on topic and task?
Does the writing sample include thorough, relevant, and complete ideas? Does it
•	include in-depth information and exceptional supporting details that are fully developed?
•	fully explore many facets of the topic?
Organization
Are the ideas in the writing sample organized logically? Does the writing
•	present a meaningful, cohesive whole with a beginning, a middle, and an end (i.e., include an inviting introduction and a strong conclusion)?
•	progress in an order that enhances meaning?
•	include smooth transitions between ideas, sentences, and paragraphs to enhance meaning of text (i.e., have a clear connection of ideas and use topic sentences)?
Style
Does the writing sample exhibit exceptional word usage? Does it
•	include vocabulary to make explanations detailed and precise, descriptions rich, and actions clear and vivid (e.g., varied word choices, action words, appropriate modifiers, sensory details)?
•	demonstrate control of a challenging vocabulary?
Does the writing sample demonstrate exceptional writing technique?
•	Is the writing exceptionally fluent?
•	Does it include varied sentence patterns, including complex sentences?
•	Does it demonstrate use of writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?
Voice
Does the writing sample demonstrate effective adjustment of language and tone to task and reader? Does it
•	exhibit appropriate register (e.g., formal, personal, or dialect) to suit task?
•	demonstrate a strong sense of audience?
•	exhibit an original perspective (e.g., authoritative, lively, and/or exciting)?
Score Point 5: A Score Point 5 paper represents a solid performance. It fully accomplishes the task, but lacks the overall level of sophistication and consistency of a Score Point 6 paper.
Ideas and Content
Does the writing sample fully accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
•	present a unifying theme or main idea without going off on tangents?
•	stay focused on topic and task?
Does the writing sample include many relevant ideas? Does it
•	provide in-depth information and more than adequate supporting details that are developed?
•	explore many facets of the topic?
Organization
Are the ideas in the writing sample organized logically? Does the writing
•	present a meaningful, cohesive whole with a beginning, a middle, and an end (i.e., include a solid introduction and conclusion)?
•	progress in an order that enhances meaning of text?
•	include smooth transitions (e.g., use topic sentences) between sentences and paragraphs to enhance meaning of text? (Writing may have an occasional lapse.)
Style
Does the writing sample exhibit very good word usage? Does it
•	include vocabulary to make explanations detailed and precise, descriptions rich, and actions clear and vivid?
•	demonstrate control of vocabulary?
Does the writing sample demonstrate very good writing technique?
•	Is the writing very fluent?
•	Does it include varied sentence patterns, including complex sentences?
•	Does it demonstrate use of writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?
Voice
Does the writing sample demonstrate effective adjustment of language and tone to task and reader? Does it
•	exhibit appropriate register (e.g., formal, personal, or dialect) to suit task?
•	demonstrate a sense of audience?
•	exhibit an original perspective (e.g., authoritative, lively, and/or exciting)?
Score Point 4: A Score Point 4 paper represents a good performance. It accomplishes the task, but generally needs to exhibit more development, better organization, or a more sophisticated writing style to receive a higher score.
Ideas and Content
Does the writing sample accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
•	present a unifying theme or main idea? (Writing may include minor tangents.)
•	stay mostly focused on topic and task?
Does the writing sample include relevant ideas? Does it
•	include sufficient information and supporting details? (Details may not be fully developed; ideas may be listed.)
•	explore some facets of the topic?
Organization
Are the ideas in the writing sample organized logically? Does the writing
•	present a meaningful whole with a beginning, a middle, and an end despite an occasional lapse (e.g., a weak introduction or conclusion)?
•	generally progress in an order that enhances meaning of text?
•	include transitions between sentences and paragraphs to enhance meaning of text? (Transitions may be rough, although some topic sentences are included.)
Style
Does the writing sample exhibit good word usage? Does it
•	include vocabulary that is appropriately chosen, with words that clearly convey the writer’s meaning?
•	demonstrate control of basic vocabulary?
Does the writing sample demonstrate good writing technique?
•	Is the writing fluent?
•	Does it exhibit some varied sentence patterns, including some complex sentences?
•	Does it demonstrate an attempt to use writer’s techniques (e.g., literary conventions such as imagery and dialogue and/or literary genres such as humor and suspense)?
Voice
Does the writing sample demonstrate an attempt to adjust language and tone to task and reader? Does it
•	generally exhibit appropriate register (e.g., formal, personal, or dialect) to suit task? (The writing may occasionally slip out of register.)
•	demonstrate some sense of audience?
•	attempt an original perspective?
Score Point 3: A Score Point 3 paper represents a performance that minimally accomplishes the task. Some elements of development, organization, and writing style are weak.
Ideas and Content
Does the writing sample minimally accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
•	attempt a unifying theme or main idea?
•	stay somewhat focused on topic and task?
Does the writing sample include some relevant ideas? Does it
•	include some information with only a few details, or list ideas without supporting details?
•	explore some facets of the topic?
Organization
Is there an attempt to logically organize ideas in the writing sample? Does the writing
•	have a beginning, a middle, or an end that may be weak or absent?
•	demonstrate an attempt to progress in an order that enhances meaning? (Progression of text may sometimes be unclear or out of order.)
•	demonstrate an attempt to include transitions? (Are some topic sentences used? Are transitions between sentences and paragraphs weak or absent?)
Style
Does the writing sample exhibit ordinary word usage? Does it
•	contain basic vocabulary, with words that are predictable and common?
•	demonstrate some control of vocabulary?
Does the writing sample demonstrate average writing technique?
•	Is the writing generally fluent?
•	Does it contain mostly simple sentences (although there may be an attempt at more varied sentence patterns)?
•	Is it generally ordinary and predictable?
Voice
Does the writing sample demonstrate an attempt to adjust language and tone to task and reader? Does it
•	demonstrate a difficulty in establishing a register (e.g., formal, personal, or dialect)?
•	demonstrate little sense of audience?
•	generally lack an original perspective?
Score Point 2: A Score Point 2 paper represents a performance that only partially accomplishes the task. Some responses may exhibit difficulty maintaining a focus. Others may be too brief to provide sufficient development of the topic or evidence of adequate organizational or writing style.
Ideas and Content
Does the writing sample only partially accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Does it
•	attempt a main idea?
•	sometimes lose focus or ineffectively display focus?
Does the writing sample include few relevant ideas? Does it
•	include little information and few or no details?
•	explore only one or two facets of the topic?
Organization
Is there a minimal attempt to logically organize ideas in the writing sample?
•	Does the writing have only one or two of the three elements: beginning, middle, and end?
•	Is the writing sometimes difficult to follow? (Progression of text may be confusing or unclear.)
•	Are transitions weak or absent (e.g., few or no topic sentences)?
Style
Does the writing sample exhibit minimal word usage? Does it
•	contain limited vocabulary? (Some words may be used incorrectly.)
•	demonstrate minimal control of vocabulary?
Does the writing sample demonstrate minimal writing technique?
•	Does the writing exhibit some fluency?
•	Does it rely mostly on simple sentences?
•	Is it often repetitive, predictable, or dull?
Voice
Does the writing sample demonstrate language and tone that may be inappropriate to task and reader? Does it
•	demonstrate use of a register inappropriate to the task (e.g., slang or dialect in a formal setting)?
•	demonstrate little or no sense of audience?
•	lack an original perspective?
Score Point 1: A Score Point 1 paper represents a performance that fails to accomplish the task. It exhibits considerable difficulty in areas of development, organization, and writing style. The writing is generally either very brief or rambling and repetitive, sometimes resulting in a response that may be difficult to read or comprehend.
Ideas and Content
Does the writing sample fail to accomplish the task (e.g., support an opinion, summarize, tell a story, or write an article)? Is it
•	difficult for the reader to discern the main idea?
•	too brief or too repetitive to establish or maintain a focus?
Does the writing sample include very few relevant ideas?
•	Does it include little information with few or no details or unrelated details?
•	Is it unsuccessful in attempts to explore any facets of the prompt?
Organization
Are the ideas in the writing sample organized illogically?
•	Does it have only one or two of the three elements: beginning, middle, or end?
•	Is it difficult to follow, with the order possibly difficult to discern?
•	Are transitions weak or absent (e.g., without topic sentences)?
Style
Does the writing sample exhibit less than minimal word usage? Does it
•	contain limited vocabulary, with many words used incorrectly?
•	demonstrate minimal or less than minimal control of vocabulary?
Does the writing sample demonstrate less than minimal writing technique? Does it
•	lack fluency?
•	demonstrate problems with sentence patterns?
•	consist of writing that is flat and lifeless?
Voice
Does the writing sample demonstrate language and tone that may be inappropriate to task and reader? Does it
•	demonstrate difficulty in choosing an appropriate register?
•	demonstrate a lack of a sense of audience?
•	lack an original perspective?
"""

    rubric_prompts = [rubric_prompt_1, rubric_prompt_2]
    score_range = "1-6"

    output_format = """
Your output should be a JSON in the following format:
{
    "score": (your generated score),
    "explanation": (your explanation for the score)
}
"""

    calibration_examples = """
<example_essay>
{example_essay_1}
</example_essay>

<example_score>
{example_score_1}
</example_score>

<example_essay>
{example_essay_2}
</example_essay>

<example_score>
{example_score_2}
</example_score>
"""

    


    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "system_prompt": cls.system_prompt,
            "input_prompt": cls.input_prompt,
            "output_format": cls.output_format,
        }
    
    @classmethod
    def format_prompt_inference(cls, grading_instruction, add_argument_annotation: bool = False) -> str:
        # By default, use the essay text
        essay_text = grading_instruction["essay_text"]
        essay_set = int(grading_instruction["essay_set"])

        if add_argument_annotation:
            essay_text = grading_instruction["argument_annotation"]

        system_prompt_formatted = cls.system_prompt.format(
            prompt=grading_instruction["prompt"],
            task_directions=grading_instruction["task_directions"],
            rubric=cls.rubric_prompts[essay_set-1],
            output_format=cls.output_format,
            score_range=cls.score_range,
            dataset_specific_instructions=cls.dataset_specific_instructions,
        )

        input_prompt_formatted = cls.input_prompt.format(
            essay_text=essay_text
        )
        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")
    

class ASAPOrchestratorPrompts(ASAPGeneralGraderPrompts):
    system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on other expert grader's scores and reasoning.
Please provide a numerical score for the provided essay according to the opinions of the other expert grader's scores and reasoning.
Each expert grader is an expert grader for a specific aspect of the essay.

- You will carefully read each expert grader's score and reasoning, prompt (<essay_prompt>) and student essay (<student_score>), as many times as needed.
- You will reason carefully as to why you chose this score balancing the opinions of the other expert grader's scores and reasoning.
- You will provide a detailed explanation of your reasoning for the score.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.
{dataset_specific_instructions}

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
Provide an integer score between {score_range} by balancing the provided expert grader's scores and reasoning.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{output_format}
"""

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "system_prompt": cls.system_prompt,
            "input_prompt": cls.input_prompt,
            "output_format": cls.output_format,

        }

    @classmethod
    def format_prompt_inference(cls, grading_instruction, domain_outputs) -> str:
        expert_grader_scores_and_reasoning = ""
        for domain_output, aspect_name in zip(domain_outputs, ASAPAgentAlphaPrompts.aspect_names):
            expert_grader_scores_and_reasoning += f"""
<expert_grader_judgement>
{aspect_name}
{domain_output}
</expert_grader_judgement>\n
"""

        system_prompt_formatted = cls.system_prompt.format(
            expert_grader_scores_and_reasoning=expert_grader_scores_and_reasoning,
            prompt=grading_instruction["prompt"],
            task_directions=grading_instruction["task_directions"],
            output_format=cls.output_format,
            score_range=cls.score_range,
            dataset_specific_instructions=cls.dataset_specific_instructions,
        )
        input_prompt_formatted = cls.input_prompt.format(
            essay_text=grading_instruction["essay_text"]
        )

        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")


class ASAPAgentAlphaPrompts(ASAPGeneralGraderPrompts):
    aspect_1_rubric = """
1. Ideas & Content
This property checks for the amount of content and ideas present in the essay.
Score 6: The writing is exceptionally clear, focused, and interesting. It holds the reader’s
attention throughout. Main ideas stand out and are developed by strong support and rich details
suitable to audience and purpose. The writing is characterized by
• clarity, focus, and control.
• main idea(s) that stand out.
• supporting, relevant, carefully selected details; when appropriate, use of
resources provides strong, accurate, credible support.
• a thorough, balanced, in-depth explanation / exploration of the topic; the writing
makes connections and shares insights.
• content and selected details that are well-suited to audience and purpose.
Score 5: The writing is clear, focused and interesting. It holds the reader’s attention. Main ideas
stand out and are developed by supporting details suitable to audience and purpose. The
writing is characterized by
• clarity, focus, and control.
• main idea(s) that stand out.
• supporting, relevant, carefully selected details; when appropriate, use of
resources provides strong, accurate, credible support.
• a thorough, balanced explanation / exploration of the topic; the writing makes
connections and shares insights.
• content and selected details that are well-suited to audience and purpose.
Score 4: The writing is clear and focused. The reader can easily understand the main ideas.
Support is present, although it may be limited or rather general. The writing is characterized by
• an easily identifiable purpose.
• clear main idea(s).
• supporting details that are relevant, but may be overly general or limited in
places; when appropriate, resources are used to provide accurate support.
• a topic that is explored / explained, although developmental details may
occasionally be out of balance with the main idea(s); some connections and insights may be
present.
• content and selected details that are relevant, but perhaps not consistently
well-chosen for audience and purpose.
Score 3: The reader can understand the main ideas, although they may be overly broad or
simplistic, and the results may not be effective. Supporting detail is often limited, insubstantial,
overly general, or occasionally slightly off-topic. The writing is characterized by
• an easily identifiable purpose and main idea(s).
• predictable or overly-obvious main ideas; or points that echo observations heard
elsewhere; or a close retelling of another work.
• support that is attempted, but developmental details are often limited, uneven,
somewhat off-topic, predictable, or too general (e.g., a list of underdeveloped points).
• details that may not be well-grounded in credible resources; they may be based
on clichés, stereotypes or questionable sources of information.
• difficulties when moving from general observations to specifics.
Score 2: Main ideas and purpose are somewhat unclear or development is attempted but
minimal. The writing is characterized by
• a purpose and main idea(s) that may require extensive inferences by the reader.
• minimal development; insufficient details.
• irrelevant details that clutter the text.
• extensive repetition of detail.
Score 1: The writing lacks a central idea or purpose. The writing is characterized by
• ideas that are extremely limited or simply unclear.
• attempts at development that are minimal or nonexistent; the paper is too short to
demonstrate the development of an idea.
"""

    aspect_2_rubric = """
2. Organization
This property checks how well structured the essay is. NOTE: Since the dataset has the essays
compressed into one line, please bear in mind that the paragraph information is lost. Hence,
give writers the benefit of the doubt here.
Score 6: The essay is well-organized. There is a clear flow of ideas with each idea
self-contained (this is where we assume that each idea is contained in a paragraph). The essay
has the appropriate form as a letter to the editor.
Score 5: The essay shows good organization. There is a flow of ideas. However, the ideas are
mostly self-contained. The essay has the appropriate form as a letter to the editor.
Score 4: The essay shows satisfactory organization. It contains a basic introduction, body and
conclusion.
Score 3: The essay shows some organization. Its form may not be that of a letter to the editor.
Its ideas are not necessarily self-contained.
Score 2: Shows little or no evidence of organization.
Score 1: The essay is awkward and fragmented. Ideas are not self-contained.
"""

    aspect_3_rubric = """
3. Word Choice
Score 6: Words convey the intended message in an exceptionally interesting, precise, and
natural way appropriate to audience and purpose. The writer employs a rich, broad range of
words which have been carefully chosen and thoughtfully placed for impact. The writing is
characterized by
• accurate, strong, specific words; powerful words energize the writing.
• fresh, original expression; slang, if used, seems purposeful and is effective.
• vocabulary that is striking and varied, but that is natural and not overdone.
• ordinary words used in an unusual way.
• words that evoke strong images; figurative language may be used.
Score 5: Words convey the intended message in an interesting, precise, and natural way
appropriate to audience and purpose. The writer employs a broad range of words which have
been carefully chosen and thoughtfully placed for impact. The writing is characterized by
• accurate, specific words; word choices energize the writing.
• fresh, vivid expression; slang, if used, seems purposeful and is effective.
• vocabulary that may be striking and varied, but that is natural and not overdone.
• ordinary words used in an unusual way.
• words that evoke clear images; figurative language may be used.
Score 4: Words effectively convey the intended message. The writer employs a variety of words
that are functional and appropriate to audience and purpose. The writing is characterized by
• words that work but do not particularly energize the writing.
• expression that is functional; however, slang, if used, does not seem purposeful
and is not particularly effective.
• attempts at colorful language that may occasionally seem overdone.
• occasional overuse of technical language or jargon.
• rare experiments with language; however, the writing may have some fine
moments and generally avoids clichés.
Score 3: Language lacks precision and variety, or may be inappropriate to audience and
purpose in places. The writer does not employ a variety of words, producing a sort of “generic”
paper filled with familiar words and phrases. The writing is characterized by
• words that work, but that rarely capture the reader’s interest.
• expression that seems mundane and general; slang, if used, does not seem
purposeful and is not effective.
• attempts at colorful language that seem overdone or forced.
• words that are accurate for the most part, although misused words may
occasionally appear; technical language or jargon may be overused or inappropriately used.
• reliance on clichés and overused expressions.
• text that is too short to demonstrate variety.
Score 2: Language is monotonous and/or misused, detracting from the meaning and impact.
The writing is characterized by
• words that are colorless, flat or imprecise.
• monotonous repetition or overwhelming reliance on worn expressions that
repeatedly detract from the message.
Score 1: The writing shows an extremely limited vocabulary or is so filled with misuses of words
that the meaning is obscured. Only the most general kind of message is communicated
because of vague or imprecise language. The writing is characterized by
• general, vague words that fail to communicate.
• an extremely limited range of words.
• words that simply do not fit the text; they seem imprecise, inadequate, or just
plain wrong.
"""

    aspect_4_rubric = """
4. Sentence Fluency
Score 6: The writing has an effective flow and rhythm. Sentences show a high degree of
craftsmanship, with consistently strong and varied structure that makes expressive oral reading
easy and enjoyable. The writing is characterized by
• a natural, fluent sound; it glides along with one sentence flowing effortlessly into
the next.
• extensive variation in sentence structure, length, and beginnings that add interest
to the text.
• sentence structure that enhances meaning by drawing attention to key ideas or
reinforcing relationships among ideas.
• varied sentence patterns that create an effective combination of power and
grace.
• strong control over sentence structure; fragments, if used at all, work well.
• stylistic control; dialogue, if used, sounds natural.
Score 5: The writing has an easy flow and rhythm. Sentences are carefully crafted, with strong
and varied structure that makes expressive oral reading easy and enjoyable. The writing is
characterized by
• a natural, fluent sound; it glides along with one sentence flowing into the next.
• variation in sentence structure, length, and beginnings that add interest to the
text.
• sentence structure that enhances meaning.
• control over sentence structure; fragments, if used at all, work well.
• stylistic control; dialogue, if used, sounds natural.
Score 4: The writing flows; however, connections between phrases or sentences may be less
than fluid. Sentence patterns are somewhat varied, contributing to ease in oral reading. The
writing is characterized by
• a natural sound; the reader can move easily through the piece, although it may
lack a certain rhythm and grace.
• some repeated patterns of sentence structure, length, and beginnings that may
detract somewhat from overall impact.
• strong control over simple sentence structures, but variable control over more
complex sentences; fragments, if present, are usually effective.
• occasional lapses in stylistic control; dialogue, if used, sounds natural for the
most part, but may at times sound stilted or unnatural.
Score 3: The writing tends to be mechanical rather than fluid. Occasional awkward
constructions may force the reader to slow down or reread. The writing is characterized by
• some passages that invite fluid oral reading; however, others do not.
• some variety in sentence structure, length, and beginnings, although the writer
falls into repetitive sentence patterns.
• good control over simple sentence structures, but little control over more complex
sentences; fragments, if present, may not be effective.
• sentences which, although functional, lack energy.
• lapses in stylistic control; dialogue, if used, may sound stilted or unnatural.
• text that is too short to demonstrate variety and control.
Score 2: The writing tends to be either choppy or rambling. Awkward constructions often force
the reader to slow down or reread. The writing is characterized by
• significant portions of the text that are difficult to follow or read aloud.
• sentence patterns that are monotonous (e.g., subject-verb or
subject-verb-object).
• a significant number of awkward, choppy, or rambling constructions.
Score 1: The writing is difficult to follow or to read aloud. Sentences tend to be incomplete,
rambling, or very awkward. The writing is characterized by
• text that does not invite—and may not even permit—smooth oral reading.
• confusing word order that is often jarring and irregular.
• sentence structure that frequently obscures meaning.
• sentences that are disjointed, confusing, or rambling.
"""

    aspect_5_rubric = """
5. Conventions
Score 6: The writing demonstrates exceptionally strong control of standard writing conventions
(e.g., punctuation, spelling, capitalization, grammar and usage) and uses them effectively to
enhance communication. Errors are so few and so minor that the reader can easily skim right
over them unless specifically searching for them. The writing is characterized by
• strong control of conventions; manipulation of conventions may occur for stylistic
effect.
• strong, effective use of punctuation that guides the reader through the text.
• correct spelling, even of more difficult words.
• correct grammar and usage that contribute to clarity and style.
• skill in using a wide range of conventions in a sufficiently long and complex piece.
• little or no need for editing.
Score 5: The writing demonstrates strong control of standard writing conventions (e.g.,
punctuation, spelling, capitalization, grammar and usage) and uses them effectively to enhance
communication. Errors are few and minor. Conventions support readability. The writing is
characterized by
• strong control of conventions.
• effective use of punctuation that guides the reader through the text.
• correct spelling, even of more difficult words.
• correct capitalization; errors, if any, are minor.
• correct grammar and usage that contribute to clarity and style.
• skill in using a wide range of conventions in a sufficiently long and complex piece.
• little need for editing.
Score 4: The writing demonstrates control of standard writing conventions (e.g., punctuation,
spelling, capitalization, grammar and usage). Significant errors do not occur frequently. Minor
errors, while perhaps noticeable, do not impede readability. The writing is characterized by
• control over conventions used, although a wide range is not demonstrated.
• correct end-of-sentence punctuation; internal punctuation may sometimes be
incorrect.
• spelling that is usually correct, especially on common words.
• correct capitalization; errors, if any, are minor.
• occasional lapses in correct grammar and usage; problems are not severe
enough to distort meaning or confuse the reader.
• moderate need for editing.
Score 3: The writing demonstrates limited control of standard writing conventions (e.g.,
punctuation, spelling, capitalization, grammar and usage). Errors begin to impede readability.
The writing is characterized by
• some control over basic conventions; the text may be too simple or too short to
reveal mastery.
• end-of-sentence punctuation that is usually correct; however, internal punctuation
contains frequent errors.
• spelling errors that distract the reader; misspelling of common words occurs.
• capitalization errors.
• errors in grammar and usage that do not block meaning but do distract the
reader.
• significant need for editing.
Score 2: The writing demonstrates little control of standard writing conventions. Frequent,
significant errors impede readability. The writing is characterized by
• little control over basic conventions.
• many end-of-sentence punctuation errors; internal punctuation contains frequent
errors.
• spelling errors that frequently distract the reader; misspelling of common words
often occurs.
• capitalization that is inconsistent or often incorrect.
• errors in grammar and usage that interfere with readability and meaning.
• substantial need for editing.
Score 1: Numerous errors in usage, spelling, capitalization, and punctuation repeatedly distract
the reader and make the text difficult to read. In fact, the severity and frequency of errors are so
overwhelming that the reader finds it difficult to focus on the message and must reread for
meaning. The writing is characterized by
• very limited skill in using conventions.
• basic punctuation (including end-of-sentence punctuation) that tends to be
omitted, haphazard, or incorrect.
• frequent spelling errors that significantly impair readability.
• capitalization that appears to be random.
• a need for extensive editing
"""

    aspect_names = ["Content", "Organization", "Word Choice", "Sentence Fluency", "Conventions"]
    aspect_rubrics = [aspect_1_rubric, aspect_2_rubric, aspect_3_rubric, aspect_4_rubric, aspect_5_rubric]

    per_trait_system_prompt = """
You are an expert professional grader who scores student essays tagged <student_essay> based on a rubric. 
You specialize in scoring {aspect_name}.
Please provide a numerical score for the provided essay considering the specified rubric for {aspect_name}.

- Provide an appropriate holistic {aspect_name} score.
- You will carefully read the rubric (<{aspect_name}_rubric>), prompt (<essay_prompt>) and student essay (<student_essay>), as many times as needed.
- You will reason carefully as to why you chose this score following the rubric and guidelines.
- You will provide a detailed step-by-step explanation of your reasoning for the score.
- A low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.
{dataset_specific_instructions}

The rubric or rubrics for this essay is as follows:
<{aspect_name}_rubric>
{aspect_rubric}
</{aspect_name}_rubric>

The given task is as follows:
<task_directions>
{task_directions}
</task_directions>

The prompt is as follows:
<essay_prompt>
{prompt}
</essay_prompt>

Review the given rubric and prompt carefully and score the <student_essay>.
Provide a numerical score by using the provided rubric's guidance for {aspect_name}. The score should be a number between {score_range}.
Remember, a low score isn't harmful to the student. Rather, an accurate match to the rubric will help the student improve their score in future essays.

{example_output}

{output_format}
"""

    @classmethod
    def dump_prompts(cls) -> dict:
        return {
            "per_trait_system_prompt": cls.per_trait_system_prompt,
            "input_prompt": cls.input_prompt,
            "aspect_rubrics": cls.aspect_rubrics,
            "aspect_names": cls.aspect_names,
            "output_format": cls.output_format,
            "dataset_specific_instructions": cls.dataset_specific_instructions,
        }

    @classmethod
    def format_prompt_inference(cls, grading_instruction, agent_rubric_item: int, add_argument_annotation: bool = False, calibration_examples = None) -> str:
        aspect_name = cls.aspect_names[agent_rubric_item-1]
        examples = cls.calibration_examples.format(
            example_essay_1=calibration_examples["essay_text"][0],
            example_score_1=calibration_examples["trait_scores"][0][aspect_name],
            example_essay_2=calibration_examples["essay_text"][1],
            example_score_2=calibration_examples["trait_scores"][1][aspect_name],
        ) if calibration_examples is not None else ""
        # By default, use the essay text
        essay_text = grading_instruction["essay_text"]
        assert agent_rubric_item in [1, 2, 3, 4, 5]
        system_prompt_formatted = cls.per_trait_system_prompt.format(
            aspect_name=c,
            aspect_rubric=cls.aspect_rubrics[agent_rubric_item-1],
            prompt=grading_instruction["prompt"],
            task_directions=grading_instruction["task_directions"],
            output_format=cls.output_format,
            score_range=cls.score_range,
            dataset_specific_instructions=cls.dataset_specific_instructions,
            example_output=examples,
        )
        # Use argument annotations if provided (only for argumentative aspects)
        if add_argument_annotation:
            essay_text = grading_instruction["argument_annotation"]
        
        input_prompt_formatted = cls.input_prompt.format(
            essay_text=essay_text
        )

        return cls.alpaca_prompt.format(system_prompt_formatted, input_prompt_formatted, "")




class GREAgentPrompts(BasePrompts):
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
- You will provide a detailed step-by-step explanation of your reasoning for the score.
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
    RED = "\033[0;31m"
