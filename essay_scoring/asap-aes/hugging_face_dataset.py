import argparse
import json
import math
import os

# import chevron
import pandas as pd
from datasets import Dataset, DatasetDict
from openai import OpenAI
import glob

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

explanation_prompt = """
You will receive a rubric and an expert grader's rating for a given essay.
Your job is to explain why the expert grader gave a score of {score} and not a lower or a higher score.

Here is the prompt:
{prompt}

Here is the task directions:
{task_directions}

Here is the rubric:
{rubric}

Here is the human grader's rating:
{score}

Here is the essay:
{essay}
"""

holistic_explanation_prompt = """
You will receive expert grader's scores and explanations for every aspect of an essay.
Your job is to explain why the expert grader gave a score of {score} and not a lower or a higher score given the other expert grader's scores and explanations.

Here is the prompt:
{prompt}

Here is the task directions:
{task_directions}

Here is the grader's scores and explanations:
{scores_and_explanations}

Here is the essay:
{essay}
"""

dataset_specific_instructions = """
- The essay has been anonymized by replacing revealing details with tags that start with '@' and all letters are capitalized, such as '@ORGANIZATION1', '@CAPS2', '@DATE1', and etc. Do not penalize this. 
- Provide an appropriate score for limited timed test conditions where there is little to no time for revision.
- These essays were written by students ranging in grade levels from Grade 7 to Grade 10 (ages 11-16).
- Use your understanding of the capabilities of 7th to 10th graders to score the essay appropriately.
- The length of the essay matters, a well developed essay should have at least 3-4 well written paragraphs.
"""

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
{rubric}
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

{output_format}

<student_essay>
{student_essay}
</student_essay>
"""

scoring_prompt = """
You will receive a rubric and a student essay.
Your job is to score the essay based on the rubric.

Here is the prompt:
{prompt}

Here is the task directions:
{task_directions}

Here is the rubric:
{rubric}

Here is the essay:
{essay}
"""

output_format = """
Your output should be a JSON in the following format:
{
    "score": (your generated score),
    "explanation": (your explanation for the score)
}
"""


def generate(client, prompt):
    assert False
    response = client.chat.completions.create(
            model="o4-mini",
            reasoning_effort="high",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
    )
    print(response.choices[0].message.content)
    print("--------------------------------")
    return response.choices[0].message.content

def load(essay_id, aspect):
    with open(f"generations.jsonl", "r") as f:
        for line in f:
            json_data = json.loads(line)
            if json_data["essay_id"] != essay_id:
                continue
            if aspect == "holistic":
                return json_data["holistic_explanation"]
            else:
                return json_data["trait_explanations"][aspect]
    return None

def main():
    client = OpenAI()
    original_data = pd.read_csv("training_set_rel3.tsv", sep="\t", encoding="ISO-8859-1")
    rubrics_list = []
    task_directions_list = []
    prompts_list = []

    # Read the corresponding prompt from prompts/{essay_set}.txt
    for essay_set in [1, 2]:
        prompt_path = os.path.join("prompts", f"{essay_set}.txt")
        with open(prompt_path, "r") as f:
            prompt_information = f.read()
            prompts_list.append(prompt_information)
            # Read the corresponding rubric from rubrics/{essay_set}.txt
            rubric_path = os.path.join("rubrics", f"{essay_set}.txt")
            with open(rubric_path, "r") as f:
                rubric = f.read()
            rubrics_list.append(rubric)
            task_directions_path = os.path.join("task-directions", f"{essay_set}.txt")
            with open(task_directions_path, "r") as f:
                task_directions = f.read()
            task_directions_list.append(task_directions)
    
    # NaN should be none for ints
    original_data = original_data.replace({float('nan'): None})

    trait_data = {}
    for path in glob.glob('trait-data/Prompt-*.csv'):
        # extract the essay_set number from the filename, e.g. Prompt-1.csv → 1
        fname = os.path.basename(path)
        essay_set = int(fname.split('-')[1].split('.')[0])
        
        # read and index by EssayID immediately
        df = pd.read_csv(path).set_index('EssayID')
        df.columns = df.columns.astype(str)  # Ensure column names are strings
        print(df.head())
        print(path, df.columns.tolist())
        trait_data[essay_set] = df

    current_outfile = "train.jsonl"
    data = []
    # Open the output jsonl file for writing
    with open(current_outfile, "w") as out_f:
        for idx, row in original_data.iterrows():
            essay_set = row['essay_set']
            # Skip other essay sets for now
            if essay_set not in [1, 2]:
                continue
            essay_text = row['essay']
            essay_id = row['essay_id']
            # print("essay_id", essay_id)
            domain1_score = int(row["domain1_score"])
            # Process essay set 1 by halving due to rubric mismatch
            if essay_set == 1:
                print("Changing score from", domain1_score, "to", math.ceil(domain1_score / 2))
                domain1_score = int(math.ceil(domain1_score / 2))
            domain2_score = int(row["domain2_score"]) if row["domain2_score"] is not None else None

            
            try:
                trait_scores_raw = trait_data[essay_set].loc[essay_id].to_dict()
                trait_scores = {str(k): v for k, v in trait_scores_raw.items()}
            except:
                print("essay_id", essay_id, "not found in trait_data")
                continue

            
            row = {
                "essay_id": essay_id,
                "prompt": prompt_information,
                "essay_text": essay_text,
                "task_directions": task_directions,
                "essay_set": essay_set,
                "score": domain1_score,
                "trait_scores": trait_scores,
                # "argument_annotation": argument_annotation_original_data[argument_annotation_original_data["essay_id"] == essay_id]["segmented_essays"].values[0]
            }
            data.append(row)
        # print(f"Generated {current_outfile}")
    df = pd.DataFrame(data)
    print(df.head()["trait_scores"])

    # Convert all columns to string type
    # This will change the trait_scores column from dicts to string representations of dicts
    # for column in df.columns:
        # if column == "trait_scores":
            # df[column] = df[column].apply(json.dumps)
        # else:
            # df[column] = df[column].astype(str)
    # print(df["trait_scores"])
    # for value in df["trait_scores"]:
    #     assert value.keys() == ["Content", "Organization", "Word Choice", "Sentence Fluency", "Conventions"]

    # Print the trait_scores for essay_id 4335
    for idx, row in df.iterrows():
        assert row["essay_text"] != ""
        assert row["prompt"] != ""
        assert row["task_directions"] != ""
        assert list(row["trait_scores"].keys()) == ["Content", "Organization", "Word Choice", "Sentence Fluency", "Conventions"]

    # Pick 30 samples from each essay set
    df1 = df[df["essay_set"] == 1]
    df2 = df[df["essay_set"] == 2]

    train_df = pd.concat([df1.sample(n=30), df2.sample(n=30)])
    # number of training examples
    n = len(train_df)

    # trait_scores_explanations should be a dict for each row:
    train_df["trait_explanations"] = [ {"Content": "", "Organization": "", "Word Choice": "", "Sentence Fluency": "", "Conventions": ""} for _ in range(n) ]

    # holistic_explanation should be a string for each row:
    train_df["holistic_explanation"] = [ "" for _ in range(n) ]
    print(train_df.columns.tolist())
    
   
    # Pick exactly 1 min score sample and 1 max score sample from each essay set, and ensure they are not in the samples
    df1_min = df1[df1["score"] == df1["score"].min()]
    df1_max = df1[df1["score"] == df1["score"].max()]
    df2_min = df2[df2["score"] == df2["score"].min()]
    df2_max = df2[df2["score"] == df2["score"].max()]
    
    # Ensure they are not in the samples and extract one
    df1_min = df1_min[~df1_min.index.isin(train_df.index)].head(1)
    df1_max = df1_max[~df1_max.index.isin(train_df.index)].head(1)
    df2_min = df2_min[~df2_min.index.isin(train_df.index)].head(1)
    df2_max = df2_max[~df2_max.index.isin(train_df.index)].head(1)

    # Few-show examples
    few_shot_examples_df = pd.concat([df1_min, df1_max, df2_min, df2_max])
   

    print(few_shot_examples_df.head())
    print(few_shot_examples_df.columns.tolist())

    trait_explanations_list = []
    holistic_explanations_list = []

    tmp_file = "generations.jsonl"

    outfile = open(tmp_file, "w")

    for idx, row in few_shot_examples_df.iterrows():
        essay_set_idx = row["essay_set"] - 1
        rubric = rubrics_list[essay_set_idx]
        score = row["score"]
        essay = row["essay_text"]
        prompt = prompts_list[essay_set_idx]
        task_directions = task_directions_list[essay_set_idx]
        trait_scores = row["trait_scores"]
        trait_scores_explanations = dict()
        trait_explanations_str = ""
        for aspect_idx, aspect in enumerate(trait_scores):
            formatted_prompt = explanation_prompt.format(rubric=aspect_rubrics[aspect_idx], score=score, essay=essay, prompt=prompt, task_directions=task_directions)
            print(formatted_prompt)
            print("--------------------------------")
            trait_scores_explanations[aspect] = load(essay_id, aspect)
            trait_explanations_str += "Aspect: " + aspect + "\n" + trait_scores_explanations[aspect] + "\n"
        
        holistic_formatted = holistic_explanation_prompt.format(scores_and_explanations=trait_explanations_str, score=score, essay=essay, prompt=prompt, task_directions=task_directions)
        holistic_explanation = load(essay_id, "holistic")
        print(holistic_explanation)
        print("--------------------------------")
        trait_explanations_list.append(trait_scores_explanations)
        holistic_explanations_list.append(holistic_explanation)

        outfile.write(json.dumps({"essay_id": row["essay_id"], "prompt": prompt, "essay_text": essay, "task_directions": task_directions, "essay_set": row["essay_set"], "score": score, "trait_scores": trait_scores, "trait_explanations": trait_scores_explanations, "holistic_explanation": holistic_explanation}))
        outfile.write("\n")
    
    few_shot_examples_df["trait_explanations"] = trait_explanations_list
    few_shot_examples_df["holistic_explanation"] = holistic_explanations_list

    train_df.to_csv("train.csv", index=False)
    few_shot_examples_df.to_csv("few_shot_examples.csv", index=False)

    few_shot_examples_ds = Dataset.from_pandas(few_shot_examples_df, preserve_index=False)
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)

    ddict = DatasetDict({
        "train" : train_ds,
        "shots" : few_shot_examples_ds
    })
    ddict.push_to_hub("jjordanoc/argumentative-asap-plus")
    
    print("Pushed to hub")

if __name__ == "__main__":
    main()
