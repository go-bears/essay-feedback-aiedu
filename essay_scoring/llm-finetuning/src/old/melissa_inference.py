# # Melissa's Llama 3.2 3B inference
# def analysis_iter0_local(
#     run_folder: str,
#     model_name: str,
#     backend: Literal["ollama", "vllm"] = "ollama",
#     local_dataset_path: str = "",
# ):
#     from datasets import load_dataset
#     import time
#     import pandas as pd

#     results: ASAPResults = {}
#     ground_truths: ASAPResults = {}
#     raw_outputs = open(os.path.join(run_folder, "raw_outputs.txt"), "w")
#     none_count = 0

#     eval_dataset = load_dataset("jjordanoc/argumentative-asap", split="test")

#     train_dataset = load_dataset("jjordanoc/argumentative-asap", split="train")
#     train_df = pd.DataFrame(train_dataset)

#     n = min(len(eval_dataset), limit) if limit is not None else len(eval_dataset)
#     times = np.zeros((n + 1, 1), dtype=np.float32)

#     if not remote_job:
#         import pandas as pd

#         df = pd.read_csv(local_dataset_path, sep="\t", encoding="ISO-8859-1", dtype=str)

#     # Initialize the inference engine
#     inference = UnifiedInference(
#         backend=backend, model_name=model_name, adapters_repo=adapters_repo
#     )

#     for idx, grading_instruction in enumerate(eval_dataset):
#         print("*" * 120)
#         print("Processing essay", idx)
#         print("=" * 80)

#         essay_set = int(grading_instruction["essay_set"])

#         content: Optional[str] = None
#         if remote_job:
#             if adapters_repo != "":
#                 full_prompt = format_prompt_inference_ft(grading_instruction)
#             else:
#                 full_prompt = format_prompt_inference_iter1(
#                     grading_instruction, few_shot_n, add_argument_annotation, train_df
#                 )
#             # Use the unified inference interface
#             start_time = time.time()
#             content = inference.generate(full_prompt)
#             times[idx] = (time.time() - start_time) * 1000  # Convert to milliseconds
#         else:
#             full_prompt = ""
#             content = (
#                 df[df["essay_id"] == (grading_instruction["essay_id"])]["comments"]
#             ).values[0]

#         out_str = (
#             "=" * 30
#             + f"Interaction {idx}"
#             + "=" * 30
#             + "\nPrompt:\n"
#             + full_prompt
#             + "\n\n"
#             + "Response:\n"
#             + content
#             + "\n\n"
#         )

#         # Log the output
#         raw_outputs.write(out_str)

#         print("=" * 80)
#         print("Answer:")

#         try:
#             score_1 = extract_domain_score(content, 1)
#             score_2 = -1
#             print(score_1)

#             if essay_set == 2:
#                 score_2 = extract_domain_score(content, 2)
#                 print(score_2)
#             score = (score_1, score_2)
#         except Exception as e:
#             # skip this essay
#             print(e)
#             continue

#         # Discard from analysis
#         if score_1 is None or score_2 is None:
#             none_count += 1
#             continue

#         grader_score_1 = -1
#         grader_score_2 = -1
#         if essay_set == 2:
#             split = grading_instruction["grader_score"].split(" ")
#             grader_score_1 = int(split[0])
#             grader_score_2 = int(split[1])
#         else:
#             grader_score_1 = int(grading_instruction["grader_score"])
#         grader_score = (grader_score_1, grader_score_2)

#         if essay_set not in results:
#             results[essay_set] = [score]
#             ground_truths[essay_set] = [grader_score]
#         else:
#             results[essay_set].append(score)
#             ground_truths[essay_set].append(grader_score)

#         print("*" * 120)

#         if idx == n:
#             break

#     # Store data in a traceable format
#     output = {
#         "qwk_summary": compute_kappa_summary(ground_truths, results),
#         "predicted_labels": results,
#         "ground_truths": ground_truths,
#         "avg_time_ms": float(np.average(times)),
#         "sample_size": n,
#         "none_count": none_count,
#     }
#     outfile = open(os.path.join(run_folder, "run.json"), "w")
#     json.dump(output, outfile)
#     outfile.close()
#     raw_outputs.close()
#     tmp_outs.close()

# def local_main():
#     run_folder = "../local_runs"
#     inference_loop(
#         run_folder,
#         remote_job=False,
#         local_dataset_path="/Users/joaquin/Desktop/ai_education/essay-feedback-aiedu/essay_scoring/final_llama3.2-scoring-output-2025-04-09-12-19.tsv",
#     )
# if __name__ == "__main__":
#     local_main()
