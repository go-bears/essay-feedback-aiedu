import json

data = json.load(open("judge_tmp.json"))

outputs = data["summary_outputs"]
decisions = data["results"]

c1_winsllm = 0

for idx, reasoning, decision in zip(range(len(outputs)), outputs, decisions):
    print(f"Essay {idx}")
    print(f"Reasoning: {reasoning}")
    print(f"Decision: {decision}")
    print("-" * 100)
    if decision["c1"] == 1:
        c1_winsllm += 1
    # if idx == 3:
    #     break

print(f"C1 wins: {c1_winsllm}")

