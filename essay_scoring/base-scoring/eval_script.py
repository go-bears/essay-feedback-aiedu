import os
import pandas as pd

BASE_DIR = "/home/kaiju/melissa_dev/ai-ed"
EVAL_DIR = os.path.join(BASE_DIR, "base-scoring-outputs/evals")
EVAL_FILE = os.path.join(EVAL_DIR, "compare_scores_all_2025-04-07-16-03.tsv")

df = pd.read_csv(EVAL_FILE, sep="\t")
idx_to_remove = []

for i, row in df.iterrows():
    try:
        row["domain1_score_pred_org"] = float(row["domain1_score_pred_org"])
        row["domain2_score_pred_org"] = float(row["domain2_score_pred_org"])
    except:

        idx_to_remove.append(i)

df.to_csv(EVAL_FILE, sep="\t", index=False)
df = df.drop( idx_to_remove)

df["domain1_score_pred_total"] = df["domain1_score_pred_org"] * 2
df["domain2_score_pred_total"] = df["domain2_score_pred_org"] * 2

df["mse_domain1"] = (df["domain1_score_pred_total"] - df["domain1_score_gt"]) ** 2
df["mse_domain2"] = (df["domain2_score_pred_total"] - df["domain2_score_gt"]) ** 2

mse_domain1 = df["mse_domain1"].mean()
mse_domain2 = df["mse_domain2"].mean()

print(f"MSE for domain 1: {mse_domain1}")
print(f"MSE for domain 2: {mse_domain2}")




