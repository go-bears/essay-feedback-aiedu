import os
from datetime import datetime
import pandas as pd

BASE_DIR = "/home/kaiju/melissa_dev/ai-ed"
EVAL_DIR = os.path.join(BASE_DIR, "base-scoring-outputs/evals")
EVAL_FILE = os.path.join(EVAL_DIR, "compare_scores_all_2025-04-07-16-03.tsv")
date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")

df = pd.read_csv(EVAL_FILE, sep="\t")
df.fillna(0, inplace=True)
idx_to_remove = []

for i, row in df.iterrows():
    try:
        row["domain1_score_pred_org"] = float(row["domain1_score_pred_org"])
        row["domain2_score_pred_org"] = float(row["domain2_score_pred_org"])
        row["domain1_score_gt"] = float(row["domain1_score_gt"])
        row["domain2_score_gt"] = float(row["domain2_score_gt"])
    except:
        idx_to_remove.append(i)

eval_df = df.drop(idx_to_remove)

eval_df["domain1_score_pred_total"] = eval_df["domain1_score_pred_org"].astype(float) * 2
eval_df["domain2_score_pred_total"] = eval_df["domain2_score_pred_org"].astype(float) * 2
eval_df["domain1_score_gt"] = eval_df["domain1_score_gt"].astype(float)
eval_df["domain2_score_gt"] = eval_df["domain2_score_gt"].astype(float)


eval_df["mse_domain1"] = (eval_df["domain1_score_pred_total"] - eval_df["domain1_score_gt"]) ** 2
eval_df["mse_domain2"] = (eval_df["domain2_score_pred_total"] - eval_df["domain2_score_gt"]) ** 2
mse_domain1 = eval_df["mse_domain1"].sum() / len(eval_df)
mse_domain2 = eval_df["mse_domain2"].sum() / len(eval_df)

print(f"MSE for domain 1: {mse_domain1}")
print(f"MSE for domain 2: {mse_domain2}")
eval_df.to_csv(os.path.join(EVAL_DIR, f"calcuated_eval_{date_str}.tsv"), sep="\t", index=False)




