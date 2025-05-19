
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

output_str = """{"labels": ["human", "runs-2025-05-17-14-12-52-google-gemma-3-12b-it-baseline", "runs-2025-05-17-14-12-52-google-gemma-3-12b-it-orchestration-ablation_all", "runs-2025-05-17-14-12-52-google-gemma-3-27b-it-baseline", "runs-2025-05-17-14-12-52-google-gemma-3-27b-it-orchestration-ablation_all", "runs-2025-05-17-14-12-52-meta-llama-Llama-3.1-8B-Instruct-baseline", "runs-2025-05-17-14-12-52-meta-llama-Llama-3.1-8B-Instruct-orchestration-ablation_all", "runs-2025-05-17-14-12-52-meta-llama-Llama-3.3-70B-Instruct-baseline", "runs-2025-05-17-14-12-52-meta-llama-Llama-3.3-70B-Instruct-orchestration-ablation_all"], "average_matrix": [[0.5, 0.2374, 0.19119999999999998, 0.1834, 0.229, 0.5681999999999999, 0.6334, 0.7124, 0.6348], [0.7626, 0.5, 0.2222, 0.275, 0.4248, 0.9592, 0.8836, 0.9916, 0.9262], [0.8088000000000001, 0.7778, 0.5, 0.4934, 0.6176, 0.9315999999999999, 0.9400000000000001, 0.9865999999999999, 0.9907999999999999], [0.8166, 0.725, 0.5066, 0.5, 0.4917999999999999, 0.9544, 0.95, 0.9916, 0.974], [0.771, 0.5752, 0.38239999999999996, 0.5081999999999999, 0.5, 0.8954000000000001, 0.9334, 0.9666, 0.9654], [0.4318000000000001, 0.04080000000000002, 0.06840000000000002, 0.045599999999999995, 0.1046, 0.5, 0.6166, 0.6910000000000001, 0.5476000000000001], [0.3666, 0.11639999999999998, 0.059999999999999984, 0.05, 0.06659999999999999, 0.3834, 0.5, 0.7333999999999999, 0.4182], [0.2876, 0.008400000000000008, 0.01339999999999999, 0.008400000000000008, 0.033400000000000006, 0.309, 0.2666, 0.5, 0.33919999999999995], [0.3652, 0.07379999999999998, 0.009200000000000009, 0.026000000000000002, 0.034600000000000034, 0.4524, 0.5818, 0.6608, 0.5]], "majority_matrix": [[0.5, 0.10416666666666667, 0.022222222222222223, 0.0625, 0.0625, 0.45454545454545453, 0.6666666666666666, 0.7083333333333334, 0.6739130434782609], [0.8958333333333334, 0.5, 0.08888888888888889, 0.125, 0.3333333333333333, 1.0, 0.9166666666666666, 1.0, 0.9347826086956522], [0.9777777777777777, 0.9111111111111111, 0.5, 0.5555555555555556, 0.6666666666666666, 1.0, 1.0, 1.0, 1.0], [0.9375, 0.875, 0.4444444444444444, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], [0.9375, 0.6666666666666666, 0.3333333333333333, 0.5, 0.5, 1.0, 1.0, 1.0, 0.9782608695652174], [0.5454545454545454, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5833333333333334, 0.6363636363636364, 0.4523809523809524], [0.3333333333333333, 0.08333333333333333, 0.0, 0.0, 0.0, 0.4166666666666667, 0.5, 0.75, 0.36363636363636365], [0.2916666666666667, 0.0, 0.0, 0.0, 0.0, 0.36363636363636365, 0.25, 0.5, 0.2608695652173913], [0.32608695652173914, 0.06521739130434782, 0.0, 0.0, 0.021739130434782608, 0.5476190476190477, 0.6363636363636364, 0.7391304347826086, 0.5]]}"""
# output_str = """{"labels": ["human", "gemma-3-12b-it-baseline-with-argument-annotation", "gemma-3-12b-it-baseline-no-argument-annotation", "gemma-3-12b-it-orchestration-with-argument-annotation", "gemma-3-12b-it-orchestration-no-argument-annotation", "llama3.3-70b-it-baseline-with-argument-annotation", "llama3.3-70b-it-baseline-no-argument-annotation", "llama3.3-70b-it-orchestration-with-argument-annotation", "llama3.3-70b-it-orchestration-no-argument-annotation"], "matrix": [[0.0, 0.07317073170731707, 0.07142857142857142, 0.027777777777777776, 0.06060606060606061, 0.723404255319149, 0.7083333333333334, 0.4666666666666667, 0.6086956521739131], [0.926829268292683, 0.0, 0.4722222222222222, 0.125, 0.10714285714285714, 1.0, 1.0, 0.9743589743589743, 1.0], [0.9285714285714286, 0.5277777777777778, 0.0, 0.06060606060606061, 0.06896551724137931, 1.0, 1.0, 1.0, 0.975], [0.9722222222222222, 0.875, 0.9393939393939394, 0.0, 0.4074074074074074, 1.0, 1.0, 1.0, 1.0], [0.9393939393939394, 0.8928571428571429, 0.9310344827586207, 0.5925925925925926, 0.0, 1.0, 1.0, 1.0, 1.0], [0.2765957446808511, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44680851063829785, 0.18181818181818182, 0.2], [0.2916666666666667, 0.0, 0.0, 0.0, 0.0, 0.5531914893617021, 0.0, 0.35555555555555557, 0.2608695652173913], [0.5333333333333333, 0.02564102564102564, 0.0, 0.0, 0.0, 0.8181818181818182, 0.6444444444444445, 0.0, 0.5348837209302325], [0.391304347826087, 0.0, 0.025, 0.0, 0.0, 0.8, 0.7391304347826086, 0.46511627906976744, 0.0]]}"""
import textwrap


def transform_fun(labels):
    result = []
    for label in labels:
        curr_label = ""
        if "gemma" in label:
            if "12b" in label:
                curr_label = "gemma-12b"
            elif "27b" in label:
                curr_label = "gemma-27b"
            else:
                curr_label = "gemma"
        elif "llama" in label:
            if "3.1" in label:
                curr_label = "llama-8b"
            elif "3.3" in label:
                curr_label = "llama-70b"
            else:
                curr_label = "llama"
        else:
            curr_label = "human"
        if "orchestration" in label:
            curr_label += " MAGIC"
        result.append(curr_label)
    return result

# — your data load —
data = json.loads(output_str)
matrix = np.array(data["average_matrix"])
# labels = [
#   "human",
#   "gemma arg.",
#   "gemma",
#   "gemma MAGIC & arg.",
#   "gemma MAGIC",
#   "llama arg.",
#   "llama",
#   "llama MAGIC & arg.",
#   "llama MAGIC",
# ]
labels = transform_fun(data["labels"])

for i in range(len(matrix)):
    matrix[i][i] = 0.5



# ── Figure + Axes ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))

# ── Draw heatmap ─────────────────────────────────────────────────────────────
cax = ax.imshow(matrix, cmap="PiYG", interpolation="nearest", aspect="equal")


# ── Annotate each cell with its value ─────────────────────────────────────────
# pick a contrasting text color automatically
# thresh = 0.5
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        color = "white" if val <= 0.2 or val >= 0.8 else "black"
        ax.text(
            j, i,                       # x=j, y=i
            f"{val:.2f}",               # formatted number
            ha="center", va="center",
            color=color,
            fontsize=20
        )


# ── Move ticks to top ─────────────────────────────────────────────────────────
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

# ── Set tick positions & labels ───────────────────────────────────────────────
n = len(labels)
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# ── Style tick labels ─────────────────────────────────────────────────────────
ax.tick_params(
    axis="x",
    labelsize=20,
    labeltop=True,
    labelbottom=False,
    # pad=100            # ← give 10 points of extra spacing *above* the spine
)
ax.tick_params(axis="y", labelsize=20)

# ── Rotate + align x-labels in place ─────────────────────────────────────────
for lbl in ax.get_xticklabels():
    lbl.set_rotation(45)
    lbl.set_ha("left")
    lbl.set_rotation_mode("anchor")

# ── Colorbar + layout ───────────────────────────────────────────────────────
fig.colorbar(cax, ax=ax, shrink=0.8)

# Optionally push out the top margin a bit more:
fig.subplots_adjust(top=0.88)

plt.tight_layout()
plt.show()