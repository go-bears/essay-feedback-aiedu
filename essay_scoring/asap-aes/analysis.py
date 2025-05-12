
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

output_str = """{"labels": ["human", "gemma-3-12b-it-baseline-with-argument-annotation", "gemma-3-12b-it-baseline-no-argument-annotation", "gemma-3-12b-it-orchestration-with-argument-annotation", "gemma-3-12b-it-orchestration-no-argument-annotation", "llama3.3-70b-it-baseline-with-argument-annotation", "llama3.3-70b-it-baseline-no-argument-annotation", "llama3.3-70b-it-orchestration-with-argument-annotation", "llama3.3-70b-it-orchestration-no-argument-annotation"], "matrix": [[0.0, 0.07317073170731707, 0.07142857142857142, 0.027777777777777776, 0.06060606060606061, 0.723404255319149, 0.7083333333333334, 0.4666666666666667, 0.6086956521739131], [0.926829268292683, 0.0, 0.4722222222222222, 0.125, 0.10714285714285714, 1.0, 1.0, 0.9743589743589743, 1.0], [0.9285714285714286, 0.5277777777777778, 0.0, 0.06060606060606061, 0.06896551724137931, 1.0, 1.0, 1.0, 0.975], [0.9722222222222222, 0.875, 0.9393939393939394, 0.0, 0.4074074074074074, 1.0, 1.0, 1.0, 1.0], [0.9393939393939394, 0.8928571428571429, 0.9310344827586207, 0.5925925925925926, 0.0, 1.0, 1.0, 1.0, 1.0], [0.2765957446808511, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44680851063829785, 0.18181818181818182, 0.2], [0.2916666666666667, 0.0, 0.0, 0.0, 0.0, 0.5531914893617021, 0.0, 0.35555555555555557, 0.2608695652173913], [0.5333333333333333, 0.02564102564102564, 0.0, 0.0, 0.0, 0.8181818181818182, 0.6444444444444445, 0.0, 0.5348837209302325], [0.391304347826087, 0.0, 0.025, 0.0, 0.0, 0.8, 0.7391304347826086, 0.46511627906976744, 0.0]]}"""
import textwrap

# — your data load —
data = json.loads(output_str)
matrix = np.array(data["matrix"])
labels = data["labels"]

for i in range(len(matrix)):
    matrix[i][i] = 0.5

# 1) Wrap each label after, say, 20 characters
wrapped_labels = [
    "\n".join(textwrap.wrap(lbl, width=20, break_long_words=False))
    for lbl in labels
]

# 2) Plot as before, but with the wrapped labels
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=wrapped_labels,
    yticklabels=wrapped_labels,
    cbar_kws={"shrink": 0.8},
    ax=ax
)

# 3) Move labels to the top, rotate at 45°, shrink font
ax.xaxis.tick_top()
ax.set_xticklabels(
    wrapped_labels,
    rotation=0,
    ha="center",
    fontsize=8
)
ax.set_yticklabels(
    wrapped_labels,
    rotation=0,
    va="center",
    fontsize=8
)

plt.tight_layout()
plt.show()


