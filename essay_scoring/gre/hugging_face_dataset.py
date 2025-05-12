
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

df = pd.read_csv("gre_data.tsv", delimiter="\t")

# df = df.astype(str)
print(df["aspect_1"])

print(df.dtypes)

dataset = Dataset.from_pandas(df)

dataset.push_to_hub("jjordanoc/gre-scoring-dataset")
