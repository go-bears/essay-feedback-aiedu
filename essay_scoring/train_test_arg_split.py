import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "asap-aes/training_set_rel3.tsv"
df = pd.read_csv(file_path, sep="\t", encoding="ISO-8859-1")

# Filter out non argumentative essays and useless columns
argumentative_essays = df[df['essay_set'].isin([1, 2, 3, 4, 5, 6])]
cols_essay_set_2 = ['essay_id', 'essay_set', 'essay', 'domain1_score', 'domain2_score']
df_filtered = df.drop(columns=[col for col in df.columns if col not in cols_essay_set_2])

# Stratified 80-10-10 split
train, temp = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered['essay_set'], random_state=42)
test, val = train_test_split(temp, test_size=0.5, stratify=temp['essay_set'], random_state=42)

# Save splits
train.to_csv("argumentative-aes/train.csv", index=False)
test.to_csv("argumentative-aes/test.csv", index=False)
val.to_csv("argumentative-aes/val.csv", index=False)

print("Dataset split completed: train, test, validation saved as CSV files.")
