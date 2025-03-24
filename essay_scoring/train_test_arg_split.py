import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "asap-aes/training_set_rel3.tsv"
df = pd.read_csv(file_path, sep="\t", encoding="ISO-8859-1")

# Filter for argumentative essays (essay_set in [1, 2, 3, 4, 5, 6])
argumentative_essays = df[df['essay_set'].isin([1, 2, 3, 4, 5, 6])]

# Stratified 80-10-10 split based on essay_set
train, temp = train_test_split(argumentative_essays, test_size=0.2, stratify=argumentative_essays['essay_set'], random_state=42)
test, val = train_test_split(temp, test_size=0.5, stratify=temp['essay_set'], random_state=42)

# Save only essay IDs
train[['essay_id']].to_csv("train_ids.csv", index=False)
test[['essay_id']].to_csv("test_ids.csv", index=False)
val[['essay_id']].to_csv("val_ids.csv", index=False)

print("Dataset split completed: train_ids.csv, test_ids.csv, val_ids.csv saved.")
