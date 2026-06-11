# upload_huggingface.py
import os
from huggingface_hub import login
from datasets import Dataset, DatasetDict
import pandas as pd

df = pd.read_csv("data/dataset.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

dataset = Dataset.from_pandas(df)
split = dataset.train_test_split(test_size=0.3, seed=42)
val_test = split["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    "train":      split["train"],
    "validation": val_test["train"],
    "test":       val_test["test"]
})

print(f"Train: {len(dataset_dict['train'])}")
print(f"Val:   {len(dataset_dict['validation'])}")
print(f"Test:  {len(dataset_dict['test'])}")

# REPLACE with your actual HuggingFace username

# Step 1: Login with your token
login(token = os.getenv("HF_TOKEN"))

# Step 2: Push with your actual username
dataset_dict.push_to_hub("fdvfvdvdvqw/ai-model-fingerprinting-dataset")
#                          ^^^ replace this with YOUR actual HuggingFace username
print("✅ Done! Check huggingface.co/datasets/YOUR_HF_USERNAME/ai-model-fingerprinting-dataset")
