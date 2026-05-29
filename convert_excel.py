import pandas as pd

# Read Excel
df = pd.read_excel("data/dataset_csv.xlsx")

# Rename columns to standard names
df.columns = ["prompt_id", "category", "prompt_text", "claude", "gemini", "gpt4o"]

# Convert to long format
records = []
for _, row in df.iterrows():
    for model_name, col in [("claude","claude"), ("gemini","gemini"), ("gpt4o","gpt4o")]:
        if row[col]:
            records.append({
                "prompt_id":     row["prompt_id"],
                "category":      row["category"],
                "prompt_text":   row["prompt_text"],
                "model_name":    model_name,
                "response_text": row[col]
            })

dataset = pd.DataFrame(records)
dataset.to_csv("data/dataset.csv", index=False)

print("Done!")
print("Shape:", dataset.shape)
print(dataset["model_name"].value_counts())