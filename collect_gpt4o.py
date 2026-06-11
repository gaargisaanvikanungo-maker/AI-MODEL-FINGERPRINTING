import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

github_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
)


def query_gpt4o(prompt):
    response = github_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# Test connection
print("Testing connection...")
test = query_gpt4o("Say hello in one sentence.")
print(f"Test: {test[:80]}...")
print("Connection successful!\n")

# Load dataset
dataset_path = "data/dataset.csv"
df = pd.read_csv(dataset_path)

print("Current dataset breakdown:")
print(df['model_name'].value_counts())
print()

# Find already collected gpt4o prompt IDs
already_done = df[df['model_name'] == 'gpt4o']['prompt_id'].tolist()
print(f"Already collected: {len(already_done)} gpt4o responses")

# Remove old gpt4o rows
df_no_gpt4o = df[df['model_name'] != 'gpt4o'].copy()

# Get all prompts from claude rows
all_prompts = df[df['model_name'] == 'claude'][
    ['prompt_id', 'category', 'prompt_text']
].copy()

# Keep only prompts NOT yet collected
remaining = all_prompts[
    ~all_prompts['prompt_id'].isin(already_done)
].copy()

print(f"Remaining to collect: {len(remaining)}")
print()

if len(remaining) == 0:
    print("All 200 gpt4o responses already collected!")
    exit()

# Collect responses
new_records = []
total = len(remaining)

for i, (_, row) in enumerate(remaining.iterrows()):
    print(f"[{i+1}/{total}] Prompt {row['prompt_id']} | {row['category']}...")

    try:
        response = query_gpt4o(row['prompt_text'])
        new_records.append({
            "prompt_id":     row['prompt_id'],
            "category":      row['category'],
            "prompt_text":   row['prompt_text'],
            "model_name":    "gpt-4o-mini",        # ✅ THIS WAS MISSING BEFORE
            "response_text": response
        })
        print(f"  OK: {response[:60]}...")

    except Exception as e:
        print(f"  ERROR: {e}")
        if new_records:
            temp = pd.concat(
                [df_no_gpt4o, pd.DataFrame(new_records)],
                ignore_index=True
            )
            temp.to_csv(dataset_path, index=False)
            print(f"  Saved {len(new_records)} responses before stopping")
        break

    # Auto save every 10 prompts
    if (i + 1) % 10 == 0:
        temp = pd.concat(
            [df_no_gpt4o, pd.DataFrame(new_records)],
            ignore_index=True
        )
        temp.to_csv(dataset_path, index=False)
        print(f"  Auto-saved at {i+1}")

    time.sleep(3)

# Final save
if new_records:
    final_df = pd.concat(
        [df_no_gpt4o, pd.DataFrame(new_records)],
        ignore_index=True
    )
    final_df.to_csv(dataset_path, index=False)
    print(f"\nDone! Collected {len(new_records)} responses")
    print("\nFinal breakdown:")
    print(final_df['model_name'].value_counts())
