import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()

groq_client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

def query_groq(prompt):
    r = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500, temperature=0.7)
    return r.choices[0].message.content.strip()

def query_mistral(prompt):
    r = mistral_client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(role="user", content=prompt)],
        max_tokens=500, temperature=0.7)
    return r.choices[0].message.content.strip()

# Load existing dataset
existing = pd.read_csv("data/dataset.csv")
prompts_df = existing[["prompt_id","category","prompt_text"]].drop_duplicates()

new_records = []

for _, row in prompts_df.iterrows():
    for model_name, query_fn in [("llama", query_groq), ("mistral", query_mistral)]:
        print(f"Querying {model_name} | prompt {row['prompt_id']}...")
        try:
            response = query_fn(row["prompt_text"])
            new_records.append({
                "prompt_id":     row["prompt_id"],
                "category":      row["category"],
                "prompt_text":   row["prompt_text"],
                "model_name":    model_name,
                "response_text": response
            })
        except Exception as e:
            print(f"  ERROR — {model_name} prompt {row['prompt_id']}: {e}")
            new_records.append({
                "prompt_id":     row["prompt_id"],
                "category":      row["category"],
                "prompt_text":   row["prompt_text"],
                "model_name":    model_name,
                "response_text": "ERROR"
            })
        time.sleep(1)

    # Save after every prompt
    combined = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
    combined.to_csv("data/dataset.csv", index=False)
    print(f"  ✅ Saved after prompt {row['prompt_id']}")

print("\n🎉 Done!")
print(pd.read_csv("data/dataset.csv")["model_name"].value_counts())