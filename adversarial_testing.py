# adversarial_testing.py  — GROQ VERSION (fixed for your CSV columns)
# Run: python adversarial_testing.py

import pandas as pd
import numpy as np
import pickle
import time
import json
import os
from groq import Groq
from sklearn.metrics import accuracy_score

# ── CONFIG ──────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()  # loads the .env file
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)
GROQ_MODEL = "llama-3.1-8b-instant"

# ── LOAD YOUR MODEL ──────────────────────────────────────────────────────────
with open("models/best_model.pkl", "rb") as f:
    model_data = pickle.load(f)

clf      = model_data["model"]
le       = model_data["encoder"]
features = model_data["features"]

from features.extract_features import extract_features

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/dataset.csv")

# Verify columns
print("Columns:", df.columns.tolist())
print("Model names found:", df["model_name"].unique())

frames = []
for model in df["model_name"].unique():
    subset = df[df["model_name"] == model].sample(20, random_state=42)
    frames.append(subset)
test_samples = pd.concat(frames).reset_index(drop=True)

print(f"\nTest samples: {len(test_samples)}")
print(test_samples["model_name"].value_counts())
print(f"\nTest samples: {len(test_samples)}")
print(test_samples["model_name"].value_counts())

# ── HELPER: Call Groq with retry ─────────────────────────────────────────────
def call_groq(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7
            )
            time.sleep(0.5)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Groq error attempt {attempt+1}: {e}")
            time.sleep(10)
    return None

# ── HELPER: Extract features from a list of texts ────────────────────────────
def get_features(texts):
    feats = []
    for text in texts:
        f = extract_features(str(text))
        feats.append([f.get(feat, 0) for feat in features] if f else [0] * len(features))
    return np.array(feats)

# ── SAVE FILE PATHS ───────────────────────────────────────────────────────────
SAVE_FILE_PARA  = "data/paraphrased_samples.csv"
SAVE_FILE_STYLE = "data/style_transfer_samples.csv"

# ══════════════════════════════════════════════════════════════════════════════
# ATTACK 1: PARAPHRASE ATTACK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ATTACK 1: PARAPHRASE ATTACK")
print("="*60)

PARAPHRASE_PROMPT = """Rewrite the following text completely in different words while preserving the exact meaning.
RULES:
- Change ALL sentence structures
- Use completely different vocabulary
- Vary sentence lengths
- Do NOT add new information
- Output ONLY the rewritten text, nothing else

TEXT:
{text}"""

# Resume if partially done
if os.path.exists(SAVE_FILE_PARA):
    paraphrased_df_existing = pd.read_csv(SAVE_FILE_PARA)
    done_count = len(paraphrased_df_existing)
    paraphrased_texts = paraphrased_df_existing.to_dict("records")
    print(f"Resuming — already done {done_count} samples")
else:
    paraphrased_texts = []
    done_count = 0

for i, row in test_samples.iterrows():
    if i < done_count:
        continue

    print(f"  [{i+1}/{len(test_samples)}] Paraphrasing {row['model_name']}...")
    prompt = PARAPHRASE_PROMPT.format(text=str(row["response_text"])[:1200])
    rewritten = call_groq(prompt)

    if rewritten:
        paraphrased_texts.append({
            "original_text":   row["response_text"],
            "paraphrased_text": rewritten,
            "true_model":      row["model_name"]
        })
        # Save after every sample so you never lose progress
        pd.DataFrame(paraphrased_texts).to_csv(SAVE_FILE_PARA, index=False)
        print(f"    ✓ Done")
    else:
        print(f"    ✗ FAILED — skipping")

paraphrased_df = pd.read_csv(SAVE_FILE_PARA)
print(f"\nTotal paraphrased samples saved: {len(paraphrased_df)}")

# ── Evaluate original vs paraphrased ─────────────────────────────────────────
print("\nEvaluating original accuracy...")
X_orig = get_features(paraphrased_df["original_text"])
y_true = le.transform(paraphrased_df["true_model"])
acc_original = accuracy_score(y_true, clf.predict(X_orig))

print("Evaluating post-paraphrase accuracy...")
X_para = get_features(paraphrased_df["paraphrased_text"])
acc_paraphrased = accuracy_score(y_true, clf.predict(X_para))

print(f"\n📊 PARAPHRASE ATTACK RESULTS:")
print(f"   Original accuracy:    {acc_original*100:.1f}%")
print(f"   Post-paraphrase:      {acc_paraphrased*100:.1f}%")
print(f"   Accuracy drop:        {(acc_original - acc_paraphrased)*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ATTACK 2: STYLE TRANSFER ATTACK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ATTACK 2: STYLE TRANSFER ATTACK")
print("="*60)

STYLE_PROMPT = """Rewrite the following text so it sounds like it was written by {target}.

Style characteristics of {target}:
- GPT-4o: heavy bullet points, bold headers, very confident tone, zero hedging
- Claude: uses em-dashes frequently, says 'I think' and 'perhaps', longer thoughtful paragraphs
- Gemini: conversational tone, numbered lists, adds helpful background context
- LLaMA: informal tone, high vocabulary variety, less structured
- Mistral: concise, technical, minimal filler words

RULES:
- Preserve the core information
- Strongly adopt {target}'s writing style
- Output ONLY the rewritten text, nothing else

TARGET STYLE: {target}
TEXT:
{text}"""

ALL_MODELS = ["GPT-4o", "Claude", "Gemini", "LLaMA", "Mistral"]

# Map your CSV model names to display names for the prompt
MODEL_DISPLAY = {
    "claude":  "Claude",
    "gemini":  "Gemini",
    "gpt4o":   "GPT-4o",
    "llama":   "LLaMA",
    "mistral": "Mistral"
}

# Resume if partially done
if os.path.exists(SAVE_FILE_STYLE):
    style_df_existing = pd.read_csv(SAVE_FILE_STYLE)
    done_st_count = len(style_df_existing)
    style_texts = style_df_existing.to_dict("records")
    print(f"Resuming — already done {done_st_count} samples")
else:
    style_texts = []
    done_st_count = 0

for i, row in test_samples.iterrows():
    if i < done_st_count:
        continue

    # Pick a DIFFERENT model as the target style
    current_display = MODEL_DISPLAY.get(row["model_name"].lower(), "GPT-4o")
    targets = [m for m in ALL_MODELS if m != current_display]
    target = targets[i % len(targets)]

    print(f"  [{i+1}/{len(test_samples)}] {row['model_name']} → {target} style...")
    prompt = STYLE_PROMPT.format(target=target, text=str(row["response_text"])[:1200])
    rewritten = call_groq(prompt)

    if rewritten:
        style_texts.append({
            "original_text":    row["response_text"],
            "transferred_text": rewritten,
            "true_model":       row["model_name"],
            "target_style":     target
        })
        pd.DataFrame(style_texts).to_csv(SAVE_FILE_STYLE, index=False)
        print(f"    ✓ Done")
    else:
        print(f"    ✗ FAILED — skipping")

style_df = pd.read_csv(SAVE_FILE_STYLE)
print(f"\nTotal style transfer samples saved: {len(style_df)}")

# ── Evaluate style transfer ───────────────────────────────────────────────────
print("\nEvaluating post-style-transfer accuracy...")
X_orig2    = get_features(style_df["original_text"])
X_transfer = get_features(style_df["transferred_text"])
y_true2    = le.transform(style_df["true_model"])

acc_orig2    = accuracy_score(y_true2, clf.predict(X_orig2))
acc_transfer = accuracy_score(y_true2, clf.predict(X_transfer))

print(f"\n📊 STYLE TRANSFER ATTACK RESULTS:")
print(f"   Original accuracy:      {acc_orig2*100:.1f}%")
print(f"   Post-style-transfer:    {acc_transfer*100:.1f}%")
print(f"   Accuracy drop:          {(acc_orig2 - acc_transfer)*100:.1f}%")

# ── SAVE FINAL RESULTS ────────────────────────────────────────────────────────
results = {
    "original_accuracy":            round(acc_original * 100, 1),
    "post_paraphrase_accuracy":     round(acc_paraphrased * 100, 1),
    "post_style_transfer_accuracy": round(acc_transfer * 100, 1),
    "paraphrase_drop":              round((acc_original - acc_paraphrased) * 100, 1),
    "style_transfer_drop":          round((acc_orig2 - acc_transfer) * 100, 1)
}

with open("data/adversarial_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✅ ALL DONE!")
print("="*60)
print(json.dumps(results, indent=2))
print("\nFiles saved:")
print("  data/paraphrased_samples.csv")
print("  data/style_transfer_samples.csv")
print("  data/adversarial_results.json")