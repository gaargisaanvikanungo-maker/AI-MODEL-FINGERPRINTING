import pandas as pd
import re

# Load dataset
df = pd.read_csv("data/dataset.csv")

def extract_features(text):
    if not isinstance(text, str) or text == "ERROR":
        return None

    # ── Basic counts ──────────────────────────────
    words = text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # ── Lexical features ──────────────────────────
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0

    # ── Punctuation & formatting ──────────────────
    bullet_count = len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE))
    numbered_list_count = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
    question_count = text.count('?')
    exclamation_count = text.count('!')
    comma_rate = text.count(',') / word_count if word_count > 0 else 0

    # ── Hedge words ───────────────────────────────
    hedge_words = ['perhaps', 'might', 'could', 'may', 'possibly',
                   'likely', 'probably', 'generally', 'sometimes',
                   'often', 'usually', 'typically', 'suggest']
    hedge_count = sum(1 for w in words if w.lower() in hedge_words)
    hedge_rate = hedge_count / word_count if word_count > 0 else 0

    # ── Filler words ──────────────────────────────
    filler_words = ['certainly', 'absolutely', 'definitely', 'indeed',
                    'actually', 'basically', 'essentially', 'importantly',
                    'specifically', 'particularly', 'furthermore', 'moreover',
                    'however', 'therefore', 'additionally']
    filler_count = sum(1 for w in words if w.lower() in filler_words)
    filler_rate = filler_count / word_count if word_count > 0 else 0

    # ── Empathy words ─────────────────────────────
    empathy_words = ['feel', 'feeling', 'emotions', 'understand',
                     'support', 'difficult', 'hard', 'normal',
                     'okay', 'help', 'care', 'remember']
    empathy_count = sum(1 for w in words if w.lower() in empathy_words)
    empathy_rate = empathy_count / word_count if word_count > 0 else 0

    # ── Structure ─────────────────────────────────
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    has_bold = 1 if '**' in text else 0

    return {
        'word_count':           word_count,
        'sentence_count':       sentence_count,
        'avg_word_length':      round(avg_word_length, 3),
        'avg_sentence_length':  round(avg_sentence_length, 3),
        'type_token_ratio':     round(type_token_ratio, 3),
        'bullet_count':         bullet_count,
        'numbered_list_count':  numbered_list_count,
        'question_count':       question_count,
        'exclamation_count':    exclamation_count,
        'comma_rate':           round(comma_rate, 3),
        'hedge_rate':           round(hedge_rate, 3),
        'filler_rate':          round(filler_rate, 3),
        'empathy_rate':         round(empathy_rate, 3),
        'paragraph_count':      paragraph_count,
        'has_bold':             has_bold,
    }

# Extract features for all rows
print("Extracting features...")
feature_rows = []

for _, row in df.iterrows():
    features = extract_features(row['response_text'])
    if features:
        features['prompt_id']  = row['prompt_id']
        features['category']   = row['category']
        features['model_name'] = row['model_name']
        feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)
features_df.to_csv("data/features.csv", index=False)

print(f"Done! Extracted {len(features_df)} rows with {len(features_df.columns)} columns")
print("\nFeature columns:")
print([c for c in features_df.columns if c not in ['prompt_id','category','model_name']])
print("\nAverage features per model:")
print(features_df.groupby('model_name')[['avg_word_length','avg_sentence_length','hedge_rate','filler_rate']].mean().round(3))