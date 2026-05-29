import streamlit as st
import pandas as pd
import pickle
import re

# ── Load saved model ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(BASE_DIR, "models", "best_model.pkl"), "rb") as f:
        return pickle.load(f)

model_data = load_model()
clf        = model_data['model']
le         = model_data['encoder']
features   = model_data['features']
model_name = model_data['name']

# ── Feature extraction function ───────────────────────────────────────────
def extract_features(text):
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return None

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    avg_word_length     = sum(len(w) for w in words) / word_count
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    unique_words        = set(w.lower() for w in words)
    type_token_ratio    = len(unique_words) / word_count

    bullet_count        = len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE))
    numbered_list_count = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
    question_count      = text.count('?')
    exclamation_count   = text.count('!')
    comma_rate          = text.count(',') / word_count

    hedge_words  = ['perhaps','might','could','may','possibly','likely',
                    'probably','generally','sometimes','often','usually',
                    'typically','suggest']
    hedge_rate   = sum(1 for w in words if w.lower() in hedge_words) / word_count

    filler_words = ['certainly','absolutely','definitely','indeed','actually',
                    'basically','essentially','importantly','specifically',
                    'particularly','furthermore','moreover','however',
                    'therefore','additionally']
    filler_rate  = sum(1 for w in words if w.lower() in filler_words) / word_count

    empathy_words = ['feel','feeling','emotions','understand','support',
                     'difficult','hard','normal','okay','help','care','remember']
    empathy_rate  = sum(1 for w in words if w.lower() in empathy_words) / word_count

    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    has_bold        = 1 if '**' in text else 0

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

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Model Fingerprinter",
    page_icon="🔍",
    layout="centered"
)

# ── UI ────────────────────────────────────────────────────────────────────
st.title("🔍 AI Model Fingerprinter")
st.markdown("Paste any AI-generated text below and find out **which AI model wrote it!**")
st.markdown("---")

text_input = st.text_area(
    "Paste AI-generated text here:",
    height=200,
    placeholder="Paste any response from an AI model here..."
)

if st.button("🔍 Identify Model", type="primary"):
    if not text_input.strip():
        st.warning("Please paste some text first!")
    elif len(text_input.split()) < 10:
        st.warning("Please paste a longer text (at least 10 words) for accurate results.")
    else:
        features_dict = extract_features(text_input)
        if features_dict:
            X = pd.DataFrame([features_dict])[features]
            prediction_encoded = clf.predict(X)[0]
            prediction         = le.inverse_transform([prediction_encoded])[0]
            probabilities      = clf.predict_proba(X)[0]
            confidence         = max(probabilities) * 100

            # Model display names
            model_display = {
                'claude':  '🟣 Claude (Anthropic)',
                'gemini':  '🔵 Gemini (Google)',
                'gpt4o':   '🟢 GPT-4o (OpenAI)',
                'llama':   '🟡 LLaMA (Meta)',
                'mistral': '🔴 Mistral AI'
            }

            st.success(f"### Prediction: {model_display.get(prediction, prediction)}")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.markdown("---")

            # Show all probabilities
            st.markdown("#### Probability breakdown:")
            prob_df = pd.DataFrame({
                'Model':       [model_display.get(le.classes_[i], le.classes_[i])
                                for i in range(len(le.classes_))],
                'Probability': [f"{p*100:.1f}%" for p in probabilities]
            }).sort_values('Probability', ascending=False)

            st.dataframe(prob_df, hide_index=True, use_container_width=True)

            # Show extracted features
            with st.expander("📊 View extracted features"):
                feat_df = pd.DataFrame([features_dict]).T
                feat_df.columns = ['Value']
                st.dataframe(feat_df, use_container_width=True)

st.markdown("---")
st.markdown("**How it works:** The app extracts 15 linguistic features from your text and uses a Random Forest classifier trained on 1000 AI responses to predict which model wrote it.")