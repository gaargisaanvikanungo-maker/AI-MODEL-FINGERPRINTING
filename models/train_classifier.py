import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Load features ─────────────────────────────────────────────────────────
df = pd.read_csv("data/features.csv")

feature_cols = ['word_count', 'sentence_count', 'avg_word_length',
                'avg_sentence_length', 'type_token_ratio', 'bullet_count',
                'numbered_list_count', 'question_count', 'exclamation_count',
                'comma_rate', 'hedge_rate', 'filler_rate', 'empathy_rate',
                'paragraph_count', 'has_bold']

X = df[feature_cols]
y = df['model_name']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set:  {X_test.shape[0]} rows")
print(f"Models: {list(le.classes_)}\n")

# ── Train 3 classifiers ───────────────────────────────────────────────────
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":       XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
    "SVM":           SVC(kernel='rbf', random_state=42)
}

results = {}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y_encoded, cv=5)
    results[name] = {
        'model':    clf,
        'accuracy': acc,
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'y_pred':   y_pred
    }
    print(f"  Accuracy:          {acc:.3f} ({acc*100:.1f}%)")
    print(f"  Cross-val (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}\n")

# ── Best model ────────────────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['accuracy'])
best = results[best_name]
print(f"Best Model: {best_name} ({best['accuracy']*100:.1f}% accuracy)\n")

# ── Detailed report ───────────────────────────────────────────────────────
print("Classification Report:")
print(classification_report(y_test, best['y_pred'],
                             target_names=le.classes_))

# ── Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual Model')
plt.xlabel('Predicted Model')
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()
print("Confusion matrix saved to models/confusion_matrix.png")

# ── Feature importance (Random Forest) ───────────────────────────────────
rf = results["Random Forest"]["model"]
importance_df = pd.DataFrame({
    'feature':    feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
plt.title('Top 10 Features for Model Fingerprinting')
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()
print("Feature importance saved to models/feature_importance.png")

# ── Save best model ───────────────────────────────────────────────────────
import pickle
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({'model': best['model'], 'encoder': le,
                 'features': feature_cols, 'name': best_name}, f)
print(f"\nBest model saved to models/best_model.pkl")