# shap_analysis.py — FIXED for your setup
# Run: python shap_analysis.py

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
with open("models/best_model.pkl", "rb") as f:
    model_data = pickle.load(f)

clf      = model_data["model"]
le       = model_data["encoder"]
features = model_data["features"]
model_names = list(le.classes_)   # ['claude', 'gemini', 'gpt4o', 'llama', 'mistral']

print("Models:", model_names)
print("Features:", features)

from features.extract_features import extract_features

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/dataset.csv")

print("Extracting features from dataset...")
rows = []
valid_indices = []
for i, row in df.iterrows():
    f = extract_features(row["response_text"])
    if f:
        rows.append([f.get(feat, 0) for feat in features])
        valid_indices.append(i)

X = np.array(rows)
y = le.transform(df.loc[valid_indices, "model_name"].values)
print(f"X shape: {X.shape}")

# ── COMPUTE SHAP VALUES ───────────────────────────────────────────────────────
print("Computing SHAP values (1-2 minutes)...")
explainer = shap.TreeExplainer(clf)
shap_output = explainer.shap_values(X)

# Handle both old shape [n_classes][n_samples, n_features]
# and new shape (n_samples, n_features, n_classes)
shap_array = np.array(shap_output)
print("Raw SHAP shape:", shap_array.shape)

if shap_array.ndim == 3 and shap_array.shape[2] == len(model_names):
    # New format: (n_samples, n_features, n_classes) → convert to list of (n_samples, n_features)
    shap_values = [shap_array[:, :, i] for i in range(len(model_names))]
elif shap_array.ndim == 3 and shap_array.shape[0] == len(model_names):
    # Old format: (n_classes, n_samples, n_features)
    shap_values = [shap_array[i] for i in range(len(model_names))]
else:
    shap_values = shap_output

print(f"SHAP values per class shape: {shap_values[0].shape}")
print(f"Number of classes: {len(shap_values)}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Overall Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Plot 1: Overall feature importance...")

# Average absolute SHAP across all classes
all_abs_shap = np.stack([np.abs(shap_values[i]) for i in range(len(model_names))], axis=0)
# shape: (n_classes, n_samples, n_features)
mean_importance = all_abs_shap.mean(axis=(0, 1))   # shape: (n_features,)

print(f"mean_importance shape: {mean_importance.shape}")   # should be (15,)
print(f"features length: {len(features)}")                 # should be 15

importance_df = pd.DataFrame({
    "feature":    features,
    "importance": mean_importance
}).sort_values("importance", ascending=True)

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
plt.barh(importance_df["feature"], importance_df["importance"], color=colors)
plt.xlabel("Mean |SHAP Value|", fontsize=13)
plt.title("Feature Importance Across All Models\n(higher = more discriminative)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("models/shap_overall_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: models/shap_overall_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Per-Model Top Features
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Plot 2: Per-model top features...")

model_colors = {
    "claude":  "#d4a017",
    "gemini":  "#4285f4",
    "gpt4o":   "#10a37f",
    "llama":   "#e74c3c",
    "mistral": "#9b59b6"
}

fig, axes = plt.subplots(1, len(model_names), figsize=(22, 7))
fig.suptitle("Top 8 Features Per Model (SHAP Analysis)", fontsize=16, fontweight="bold")

top_features_per_model = {}

for idx, name in enumerate(model_names):
    sv       = shap_values[idx]              # (n_samples, n_features)
    mean_sv  = sv.mean(axis=0)              # (n_features,) — direction
    abs_sv   = np.abs(sv).mean(axis=0)     # (n_features,) — importance

    top_idx  = np.argsort(abs_sv)[-8:]
    top_feat = [features[i] for i in top_idx]
    top_vals = [mean_sv[i]  for i in top_idx]
    top_imp  = [abs_sv[i]   for i in top_idx]

    top_features_per_model[name] = list(zip(top_feat, top_vals, top_imp))

    color      = model_colors.get(name, "#666666")
    bar_colors = [color if v > 0 else "#cccccc" for v in top_vals]

    axes[idx].barh(top_feat, top_imp, color=bar_colors)
    axes[idx].set_title(name.upper(), fontsize=13, fontweight="bold", color=color)
    axes[idx].set_xlabel("Mean |SHAP|", fontsize=9)
    axes[idx].tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("models/shap_per_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: models/shap_per_model.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Plot 3: Confusion matrix...")

from sklearn.metrics import confusion_matrix

y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred)
cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm_percent,
    annot=True, fmt=".1f",
    xticklabels=model_names,
    yticklabels=model_names,
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "% of True Class"}
)
plt.ylabel("True Model", fontsize=12)
plt.xlabel("Predicted Model", fontsize=12)
plt.title("Confusion Matrix (% per true class)\nDiagonal = correct, Off-diagonal = confused", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("models/confusion_matrix_percent.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: models/confusion_matrix_percent.png")

# ══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📋 SHAP INTERPRETABILITY REPORT")
print("="*60)

for name, feats in top_features_per_model.items():
    print(f"\n🔍 {name.upper()}")
    print(f"   Top identifying features:")
    sorted_feats = sorted(feats, key=lambda x: x[2], reverse=True)[:4]
    for feat, mean_val, importance in sorted_feats:
        direction = "↑ HIGH" if mean_val > 0 else "↓ LOW"
        print(f"   • {feat:35s}  {direction}  (importance={importance:.4f})")

print("\n🎯 Hardest model pairs to distinguish:")
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        confused = cm_percent[i, j] + cm_percent[j, i]
        if confused > 10:
            print(f"   {model_names[i]} ↔ {model_names[j]}: {confused:.1f}% total confusion")

print("\n✅ SHAP analysis complete!")
print("Check models/ folder for 3 new PNG files.")