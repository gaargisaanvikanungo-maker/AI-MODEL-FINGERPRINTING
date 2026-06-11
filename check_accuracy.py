import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
print("Loading model...")
data = pickle.load(open('models/best_model.pkl', 'rb'))
model    = data['model']
encoder  = data['encoder']
features = data['features']
print(f"Model: {data['name']}")
print(f"Encoder classes: {encoder.classes_}\n")

# Load features
features_df = pd.read_csv('data/features.csv')
print(f"Shape: {features_df.shape}\n")

# Split features and labels
X = features_df[features]
y = features_df['model_name']

# Encode y labels to numbers (same as training)
y_encoded = encoder.transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Test samples: {len(y_test)}\n")

# Predict
y_pred = model.predict(X_test)

# Decode back to text labels
y_test_labels = encoder.inverse_transform(y_test)
y_pred_labels = encoder.inverse_transform(y_pred)

# Accuracy
acc = accuracy_score(y_test_labels, y_pred_labels)
print(f"Overall Accuracy: {acc:.2%}\n")

# Detailed report
print("Detailed Report:")
print(classification_report(y_test_labels, y_pred_labels))

# Confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=encoder.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.title(f'Confusion Matrix — {data["name"]}')
plt.xlabel('Predicted Model')
plt.ylabel('Actual Model')
plt.tight_layout()
plt.savefig('models/confusion_matrix_new.png')
plt.show()
print("\nConfusion matrix saved!")