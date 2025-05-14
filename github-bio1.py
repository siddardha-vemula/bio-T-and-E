import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data
def load_data():
    dataset_paths = [
        "/kaggle/input/cardio-train/cardio_train.csv",
        "/kaggle/input/cardio-data/cardio_train.csv",
        "/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv"
    ]
    for path in dataset_paths:
        try:
            return pd.read_csv(path, delimiter=';')
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Dataset not found in expected paths")

df = load_data()
print("Data loaded successfully")

# Basic preprocessing
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

X = df.drop(columns=['cardio'])
y = df['cardio']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create pipeline with SMOTE and scaling
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
])

X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

# Train individual models
print("Training models...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

rf.fit(X_train_res, y_train_res)
lr.fit(X_train_res, y_train_res)

# Manual voting ensemble
def ensemble_predict(models, X):
    predictions = np.array([model.predict(X) for model in models])
    return np.round(np.mean(predictions, axis=0))

models = [rf, lr]
y_pred = ensemble_predict(models, X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Final Model Accuracy: {accuracy * 100:.2f}%")
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-validation (simplified)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    rf,  # Using just RandomForest for CV to avoid complexity
    X_train_res, 
    y_train_res, 
    cv=5, 
    scoring='accuracy'
)
print(f"\n✅ Mean Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")
print(f"Standard Deviation: {np.std(cv_scores) * 100:.2f}%")
