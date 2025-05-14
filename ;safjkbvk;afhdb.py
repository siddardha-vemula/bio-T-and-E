import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance

# Load Data
def load_data():
    # List of possible paths to try
    dataset_paths = [
        "/Users/siddardhavemula/Downloads/cardio_train1.csv"  # Home directory
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, delimiter=';')
                print(f"✅ Loaded dataset from {path}")
                return df
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
    
    # If file not found, prompt user for path
    print("❌ Dataset not found in expected paths:")
    for path in dataset_paths:
        print(f"  - {path}")
    print("\nPlease check the following:")
    print("1. Ensure 'cardio_train1.csv' exists in /workspaces/BOI1/ or another directory.")
    print("2. Verify the file name is exactly 'cardio_train1.csv' (case-sensitive).")
    print("3. Run 'find /workspaces -name \"*cardio*.csv\"' to locate the file.")
    print("4. Upload the file to /workspaces/BOI1/ if missing.")
    user_path = input("Enter the full path to cardio_train1.csv (or press Enter to exit): ").strip()
    
    if user_path:
        if os.path.exists(user_path):
            try:
                df = pd.read_csv(user_path, delimiter=';')
                print(f"✅ Loaded dataset from {user_path}")
                return df
            except Exception as e:
                print(f"Error reading {user_path}: {e}")
        else:
            print(f"❌ File not found at {user_path}")
    
    raise FileNotFoundError("Dataset not found. Please provide a valid path to cardio_train1.csv.")

df = load_data()
print(f"Initial dataset size: {df.shape}")

# Enhanced Data Cleaning
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Clean Numeric Columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.replace(',', '.').str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Relaxed Outlier Handling
def cap_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower, upper)
    return df

for col in ['ap_hi', 'ap_lo', 'height', 'weight']:
    df = cap_outliers(df, col)

# Relaxed blood pressure validation
df = df[(df['ap_hi'] >= 40) & (df['ap_hi'] <= 300) & (df['ap_lo'] >= 20) & (df['ap_lo'] <= 200)]
df = df[df['ap_hi'] > df['ap_lo']]

# Handle missing values
df.dropna(inplace=True)
print(f"Dataset size after cleaning: {df.shape}")
print(f"Class distribution after cleaning:\n{df['cardio'].value_counts()}")

# Feature Engineering
df['age_years'] = df['age'] // 365
df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['bmi_cholesterol'] = df['bmi'] * df['cholesterol']
df['ap_product'] = df['ap_hi'] * df['ap_lo']
df['age_bmi'] = df['age_years'] * df['bmi']

# Encode categorical variables
df['cholesterol'] = df['cholesterol'].map({1: 0, 2: 1, 3: 2})
df['gluc'] = df['gluc'].map({1: 0, 2: 1, 3: 2})
df['Alcohol use'] = df['Alcohol use'].astype(int)

# Split Data
X = df.drop(columns=['cardio'])
y = df['cardio']
print(f"Features: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Class distribution in training set:\n{y_train.value_counts()}")

# Pipeline with ADASYN
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('adasyn', ADASYN(random_state=42)),
])

# Apply ADASYN with error handling
try:
    X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)
    print(f"Class distribution after ADASYN:\n{pd.Series(y_train_res).value_counts()}")
except Exception as e:
    print(f"ADASYN failed: {e}")
    print("Falling back to no oversampling")
    X_train_res, y_train_res = X_train.copy(), y_train.copy()

# Feature Selection with Permutation Importance
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train_res, y_train_res)
perm_importance = permutation_importance(rf_temp, X_train_res, y_train_res, n_repeats=5, random_state=42)
important_features = X_train.columns[perm_importance.importances_mean > 0]
X_train_res_selected = X_train_res[important_features]
X_test_selected = X_test[important_features]

print(f"✅ Selected Features: {important_features.tolist()}")
print(f"Selected training set shape: {X_train_res_selected.shape}")

# Single Model (XGBoost) for Simplicity
param_grid_xgb = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_distributions=param_grid_xgb,
    n_iter=30,
    scoring='accuracy',
    cv=10,
    random_state=42,
    n_jobs=-1
)
xgb_search.fit(X_train_res_selected, y_train_res)
xgb_best = xgb_search.best_estimator_
print(f"Best XGBoost parameters: {xgb_search.best_params_}")

# Predictions
y_pred = xgb_best.predict(X_test_selected)

# Evaluation
print("\n📊 XGBoost Model Evaluation")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Cross-validation
cv_scores = cross_val_score(
    xgb_best,
    X_train_res_selected,
    y_train_res,
    cv=StratifiedKFold(n_splits=10),
    scoring='accuracy'
)
print(f"\n✅ Mean Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%")
print(f"Standard Deviation: {np.std(cv_scores) * 100:.2f}%")

# Feature Importance Plot
perm_importance = permutation_importance(xgb_best, X_test_selected, y_test, n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(important_features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.title("Feature Importance (Permutation)")
plt.xlabel("Importance")
plt.savefig('feature_importance.png')
plt.close()

# Save Model
import joblib
joblib.dump(xgb_best, 'cardio_model.pkl')
print("✅ Model saved as 'cardio_model.pkl")
