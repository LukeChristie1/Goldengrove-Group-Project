import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay
)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading Titanic dataset...")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Titanic-Dataset.csv')

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}")

df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ============================================================
# FEATURE ENGINEERING
# ============================================================
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

target = 'Survived'

features = [
    'Pclass', 'Name', 'Sex', 'Age',
    'Fare', 'Embarked', 'FamilySize',
    'IsAlone', 'FarePerPerson'
]

df = df[features + [target]].copy()

# ============================================================
# TITLE FEATURE ENGINEERING
# ============================================================
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)

title_mapping = {
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
    'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
    'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare',
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
}

df['Title'] = df['Title'].replace(title_mapping)

# Drop Name after extracting Title
df.drop(columns=['Name'], inplace=True)

# ============================================================
# HANDLE MISSING VALUES
# ============================================================
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# ============================================================
# ENCODE CATEGORICAL VARIABLES
# ============================================================
# Sex (binary)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Embarked (one-hot encoding)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Title (one-hot encoding)
df = pd.get_dummies(df, columns=['Title'], drop_first=True)

# Final feature list
features = [col for col in df.columns if col != target]

X = df[features].values
y = df[target].values

print("\nFinal feature set:")
print(features)
print("Target distribution:", np.bincount(y))

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================
scale_pos_weight = y_train[y_train == 0].size / y_train[y_train == 1].size
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================================
# BUILD XGBOOST MODEL
# ============================================================
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    early_stopping_rounds=20,
    verbosity=0
)

# ============================================================
# TRAIN MODEL
# ============================================================
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")

# ============================================================
# EVALUATION
# ============================================================
def evaluate(split_name, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba)
    print(f"{split_name} Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    return pred, proba

print("\nModel Performance:")
y_train_pred, _ = evaluate("Train", X_train, y_train)
y_val_pred, _ = evaluate("Validation", X_val, y_val)
y_test_pred, y_test_proba = evaluate("Test", X_test, y_test)

print("\nClassification Report (Test):")
print(classification_report(
    y_test, y_test_pred,
    target_names=['Did Not Survive', 'Survived']
))

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Did Not Survive', 'Survived']
)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("xgboost_confusion_matrix.png", dpi=120)
plt.close()

# ============================================================
# ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_test_proba):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("xgboost_roc_curve.png", dpi=120)
plt.close()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
plt.figure(figsize=(8, 6))
xgb.plot_importance(
    model,
    importance_type='gain',
    max_num_features=10
)
plt.title("Top 10 Feature Importances (Gain)")
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png", dpi=120)
plt.close()

print("\nTraining complete.")
print("Saved plots:")
print("- xgboost_confusion_matrix.png")
print("- xgboost_roc_curve.png")
print("- xgboost_feature_importance.png")
