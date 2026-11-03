# =============================================================================
# FILE: 4_evaluate.py
# PURPOSE: Load all 4 trained models, evaluate them with multiple metrics,
#          print a full comparison table, and save confusion matrix +
#          ROC curve charts as PNG images in the charts/ folder.
#          Run this AFTER 3_train_models.py
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Force Python to print output immediately (fixes blank terminal on Python 3.14)
sys.stdout.reconfigure(line_buffering=True)

# Metrics: different ways to measure how good a model is
# accuracy  : overall % correct
# precision : of all "Parkinson" predictions, how many were actually Parkinson?
# recall    : of all actual Parkinson patients, how many did the model catch?
# f1_score  : balanced average of precision and recall
# roc_auc   : area under the ROC curve -- 1.0 is perfect, 0.5 is random guessing
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# =============================================================================
# SECTION 1: Load Test Data and All 4 Trained Models
# =============================================================================

print("[INFO] Loading test data and trained models...")

X_test  = joblib.load(os.path.join("models", "X_test.pkl"))
y_test  = joblib.load(os.path.join("models", "y_test.pkl"))

model_files = {
    "SVM"                : "SVM.pkl",
    "KNN"                : "KNN.pkl",
    "Random Forest"      : "Random_Forest.pkl",
    "Logistic Regression": "Logistic_Regression.pkl"
}

models = {}
for name, filename in model_files.items():
    models[name] = joblib.load(os.path.join("models", filename))

print("[OK] All models loaded.\n")

# =============================================================================
# SECTION 2: Evaluate Each Model on All Metrics
# =============================================================================

results = []

print("[INFO] Evaluating all 4 models...\n")

for model_name, model in models.items():

    y_pred = model.predict(X_test)

    # predict_proba() returns confidence scores (probability of each class)
    # We take column [1] which is the probability of Parkinson's (class = 1)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = y_pred

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    results.append({
        "Model"         : model_name,
        "Accuracy (%)"  : round(acc * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)"   : round(recall * 100, 2),
        "F1 Score (%)" : round(f1 * 100, 2),
        "ROC-AUC"      : round(roc_auc, 4)
    })

    print(f"  [{model_name}]")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {precision*100:.2f}%")
    print(f"    Recall    : {recall*100:.2f}%")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print()

# =============================================================================
# SECTION 3: Print Full Comparison Table
# =============================================================================

# Sort by F1 Score -- better than accuracy alone for medical datasets
results_df = pd.DataFrame(results).sort_values("F1 Score (%)", ascending=False)
results_df = results_df.reset_index(drop=True)
results_df.index += 1

print("=" * 80)
print("                     FULL MODEL COMPARISON TABLE")
print("=" * 80)
print(results_df.to_string())
print("=" * 80)

# =============================================================================
# SECTION 4: Save Confusion Matrices as PNG
# =============================================================================

# Confusion matrix shows: TP, TN, FP, FN
# False Negatives (Parkinson MISSED) = most dangerous error in medicine!

os.makedirs("charts", exist_ok=True)

print("\n[INFO] Saving confusion matrix charts...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Healthy", "Parkinson"],
        yticklabels=["Healthy", "Parkinson"],
        ax=axes[i]
    )
    axes[i].set_title(f"{model_name}\nConfusion Matrix", fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("Actual Label")

plt.suptitle("Confusion Matrices — All 4 Models", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join("charts", "confusion_matrices.png"), bbox_inches="tight", dpi=150)
plt.close()
print("[OK] Saved: charts/confusion_matrices.png")

# =============================================================================
# SECTION 5: Save ROC Curves as PNG
# =============================================================================

# ROC Curve: closer to top-left corner = better model
# AUC: 1.0 = perfect, 0.5 = random guessing

print("\n[INFO] Saving ROC curve chart...")

plt.figure(figsize=(9, 7))
line_styles = ["-", "--", "-.", ":"]

for i, (model_name, model) in enumerate(models.items()):
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, linestyle=line_styles[i], linewidth=2,
             label=f"{model_name} (AUC = {auc_score:.4f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.5)")
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title("ROC Curves — All 4 Models", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join("charts", "roc_curves.png"), bbox_inches="tight", dpi=150)
plt.close()
print("[OK] Saved: charts/roc_curves.png")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("EVALUATION COMPLETE! Charts saved in the charts/ folder.")
print("You can now run 5_shap_explain.py")
print("=" * 60)