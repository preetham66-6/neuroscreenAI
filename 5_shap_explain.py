# =============================================================================
# FILE: 5_shap_explain.py
# PURPOSE: Use SHAP to explain WHY the best model makes its predictions.
#          SHAP tells us which voice features push the prediction toward
#          Parkinson's or Healthy -- and by how much.
#          Saves a SHAP summary bar chart as a PNG image.
#          Run this AFTER 4_evaluate.py
# =============================================================================

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Force Python to print output immediately (fixes blank terminal on Python 3.14)
sys.stdout.reconfigure(line_buffering=True)

# SHAP = SHapley Additive exPlanations
# It comes from game theory -- it fairly assigns "credit" to each feature
# for how much it contributed to pushing the model toward a given prediction.
# A high positive SHAP value = feature pushed model toward Parkinson's
# A high negative SHAP value = feature pushed model toward Healthy
import shap


# =============================================================================
# SECTION 1: Load the Best Model and Test Data
# =============================================================================

print("[INFO] Loading best model, test data, and feature names...")

best_model      = joblib.load(os.path.join("models", "best_model.pkl"))
best_model_name = joblib.load(os.path.join("models", "best_model_name.pkl"))
X_test          = joblib.load(os.path.join("models", "X_test.pkl"))
X_train         = joblib.load(os.path.join("models", "X_train.pkl"))
feature_names   = joblib.load(os.path.join("models", "feature_names.pkl"))

print(f"[OK] Best model loaded: {best_model_name}")
print(f"     Test samples  : {X_test.shape[0]}")
print(f"     Features      : {len(feature_names)}")


# =============================================================================
# SECTION 2: Choose the Right SHAP Explainer for the Model Type
# =============================================================================

# Different ML models need different SHAP explainers:
#   TreeExplainer   -- fast & exact, for Random Forest
#   LinearExplainer -- for Logistic Regression
#   KernelExplainer -- for any model (SVM, KNN) but slower

print(f"\n[INFO] Creating SHAP explainer for: {best_model_name}...")

if best_model_name == "Random Forest":
    # TreeExplainer is the fastest and most accurate for tree-based models
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # shap_values can be a list [class_0, class_1] or a 3D array -- handle both
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        # shape: (n_classes, n_samples, n_features) -- take class 1 (Parkinson's)
        shap_vals_to_plot = shap_values[1]
    elif shap_values.ndim == 4:
        # newer SHAP: (n_samples, n_features, n_classes)
        shap_vals_to_plot = shap_values[:, :, 1]
    else:
        shap_vals_to_plot = shap_values

elif best_model_name == "Logistic Regression":
    # LinearExplainer works well for linear models like Logistic Regression
    explainer         = shap.LinearExplainer(best_model, X_train)
    shap_values       = explainer.shap_values(X_test)
    shap_vals_to_plot = np.array(shap_values)

else:
    # KernelExplainer for SVM or KNN -- works with any model
    # shap.kmeans summarises training data into 50 representative points (keeps it fast)
    background  = shap.kmeans(X_train, 50)
    explainer   = shap.KernelExplainer(best_model.predict_proba, background)
    shap_values = explainer.shap_values(X_test, nsamples=100)

    # KernelExplainer returns a list of arrays [class_0_shaps, class_1_shaps]
    # We want class 1 (Parkinson's)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        if shap_values.shape[0] == 2:
            # shape (n_classes, n_samples, n_features) -- take index [1] for class 1
            shap_vals_to_plot = shap_values[1]
        else:
            # shape (n_samples, n_features, n_classes) -- take index [:,:,1] for class 1
            shap_vals_to_plot = shap_values[:, :, 1]
    else:
        shap_vals_to_plot = shap_values

print("[OK] SHAP values computed.")


# =============================================================================
# SECTION 3: Compute Mean SHAP Importance Per Feature
# =============================================================================

# Guarantee shap_vals_to_plot is exactly 2D: (n_samples, n_features)
shap_vals_to_plot = np.array(shap_vals_to_plot)

# If somehow still 3D, reduce one more time
if shap_vals_to_plot.ndim == 3:
    shap_vals_to_plot = shap_vals_to_plot[:, :, 0]

# Mean absolute SHAP value across all test samples -- gives one score per feature
# .tolist() converts numpy numbers to plain Python floats so sorting works cleanly
mean_shap = np.abs(shap_vals_to_plot).mean(axis=0).flatten().tolist()

# Pair each feature name with its importance score, then sort highest first
importance_pairs = sorted(
    zip(feature_names, mean_shap),
    key=lambda x: x[1],
    reverse=True
)


# =============================================================================
# SECTION 4: Print Top 10 Most Important Features
# =============================================================================

print("\n--- Top 10 Most Important Voice Features (SHAP) ---")
print(f"  {'Rank':<5} {'Feature':<25} {'Mean |SHAP Value|'}")
print("  " + "-" * 50)
for rank, (feat, val) in enumerate(importance_pairs[:10], start=1):
    print(f"  {rank:<5} {feat:<25} {val:.5f}")


# =============================================================================
# SECTION 5: Save SHAP Bar Chart as PNG
# =============================================================================

# The bar chart shows each feature's average impact on the model's output.
# Longer bar = that feature has more influence on the prediction.

print("\n[INFO] Saving SHAP feature importance chart...")

os.makedirs("charts", exist_ok=True)

top_n      = 15
feat_names = [pair[0] for pair in importance_pairs[:top_n]][::-1]
feat_vals  = [pair[1] for pair in importance_pairs[:top_n]][::-1]

fig, ax = plt.subplots(figsize=(10, 7))

# barh = horizontal bar chart (longer bar = more important feature)
bars = ax.barh(feat_names, feat_vals, color="steelblue", edgecolor="white")

# Add the value as a label at the end of each bar
for bar, val in zip(bars, feat_vals):
    ax.text(
        bar.get_width() + 0.0005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left", fontsize=9
    )

ax.set_xlabel("Mean |SHAP Value| (average impact on prediction)", fontsize=11)
ax.set_title(
    f"SHAP Feature Importance\nBest Model: {best_model_name}",
    fontsize=13, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()

shap_path = os.path.join("charts", "shap_importance.png")
plt.savefig(shap_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"[OK] Saved: {shap_path}")


# =============================================================================
# SECTION 6: Save SHAP Explainer for app.py
# =============================================================================

# Saving the explainer so app.py can generate SHAP charts for new patient inputs
joblib.dump(explainer, os.path.join("models", "shap_explainer.pkl"))
print("[OK] Saved: models/shap_explainer.pkl  (used by app.py)")


# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("SHAP EXPLANATION COMPLETE!")
print("Chart saved : charts/shap_importance.png")
print("You can now run 6_save_model.py")
print("=" * 60)