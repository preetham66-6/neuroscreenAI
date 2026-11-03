# =============================================================================
# FILE: 6_save_model.py
# PURPOSE: Officially save the best trained model and the scaler together
#          into the models/ folder, ready to be loaded by the Streamlit app.
#          Run this AFTER 5_shap_explain.py
# =============================================================================

import os
import sys
import joblib

# Force Python to print output immediately (fixes blank terminal on Python 3.14)
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# SECTION 1: Load the Best Model and Scaler
# =============================================================================

# Load the best model that was automatically selected in 3_train_models.py
# This is the model with the highest accuracy on the test set
print("[INFO] Loading best model and scaler...")

best_model      = joblib.load(os.path.join("models", "best_model.pkl"))
best_model_name = joblib.load(os.path.join("models", "best_model_name.pkl"))

# The scaler was saved in 2_preprocess.py -- we need it in the app too
# because any new input from the user must be scaled the same way as training data
scaler        = joblib.load(os.path.join("models", "scaler.pkl"))
feature_names = joblib.load(os.path.join("models", "feature_names.pkl"))

print(f"[OK] Loaded best model  : {best_model_name}")
print(f"[OK] Loaded scaler      : StandardScaler")
print(f"[OK] Loaded features    : {len(feature_names)} voice measurements")


# =============================================================================
# SECTION 2: Print a Summary of What Is Being Saved
# =============================================================================

print("\n--- Model Details ---")
print(f"  Model type : {type(best_model).__name__}")
print(f"  Model name : {best_model_name}")
print(f"\n--- Features the model was trained on ---")
for i, name in enumerate(feature_names, start=1):
    print(f"  {i:2}. {name}")


# =============================================================================
# SECTION 3: Save Final Model and Scaler with Clear Names
# =============================================================================

# We save them with descriptive names so app.py can load them easily
# joblib is better than pickle for saving ML models with numpy arrays inside

print("\n[INFO] Saving final model and scaler for the Streamlit app...")

# Save the final model
final_model_path = os.path.join("models", "final_model.pkl")
joblib.dump(best_model, final_model_path)
print(f"[OK] Saved: {final_model_path}")

# Save the scaler -- MUST be saved alongside the model
# If we lose the scaler, new user inputs cannot be scaled correctly
final_scaler_path = os.path.join("models", "final_scaler.pkl")
joblib.dump(scaler, final_scaler_path)
print(f"[OK] Saved: {final_scaler_path}")

# Save feature names -- used by app.py to create the right input fields
final_features_path = os.path.join("models", "final_features.pkl")
joblib.dump(feature_names, final_features_path)
print(f"[OK] Saved: {final_features_path}")

# Save the model name as a plain text file -- displayed in the app UI
model_name_path = os.path.join("models", "final_model_name.txt")
with open(model_name_path, "w") as f:
    f.write(best_model_name)
print(f"[OK] Saved: {model_name_path}")


# =============================================================================
# SECTION 4: Verify the Saved Files Can Be Loaded Back Correctly
# =============================================================================

# This is a sanity check -- we reload the files we just saved and confirm they work
print("\n[INFO] Verifying saved files by reloading them...")

loaded_model    = joblib.load(final_model_path)
loaded_scaler   = joblib.load(final_scaler_path)
loaded_features = joblib.load(final_features_path)

with open(model_name_path, "r") as f:
    loaded_name = f.read()

print(f"[OK] Model reloaded    : {type(loaded_model).__name__}")
print(f"[OK] Scaler reloaded   : {type(loaded_scaler).__name__}")
print(f"[OK] Features reloaded : {len(loaded_features)} names")
print(f"[OK] Model name        : {loaded_name}")


# =============================================================================
# SECTION 5: Quick Prediction Test
# =============================================================================

# Load test data and make one prediction to confirm the loaded model still works
print("\n[INFO] Running a quick prediction test on 3 test samples...")

X_test = joblib.load(os.path.join("models", "X_test.pkl"))
y_test = joblib.load(os.path.join("models", "y_test.pkl"))

# Predict on the first 3 test samples
sample        = X_test[:3]
predictions   = loaded_model.predict(sample)
probabilities = loaded_model.predict_proba(sample)

label_map = {0: "Healthy", 1: "Parkinson's"}

print("\n  Sample | Actual Label    | Predicted       | Confidence")
print("  " + "-" * 60)
for i in range(3):
    actual    = label_map[int(y_test.iloc[i])]
    predicted = label_map[int(predictions[i])]
    # Confidence = probability of the predicted class
    confidence = probabilities[i][int(predictions[i])] * 100
    match = "[CORRECT]" if actual == predicted else "[WRONG]"
    print(f"  {i+1}      | {actual:<15}  | {predicted:<15}  | {confidence:.1f}%  {match}")


# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY!")
print("")
print("Files ready for the Streamlit app:")
print("  models/final_model.pkl")
print("  models/final_scaler.pkl")
print("  models/final_features.pkl")
print("  models/final_model_name.txt")
print("  models/shap_explainer.pkl")
print("")
print("You can now run: streamlit run app.py")
print("=" * 60)