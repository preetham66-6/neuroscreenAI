
import os
import joblib
import numpy as np
import pandas as pd

# Each of these is a different ML algorithm -- we will train all 4 and compare
from sklearn.svm import SVC                        
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression 

# accuracy_score measures what % of predictions the model got right
from sklearn.metrics import accuracy_score



print("[INFO] Loading preprocessed data from models/ folder...")

X_train = joblib.load(os.path.join("models", "X_train.pkl"))
X_test  = joblib.load(os.path.join("models", "X_test.pkl"))
y_train = joblib.load(os.path.join("models", "y_train.pkl"))
y_test  = joblib.load(os.path.join("models", "y_test.pkl"))

print("[OK] Data loaded.")
print("     Training set size :", X_train.shape[0], "samples")
print("     Testing  set size :", X_test.shape[0],  "samples")



models = {

    # SVM: Finds the best boundary line (or curve) that separates the two classes.
    # kernel='rbf' means it can draw curved boundaries, not just straight lines.
    # C=1 controls how strictly it tries to classify training data correctly.
    # probability=True lets us get confidence scores (needed for app.py later)
    "SVM": SVC(
        kernel="rbf",
        C=1,
        random_state=42,
        probability=True
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = []

# A dictionary to store the trained model objects (we will need them in later files)
trained_models = {}

print("\n[INFO] Training all 4 models...\n")

# Loop through each model -- train it, test it, record the accuracy
for model_name, model in models.items():

    print(f"  Training: {model_name}...")

    # .fit() is how we TRAIN the model -- it learns patterns from training data
    model.fit(X_train, y_train)

    # .predict() is how the model makes predictions on NEW (test) data
    y_pred = model.predict(X_test)

    # Calculate what % of test predictions were correct
    acc = accuracy_score(y_test, y_pred)

    # Store result as a percentage rounded to 2 decimal places
    results.append({
        "Model": model_name,
        "Accuracy (%)": round(acc * 100, 2)
    })

    # Keep the trained model object for saving later
    trained_models[model_name] = model

    print(f"  [OK] {model_name} trained. Accuracy: {acc * 100:.2f}%")

results_df = pd.DataFrame(results).sort_values("Accuracy (%)", ascending=False)
results_df = results_df.reset_index(drop=True)
results_df.index += 1  # Start row numbering from 1 instead of 0

print("\n" + "="*45)
print("        MODEL COMPARISON TABLE")
print("="*45)
print(results_df.to_string())
print("="*45)

best_model_name = results_df.iloc[0]["Model"]
best_accuracy   = results_df.iloc[0]["Accuracy (%)"]
best_model      = trained_models[best_model_name]

print(f"\n[RESULT] Best model: {best_model_name} with {best_accuracy}% accuracy")

print("\n[INFO] Saving all trained models...")

for model_name, model in trained_models.items():
    # Replace spaces with underscores to make a clean filename
    filename = model_name.replace(" ", "_") + ".pkl"
    filepath = os.path.join("models", filename)
    joblib.dump(model, filepath)
    print(f"     Saved: models/{filename}")

# Also save the best model separately -- used by 5_shap_explain.py and 6_save_model.py
joblib.dump(best_model,      os.path.join("models", "best_model.pkl"))
joblib.dump(best_model_name, os.path.join("models", "best_model_name.pkl"))

print("\n[OK] Best model also saved as: models/best_model.pkl")
print("\n" + "="*60)
print("TRAINING COMPLETE! You can now run 4_evaluate.py")
print("="*60)