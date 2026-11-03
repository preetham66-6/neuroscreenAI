import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1 — Load data
df = pd.read_csv('parkinsons.csv')

# Step 2 — Separate features and target
X = df.drop(columns=['name', 'status'])  # inputs (voice features)
y = df['status']                          # output (0 = healthy, 1 = parkinson's)

# Step 3 — Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4 — Normalise (scale all features to same range)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5 — Train SVM model
model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)

# Step 6 — Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))