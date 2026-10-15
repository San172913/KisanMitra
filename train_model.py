import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

# Load and check data
df = pd.read_csv("data/Crop_recommendation.csv")
print(df.head())
print(df.shape)
df.info()
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Clean data
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna()
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
df = df[features + ["label"]]

# Simple EDA
plt.figure(figsize=(8, 4))
df["label"].value_counts().plot(kind="bar")
plt.title("Number of records for each crop")
plt.tight_layout()
plt.savefig("outputs/crop_counts.png")
plt.close()

X = df[features]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train and tune four models
models = {
    "KNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [3, 5, 7]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {"max_depth": [5, 10, None]}),
    "SVM": (make_pipeline(StandardScaler(), SVC()), {"svc__C": [1, 5], "svc__kernel": ["linear", "rbf"]}),
    "Random Forest": (RandomForestClassifier(random_state=42), {"n_estimators": [100], "max_depth": [10, None]})
}

results = []
best_model = None
best_name = ""
best_cv_score = -1

for name, (model, params) in models.items():
    print("\n", name)
    search = GridSearchCV(model, params, cv=5, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    print("Best parameters:", search.best_params_)
    print(classification_report(y_test, predictions, zero_division=0))
    results.append([name, accuracy_score(y_test, predictions), precision_score(y_test, predictions, average="macro", zero_division=0), recall_score(y_test, predictions, average="macro", zero_division=0), f1_score(y_test, predictions, average="macro", zero_division=0), search.best_score_])
    if search.best_score_ > best_cv_score:
        best_cv_score = search.best_score_
        best_model = search.best_estimator_
        best_name = name

# Save comparison and best model
results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "CV F1"])
print("\nModel comparison")
print(results.sort_values("F1", ascending=False))
results.to_csv("outputs/model_results.csv", index=False)

plt.figure(figsize=(7, 4))
plt.bar(results["Model"], results["F1"])
plt.ylabel("Macro F1 score")
plt.title("Model comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

joblib.dump(best_model, "models/best_model.joblib")
with open("models/best_model.txt", "w") as file:
    file.write(best_name)
print("\nBest model:", best_name)
print("Saved model and results.")
