import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ============================================
# 1️⃣ STUDENT DATASET - LINEAR REGRESSION
# ============================================

student = pd.read_csv("data/student.csv")

X1 = student[["Hours"]]
y1 = student["Scores"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)
pred1 = model1.predict(X1_test)

r2 = r2_score(y1_test, pred1)

# ============================================
# 2️⃣ TITANIC DATASET - RANDOM FOREST
# ============================================

titanic = pd.read_csv("data/titanic.csv")

titanic = titanic.dropna()
titanic = titanic.select_dtypes(include=[np.number])

X2 = titanic.drop("Survived", axis=1)
y2 = titanic["Survived"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = RandomForestClassifier()
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)

acc2 = accuracy_score(y2_test, pred2)
prec2 = precision_score(y2_test, pred2)
rec2 = recall_score(y2_test, pred2)
f12 = f1_score(y2_test, pred2)

# ============================================
# 3️⃣ HEART DATASET - SVM
# ============================================

heart = pd.read_csv("data/heart.csv")

X3 = heart.drop("target", axis=1)
y3 = heart["target"]

scaler = StandardScaler()
X3 = scaler.fit_transform(X3)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model3 = SVC()
model3.fit(X3_train, y3_train)
pred3 = model3.predict(X3_test)

acc3 = accuracy_score(y3_test, pred3)
prec3 = precision_score(y3_test, pred3)
rec3 = recall_score(y3_test, pred3)
f13 = f1_score(y3_test, pred3)

# ============================================
# RESULTS TABLE
# ============================================

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "SVM"],
    "Accuracy": [None, acc2, acc3],
    "Precision": [None, prec2, prec3],
    "Recall": [None, rec2, rec3],
    "F1 Score": [None, f12, f13],
    "R2 Score": [r2, None, None]
})

print("\nModel Comparison:\n")
print(results)

# ============================================
# GRAPHICAL REPRESENTATION
# ============================================

# Classification comparison
plt.figure()
classification_results = results.dropna(subset=["Accuracy"])
classification_results.set_index("Model")[["Accuracy", "F1 Score"]].plot(kind="bar")
plt.title("Classification Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# Regression comparison
plt.figure()
plt.bar(["Linear Regression"], [r2])
plt.title("Regression Model R2 Score")
plt.ylabel("R2 Score")
plt.show()