import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import json
from sklearn.metrics import classification_report

# 1. Dataset-i oxu
df = pd.read_excel("data/ADHD.xlsx", sheet_name="Sheet1")

# 2. Lazımi sütunları seç
columns_needed = ["age", "sex", "asrs1_total.x", "bdi1_total", "bai1_total", "audit1_total", "matric_mark", "nbt_ave"]
df = df[columns_needed].dropna()

# 3. sex sütununu rəqəmə çevir (female=0, male=1)
df["sex"] = df["sex"].map({"female": 0, "male": 1})
df = df.dropna(subset=["sex"])

# 4. Hədəf dəyişəni: asrs1_total.x -> 3 kateqoriya (low/medium/high)
df["adhd_risk_level"] = pd.qcut(df["asrs1_total.x"], q=3, labels=["low", "medium", "high"])

# 5. Feature-lər və hədəf
features = ["age", "sex", "bdi1_total", "bai1_total", "audit1_total", "matric_mark", "nbt_ave"]
X = df[features]
y = df["adhd_risk_level"]

# 6. Train/test bölgüsü
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Modeli qur və öyrət
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Nəticələri yoxla
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
feature_importance_dict = dict(zip(features, importances))
sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for name, score in sorted_importance:
    print(f"{name:20s} {score:.3f}")

# Nəticələri fayla saxla (sunuş üçün)
with open("model_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": accuracy_score(y_test, y_pred),
        "feature_importance": sorted_importance,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }, f, ensure_ascii=False, indent=2)

print("\nHesabat saxlanıldı: model_report.json")

# 9. Modeli və feature siyahısını saxla
joblib.dump(model, "model.pkl")
with open("features.json", "w") as f:
    json.dump(features, f)

print("Model saxlanıldı: model.pkl")