import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\Churn_Modelling.csv")
print("✅ Loaded data:", df.shape)
print("Columns:", df.columns.tolist())
df = df.dropna()
df = df.rename(columns={'Exited': 'Churn'})

features = ['Tenure', 'Balance', 'EstimatedSalary']
X = df[features].astype(float)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Pred No', 'Pred Yes'],
    yticklabels=['Actual No', 'Actual Yes']
)
plt.title("Confusion Matrix (Random Forest)")
plt.show()

importances = model.feature_importances_
imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=imp_df)
plt.title("Feature Importance")
plt.show()
