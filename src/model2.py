import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Corrected import
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier # Higher performance model


df = pd.read_csv("data/Churn_Modelling.csv")
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)


df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

df['TenureByAge'] = df['Tenure'] / (df['Age'] + 1)

df['CreditActive'] = df['HasCrCard'] * df['IsActiveMember']
# Encoding 
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)


X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- s6: Scaling ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    scale_pos_weight=4, 
    random_state=42
)
model.fit(X_train, y_train)


pred = model.predict(X_test)

print('--- Optimized Classification Report ---')
print(classification_report(y_test, pred))


plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Improved Confusion Matrix')
plt.show()

