# s1
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,mean_absolute_error,mean_squared_error,classification_report
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# s2
df=pd.read_csv("data/Churn_Modelling.csv")


# Drop useless columns
df=df.drop(['RowNumber','CustomerId','Surname'],axis=1)

# Encoding
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df=pd.get_dummies(df,columns=['Geography'])


# Features & target
X=df.drop('Exited',axis=1)
# X=df[['CreditScore','Gender','Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Geography_France','Geography_Germany','Geography_Spain']]
y=df['Exited']

# Split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,random_state=42,test_size=0.2
)

# Model

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced')
}


# Train & Compare
results = []


for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, pred)
    })


# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by best model
results_df = results_df.sort_values(by="F1 Score", ascending=False)

print("\nModel Comparison:\n")
print(results_df)


# Visualization
plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="F1 Score", data=results_df)
plt.title("Model Comparison (F1 Score)")
plt.xticks(rotation=20)
plt.show()