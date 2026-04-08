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


# s2
df=pd.read_csv("Churn_Modelling.csv")

# Drop useless columns
df=df.drop(['RowNumber','CustomerId','Surname'],axis=1)

# Encoding
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df=pd.get_dummies(df,columns=['Geography'])


# Features & target
X=df.drop('Exited',axis=1)
y=df['Exited']

# Split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,random_state=42,test_size=0.2
)

# Model
# model=LinearRegression()
# model=LogisticRegression()
# model=RandomForestClassifier()
model=RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train,y_train)

# Prediction
pred=model.predict(X_test)


# Evaluation
print('Classification report')
print(classification_report(y_test,pred))


# Step 8: Confusion matrix
conf_matrix=confusion_matrix(y_test,pred)


# 7. Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

