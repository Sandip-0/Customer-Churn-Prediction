import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction Dashboard")
st.caption("Random Forest classification with interactive Streamlit visualization")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/Churn_Modelling.csv")


@st.cache_resource
def train_model():
    df = load_data().copy()

    # Same preprocessing used in model.py
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df = pd.get_dummies(df, columns=["Geography"])

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, pred


try:
    raw_df = load_data()
    model, X_train, X_test, y_train, y_test, pred = train_model()
except FileNotFoundError:
    st.error(
        "Churn_Modelling.csv was not found. "
        "Make sure the CSV is inside the project's data/ folder."
    )
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Dashboard Navigation")
page = st.sidebar.radio(
    "Select section",
    ["Overview", "Data Visualization", "Model Performance", "Predict Churn"],
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.header("Project Overview")

    total_customers = len(raw_df)
    churned = int(raw_df["Exited"].sum())
    retained = total_customers - churned
    churn_rate = churned / total_customers * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.subheader("Churn Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=raw_df, x="Exited", ax=ax)
    ax.set_xlabel("Churn Status (0 = Stayed, 1 = Churned)")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Churn Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

# --------------------------------------------------
# Data Visualization
# --------------------------------------------------
elif page == "Data Visualization":
    st.header("Customer Data Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Geography")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=raw_df, x="Geography", hue="Exited", ax=ax)
        ax.set_title("Churn by Country")
        ax.set_xlabel("Country")
        ax.set_ylabel("Customers")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Churn by Gender")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=raw_df, x="Gender", hue="Exited", ax=ax)
        ax.set_title("Churn by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Customers")
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            data=raw_df,
            x="Age",
            hue="Exited",
            bins=30,
            kde=True,
            ax=ax,
        )
        ax.set_title("Age Distribution by Churn")
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("Credit Score Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            data=raw_df,
            x="CreditScore",
            hue="Exited",
            bins=30,
            kde=True,
            ax=ax,
        )
        ax.set_title("Credit Score by Churn")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Feature Relationships")

    selected_x = st.selectbox(
        "Choose feature",
        ["Age", "CreditScore", "Balance", "EstimatedSalary", "Tenure"],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=raw_df, x="Exited", y=selected_x, ax=ax)
    ax.set_xlabel("Churn Status")
    ax.set_title(f"{selected_x} vs Churn")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
elif page == "Model Performance":
    st.header("Random Forest Model Performance")

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.2%}")
    c2.metric("Precision", f"{precision:.2%}")
    c3.metric("Recall", f"{recall:.2%}")
    c4.metric("F1 Score", f"{f1:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            pred,
            target_names=["Stayed", "Churned"],
            output_dict=True,
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    importance.head(10).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Important Features")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# Predict Churn
# --------------------------------------------------
elif page == "Predict Churn":
    st.header("Predict Customer Churn")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=650
        )
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )
        tenure = st.number_input(
            "Tenure", min_value=0, max_value=10, value=5
        )

    with col2:
        balance = st.number_input(
            "Balance", min_value=0.0, value=50000.0
        )
        num_products = st.number_input(
            "Number of Products", min_value=1, max_value=4, value=1
        )
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=50000.0
        )

    with col3:
        geography = st.selectbox(
            "Geography", ["France", "Germany", "Spain"]
        )
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )
        has_card = st.selectbox(
            "Has Credit Card", ["Yes", "No"]
        )
        active_member = st.selectbox(
            "Is Active Member", ["Yes", "No"]
        )

    if st.button("Predict Churn", type="primary"):
        input_data = pd.DataFrame(
            {
                "CreditScore": [credit_score],
                "Gender": [0 if gender == "Male" else 1],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [num_products],
                "HasCrCard": [1 if has_card == "Yes" else 0],
                "IsActiveMember": [1 if active_member == "Yes" else 0],
                "EstimatedSalary": [estimated_salary],
                "Geography_France": [geography == "France"],
                "Geography_Germany": [geography == "Germany"],
                "Geography_Spain": [geography == "Spain"],
            }
        )

        # Ensure the prediction columns exactly match the training columns.
        input_data = input_data[X_train.columns]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(
                f"Prediction: Customer is likely to CHURN "
                f"(probability: {probability:.2%})"
            )
        else:
            st.success(
                f"Prediction: Customer is likely to STAY "
                f"(churn probability: {probability:.2%})"
            )
