# student_success_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

st.title("Student Success Prediction in Internship Programs")
st.write("Upload a CSV file with student profile and participation data")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Data")
    st.dataframe(data.head())

    # Fill missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Assume the target column is named 'Success' (1 = success, 0 = not success)
    target_column = 'Success'
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Select classifier
        classifier_name = st.selectbox("Select Classifier", ["Logistic Regression", "Random Forest", "SVM"])

        if classifier_name == "Logistic Regression":
            clf = LogisticRegression()
        elif classifier_name == "Random Forest":
            clf = RandomForestClassifier()
        else:
            clf = SVC()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.subheader("Model Evaluation")
        st.text("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        st.error("The CSV file must contain a 'Success' column as the target.")
