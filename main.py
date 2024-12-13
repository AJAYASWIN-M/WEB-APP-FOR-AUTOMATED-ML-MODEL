import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ml_utility import preprocess_data, train_model, evaluate_model

st.set_page_config(
    page_title="Automated ML",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🤖 Automated ML Model Training")

# Dataset upload (drag and drop)
uploaded_file = st.file_uploader(
    "Drag and drop your dataset here or click to upload",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")

    # Display the dataset
    if df is not None:
        st.dataframe(df.head())

        # Remaining code logic (target selection, model training, etc.)
        col1, col2, col3, col4 = st.columns(4)

        scaler_type_list = ["standard", "minmax"]

        model_dictionary = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": XGBClassifier()
        }

        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Model name")

        if st.button("Train the Model"):
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

            model_to_be_trained = model_dictionary[selected_model]

            model = train_model(X_train, y_train, model_to_be_trained, model_name)

            accuracy = evaluate_model(model, X_test, y_test)

            st.success("Test Accuracy: " + str(accuracy))
else:
    st.info("Upload a dataset to proceed.")
