import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("models/titanic_model.joblib")

st.title("Titanic Survival Predictor")
st.write("Predict passenger survival on the Titanic using manual input or CSV upload.")

# -----------------------------
# Manual Input Section
# -----------------------------
st.header("Manual Input")
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
sex = st.selectbox("Sex", ["male", "female"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert manual input into a dataframe for prediction
input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Sex_male": [1 if sex == "male" else 0],
    "Embarked_C": [1 if embarked == "C" else 0],
    "Embarked_Q": [1 if embarked == "Q" else 0],
    "Embarked_S": [1 if embarked == "S" else 0]
})

# Predict on manual input
if st.button("Predict Survival for Manual Input"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"Prediction: {'Survived' if pred == 1 else 'Did Not Survive'} (Survival probability: {prob:.2f})")

# -----------------------------
# CSV Upload Section
# -----------------------------
st.header("Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Preprocess uploaded data
    # Fill missing values if any
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Ensure all required columns exist
    required_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare",
                     "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
    else:
        if st.button("Predict Survival for CSV"):
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]
            df["Prediction"] = predictions
            df["Survival_Prob"] = probabilities
            df["Prediction_Label"] = df["Prediction"].map({0: "Did Not Survive", 1: "Survived"})
            df["Survival_Prob"] = df["Survival_Prob"].apply(lambda x: f"{x:.2f}")

            st.write("Prediction results:")
            st.dataframe(df)

            # Download predictions
            csv = df.to_csv(index=False).encode()
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
