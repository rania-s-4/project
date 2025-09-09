import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model (Random Forest for better performance)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details in the sidebar to predict the likelihood of diabetes.")

# Sidebar inputs
st.sidebar.header("Patient Details")
def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 100)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 30)

    data_input = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    return pd.DataFrame(data_input, index=[0])

input_df = user_input_features()

# Display input
st.subheader("Patient Input Details")
st.write(input_df)

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
pred_prob = model.predict_proba(input_scaled)

# Output
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error(f"⚠️ The patient is likely to have diabetes with a probability of {pred_prob[0][1]*100:.2f}%.")
else:
    st.success(f"✅ The patient is unlikely to have diabetes with a probability of {pred_prob[0][0]*100:.2f}%.")

# Show dataset preview
with st.expander("🔎 View Dataset"):
    st.dataframe(data.head())
