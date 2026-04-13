import streamlit as st
import pandas as pd
import joblib

# -------------------- Load Model --------------------
MODEL_PATH = "artifacts/xgboost_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="No Show Prediction", layout="centered")

st.title("🏥 Appointment No-Show Prediction")
st.write("Enter patient details to predict if they will miss the appointment.")

# -------------------- Input Form --------------------
def user_input():
    data = {
        "Gender": st.selectbox("Gender", [0, 1]),
        "Age": st.number_input("Age", 0, 100, 30),
        "Scholarship": st.selectbox("Scholarship", [0, 1]),
        "Hypertension": st.selectbox("Hypertension", [0, 1]),
        "Diabetes": st.selectbox("Diabetes", [0, 1]),
        "Handicap": st.selectbox("Handicap", [0, 1]),
        "SMS_received": st.selectbox("SMS Received", [0, 1]),

        "waitDays": st.number_input("Wait Days", 0, 100, 5),

        # Engineered / derived features (user must input)
        "prev_no_show_count": st.number_input("Previous No Shows", 0, 50, 0),
        "prev_show_count": st.number_input("Previous Shows", 0, 50, 1),
        "prev_total_visits": st.number_input("Total Visits", 0, 100, 1),
        "no_show_rate": st.slider("No Show Rate", 0.0, 1.0, 0.0),

        "is_weekend": st.selectbox("Is Weekend", [0, 1]),
        "is_old": st.selectbox("Is Old (>60)", [0, 1]),
        "long_wait": st.selectbox("Long Wait (>7 days)", [0, 1]),

        "comorbidity_count": st.number_input("Comorbidity Count", 0, 5, 0),
        "sms_effectiveness": st.slider("SMS Effectiveness", 0.0, 1.0, 0.0),
        "weighted_no_show_rate": st.slider("Weighted No Show Rate", 0.0, 1.0, 0.0),

        # LeadTime one-hot (IMPORTANT)
        "LeadTime_bin_1week": st.selectbox("Lead Time: 1 Week", [0, 1]),
        "LeadTime_bin_2-3weeks": st.selectbox("Lead Time: 2-3 Weeks", [0, 1]),
        "LeadTime_bin_long": st.selectbox("Lead Time: Long", [0, 1])
    }

    return pd.DataFrame([data])


input_df = user_input()

# -------------------- Align Columns --------------------
model_features = model.get_booster().feature_names

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# -------------------- Prediction --------------------
if st.button("Predict"):
    prob = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f" Likely No-Show (Probability: {prob:.2f})")
    else:
        st.success(f" Will Show (Probability: {prob:.2f})")