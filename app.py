# app
import streamlit as st
import requests
import pandas as pd
import os


@st.cache_data
def load_data():
    return pd.read_csv("artifacts/test.csv")

test_data = load_data()


# app config
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Financial Fraud Detection", page_icon=":money_with_wings:", layout="wide", initial_sidebar_state="auto")
st.markdown("### 🛡️ Real-Time Financial Fraud Detection")
st.markdown("Analyse transactions using XGBoost model trained on 284,807 transactions")
st.divider()

st.sidebar.title("Model Info")
st.sidebar.write("Model: XGBoost")
st.sidebar.write("Threshold: 0.3")
st.sidebar.write("Test Recall: 85.7%")
st.sidebar.write("Test Precision: 89.4%")
# api call function
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False
    return False

def predict_fraud(payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API call failed with status code: {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running on localhost:8000")
        return None    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# buttons 
col1, col2 = st.columns(2)

with col1:
    fraud_clicked = st.button("Sample Fraud Transaction")

with col2:
    legit_clicked = st.button("Sample Legitimate Transaction")

# sample show
if fraud_clicked:
    st.session_state.random_row = test_data[test_data['Class'] == 1].sample(1).iloc[0]

if legit_clicked:
    st.session_state.random_row = test_data[test_data['Class'] == 0].sample(1).iloc[0]

if fraud_clicked or legit_clicked:
    st.write(f"**Time:** {st.session_state.random_row['Time']:.0f} seconds")
    st.write(f"**Amount:** ${st.session_state.random_row['Amount']:.2f}")

# Analyse Transaction button
# V1-V28 features
col1, col2 = st.columns(2)
with col1:
    if st.button("Analyse Transaction"):
        if "random_row" not in st.session_state:
            st.warning("Please sample a transaction first")
        else:
            row = st.session_state.random_row
            payload = {
                "Time": row["Time"],
                **{f"V{i}": row[f"V{i}"] for i in range(1, 29)},
                "Amount": row["Amount"]
            }
            if check_api_status():
                result = predict_fraud(payload)
                if result is not None:
                    actual = "Fraud" if row['Class'] == 1 else "Legitimate"
                    st.info(f"Actual Label: {actual}")
                    if result['prediction'] == 1:
                        st.error("🚨 FRAUDULENT TRANSACTION")
                    else:
                        st.success("✅ LEGITIMATE TRANSACTION")
                    with col2:
                        st.metric("Fraud Probability", f"{result['probability']:.2%}")
                        st.metric("Risk Level", result['risk'])
                        correct = (result["prediction"]==1)==(row["Class"]==1)
                        st.write("✅ Model predicted correctly" if correct else "❌ Model predicted incorrectly")
            else:
                st.error("API is not available.")