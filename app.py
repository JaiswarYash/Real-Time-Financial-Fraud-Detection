# app
import streamlit as st
import requests
import pandas as pd


@st.cache_data
def load_data():
    return pd.read_csv("artifacts/test.csv")

test_data = load_data()


# app config
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Financial Fraud Detection", page_icon=":money_with_wings:", layout="wide", initial_sidebar_state="collapsed")
st.title("Financial Fraud Detection App")

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
    


# V1-V28 features

if st.button("Analyse Random Transaction"):
    random_row = test_data.sample(1).iloc[0]
    st.session_state.v_features = {f"V{i}": random_row[f"V{i}"] for i in range(1, 29)}
    Amount = random_row["Amount"]
    payload = {
    "Time": random_row["Time"],
    **st.session_state.v_features,
    "Amount": Amount
    }
    if check_api_status():
        result = predict_fraud(payload)
        if result is not None:
            if result['prediction'] == 1:
                st.error(f"🚨 FRAUDULENT TRANSACTION")
            else:
                st.success(f"✅ LEGITIMATE TRANSACTION")

            st.metric("Fraud Probability", f"{result['probability']:.2%}")
            st.metric("Risk Level", result['risk'])
            actual = "Fraud" if random_row['Class'] == 1 else "Legitimate"
            st.info(f"Actual Label: {actual}")
    else:
        st.error("API is not available. Please start the FastAPI server on localhost:8000")