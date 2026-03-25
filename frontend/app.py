from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "https://credit-scoring-system-spt2.onrender.com")


@st.cache_data(show_spinner=False)
def load_raw_schema() -> dict[str, Any]:
    res = requests.get(f"{BACKEND_URL}/raw-schema", timeout=10)
    res.raise_for_status()
    return res.json()


def call_predict_raw(raw_fields: dict[str, Any]) -> dict[str, Any]:
    res = requests.post(
        f"{BACKEND_URL}/predict-raw",
        json={"raw_fields": raw_fields},
        timeout=30,
    )
    if res.status_code >= 400:
        try:
            detail = res.json().get("detail", res.text)
        except Exception:
            detail = res.text
        raise RuntimeError(str(detail))
    return res.json()


def render_prediction(result: dict[str, Any]) -> None:
    decision = result["decision"]
    confidence = float(result.get("confidence", 0.0))
    reasoning = html.escape(str(result.get("reasoning", "No reasoning available.")))

    if decision == "Accepted":
        badge_class = "ok"
    elif decision == "Rejected":
        badge_class = "no"
    else:
        badge_class = "neutral"

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-row">
            <span class="badge {badge_class}">Decision: {html.escape(decision)}</span>
            <span class="confidence">Confidence: {confidence:.2%}</span>
          </div>
          <div class="reason">{reasoning}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Credit Decision", layout="centered", page_icon="💳")

st.markdown(
        """
        <style>
            :root {
                --bg: #060b14;
                --surface: #0f1724;
                --ink: #e8f0ff;
                --muted: #9aa8bf;
                --accent: #11d3c6;
                --accent-soft: #103636;
                --danger: #ff6a78;
                --danger-soft: #3a1e27;
                --edge: #1d2b42;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(1000px 500px at 0% 0%, #13233e 0%, rgba(19, 35, 62, 0) 60%),
                    radial-gradient(900px 450px at 100% 0%, #11353d 0%, rgba(17, 53, 61, 0) 60%),
                    var(--bg);
            }

            [data-testid="stHeader"] {
                background: rgba(9, 14, 24, 0.7);
                backdrop-filter: blur(6px);
            }

            [data-testid="stToolbar"] {
                color: #cbd6ea;
            }

            .main .block-container {
                padding-top: 1.4rem;
                max-width: 860px;
            }

            .hero {
                background: linear-gradient(130deg, #0b1221, #15253f 45%, #0f5f66 100%);
                color: #f4fbff;
                padding: 18px 20px;
                border-radius: 14px;
                border: 1px solid #294061;
                margin-bottom: 14px;
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.35);
            }

            .hero h1 {
                margin: 0;
                font-size: 1.4rem;
                letter-spacing: 0.2px;
            }

            .hero p {
                margin: 6px 0 0;
                color: #b5c8e6;
            }

            .result-card {
                margin-top: 14px;
                background: var(--surface);
                border: 1px solid var(--edge);
                border-radius: 14px;
                padding: 14px 16px;
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.3);
            }

            .result-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
            }

            .badge {
                font-weight: 700;
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 0.95rem;
            }

            .badge.ok {
                color: #59f0bf;
                background: #15382d;
            }

            .badge.no {
                color: #ff9aa5;
                background: #3a1e27;
            }

            .badge.neutral {
                color: #d7deeb;
                background: #283347;
            }

            .confidence {
                color: var(--muted);
                font-weight: 600;
            }

            .reason {
                margin-top: 10px;
                color: var(--ink);
                line-height: 1.5;
            }

            [data-testid="stForm"] {
                background: rgba(15, 23, 36, 0.92);
                border: 1px solid var(--edge);
                border-radius: 14px;
                padding: 14px 14px 8px;
                box-shadow: 0 16px 32px rgba(0, 0, 0, 0.35);
            }

            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            label,
            .stNumberInput label,
            .stSelectbox label,
            [data-testid="stSidebar"] * {
                color: #d9e5fb !important;
            }

            [data-testid="stNumberInputContainer"],
            [data-testid="stSelectbox"] > div > div {
                background: #0d1420;
                border: 1px solid #1b2a3f;
                border-radius: 12px;
                box-shadow: none;
            }

            [data-baseweb="input"] input,
            [data-baseweb="select"] div {
                color: #ecf3ff !important;
            }

            .stButton > button,
            [data-testid="stFormSubmitButton"] button {
                border: 0;
                color: #f4fbff;
                background: linear-gradient(120deg, #0f766e, #0d9488);
                border-radius: 11px;
                font-weight: 700;
                letter-spacing: 0.2px;
                box-shadow: 0 10px 20px rgba(13, 148, 136, 0.35);
            }

            .stButton > button:hover,
            [data-testid="stFormSubmitButton"] button:hover {
                background: linear-gradient(120deg, #0c645e, #0b7f76);
            }

            [data-testid="stSidebar"] {
                background: rgba(9, 14, 24, 0.85);
                border-right: 1px solid var(--edge);
            }

            [data-testid="stFormSubmitButton"] {
                margin-top: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
)

st.markdown(
        """
        <div class="hero">
            <h1>Credit Decision Assistant</h1>
            <p>Fill a few details and get a clear approve/reject outcome with plain-language reasoning.</p>
        </div>
        """,
        unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Credit Decision Assistant")
    st.caption("Simple applicant scoring")

try:
    raw_schema = load_raw_schema()
except Exception as exc:
    st.error(f"Could not load schema from backend: {exc}")
    st.stop()

numeric_fields = raw_schema["numeric_fields"]
categorical_fields = raw_schema["categorical_fields"]

def has_field(name: str) -> bool:
    return name in numeric_fields


def has_cat(name: str) -> bool:
    return name in categorical_fields


def map_gender_choice(choice: str, raw_options: list[str]) -> str:
    has_m = "M" in raw_options
    has_f = "F" in raw_options

    if choice == "Male":
        if has_m:
            return "M"
        if has_f:
            return "__baseline__"
    if choice == "Female":
        if has_f:
            return "F"
        if has_m:
            return "__baseline__"

    return "__baseline__"


def map_marital_choice(choice: str, raw_options: list[str]) -> str:
    has_single = "Single" in raw_options
    has_married = "Married" in raw_options

    if choice == "Single":
        if has_single:
            return "Single"
        if has_married:
            return "__baseline__"
    if choice == "Married":
        if has_married:
            return "Married"
        if has_single:
            return "__baseline__"

    return "__baseline__"


with st.container(border=True):
    with st.form("simple_applicant_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18.0, max_value=100.0, value=30.0, step=1.0)
            income = st.number_input("Net Monthly Income", min_value=0.0, value=30000.0, step=500.0)
            delinquent = st.number_input("Times Delinquent", min_value=0.0, value=0.0, step=1.0)

        with col2:
            enq_3m = st.number_input("Inquiries Last 3 Months", min_value=0.0, value=0.0, step=1.0)
            since_pay = st.number_input("Months Since Recent Payment", min_value=0.0, value=6.0, step=1.0)
            since_enq = st.number_input("Months Since Recent Inquiry", min_value=0.0, value=6.0, step=1.0)

        gender_raw_options = categorical_fields.get("GENDER", ["__baseline__"])
        marital_raw_options = categorical_fields.get("MARITALSTATUS", ["__baseline__"])
        education_options = categorical_fields.get("EDUCATION", ["__baseline__"])

        c1, c2, c3 = st.columns(3)
        with c1:
            gender_choice = st.selectbox("Gender", options=["Female", "Male"], index=0)
        with c2:
            marital_choice = st.selectbox("Marital Status", options=["Single", "Married"], index=0)
        with c3:
            education = st.selectbox("Education", options=education_options, index=0)

        submit = st.form_submit_button("Predict", type="primary", width="stretch")

if submit:
    payload: dict[str, Any] = {}

    if has_field("AGE"):
        payload["AGE"] = float(age)
    if has_field("NETMONTHLYINCOME"):
        payload["NETMONTHLYINCOME"] = float(income)
    if has_field("num_times_delinquent"):
        payload["num_times_delinquent"] = float(delinquent)
    if has_field("enq_L3m"):
        payload["enq_L3m"] = float(enq_3m)
    if has_field("time_since_recent_payment"):
        payload["time_since_recent_payment"] = float(since_pay)
    if has_field("time_since_recent_enq"):
        payload["time_since_recent_enq"] = float(since_enq)

    if has_cat("GENDER"):
        payload["GENDER"] = map_gender_choice(gender_choice, gender_raw_options)
    if has_cat("MARITALSTATUS"):
        payload["MARITALSTATUS"] = map_marital_choice(marital_choice, marital_raw_options)
    if has_cat("EDUCATION"):
        payload["EDUCATION"] = education

    try:
        result = call_predict_raw(payload)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        render_prediction(result)

st.caption("All other model inputs are automatically defaulted by the backend.")
