"""streamlit_app.py — Interactive churn prediction demo.

Run from the project root:
    streamlit run app/streamlit_app.py

Requires artifacts produced by train.py:
    artifacts/best_model.joblib
    artifacts/preprocessor.joblib
    artifacts/feature_names.json
    artifacts/best_model_info.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from the project root when Streamlit shifts cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📡",
    layout="wide",
)


# ── Load artifacts (cached so they load only once) ───────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS / "best_model.joblib")
    preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
    feature_names = json.loads((ARTIFACTS / "feature_names.json").read_text())
    info = json.loads((ARTIFACTS / "best_model_info.json").read_text())
    return model, preprocessor, feature_names, info


@st.cache_resource
def get_shap_explainer(_model):
    """Build a SHAP explainer; returns None on failure."""
    try:
        import shap
        inner = _model.named_steps["model"]
        if hasattr(inner, "estimators_") or hasattr(inner, "get_booster"):
            return shap.TreeExplainer(inner)
        return shap.Explainer(inner)
    except Exception:
        return None


model, preprocessor, feature_names, model_info = load_artifacts()
explainer = get_shap_explainer(model)
THRESHOLD = float(model_info.get("threshold_f1", 0.5))

# ── Sidebar — customer input ─────────────────────────────────────────────────
st.sidebar.header("📋 Customer Profile")


def customer_input() -> pd.DataFrame:
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.sidebar.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.sidebar.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )
    device_protection = st.sidebar.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.sidebar.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )
    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges = monthly_charges * tenure  # derived

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )


df_input = customer_input()

# ── Predict ──────────────────────────────────────────────────────────────────
X = preprocessor.transform(df_input)
proba = float(model.predict_proba(X)[0, 1])
predicted_churn = proba >= THRESHOLD

# ── Main layout ──────────────────────────────────────────────────────────────
st.title("📡 Telecom Churn Prediction")
st.caption(
    f"Model: **{model_info['name']}**  |  "
    f"Decision threshold: **{THRESHOLD:.2f}** (tuned on validation F1)"
)

col_profile, col_risk = st.columns([1, 2])

with col_profile:
    st.subheader("Customer Summary")
    st.dataframe(
        df_input.T.rename(columns={0: "Value"}),
        use_container_width=True,
    )

with col_risk:
    st.subheader("Churn Risk Assessment")

    if proba < 0.3:
        risk_label, color = "🟢  LOW RISK", "green"
    elif proba < 0.6:
        risk_label, color = "🟡  MEDIUM RISK", "orange"
    else:
        risk_label, color = "🔴  HIGH RISK", "red"

    st.metric(label="Churn Probability", value=f"{proba:.1%}")
    st.progress(proba)
    st.markdown(f"### {risk_label}")
    st.markdown(
        f"**Prediction:** {'⚠️ Likely to Churn' if predicted_churn else '✅ Likely to Stay'}"
    )

    cost_fn = model_info.get("cost_fn", 500)
    cost_fp = model_info.get("cost_fp", 50)
    if predicted_churn:
        st.info(
            f"Retention offer cost: **${cost_fp}**.  "
            f"If missed and they churn: **${cost_fn} loss**."
        )

# ── SHAP explanation ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Why this prediction?")

if explainer is not None:
    try:
        import shap

        shap_values = explainer(X)
        sv = shap_values[0].values if hasattr(shap_values[0], "values") else shap_values[0]
        if sv.ndim > 1:
            sv = sv[:, 1]

        top_n = 8
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        top_features = [feature_names[i] for i in top_idx]
        top_vals = sv[top_idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_vals]
        ax.barh(top_features[::-1], top_vals[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value  (impact on churn probability)")
        ax.set_title("Top Features Driving This Prediction")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption("🔴 Red = increases churn risk  |  🟢 Green = decreases churn risk")

    except Exception as exc:
        st.warning(f"SHAP explanation unavailable: {exc}")
else:
    st.info("SHAP explainer not available for this model type.")

# ── Retention recommendation ─────────────────────────────────────────────────
st.divider()
st.subheader("💼 Recommended Action")

if proba < 0.3:
    st.success("✅ No immediate action needed. Monitor this customer quarterly.")
elif proba < 0.6:
    st.warning(
        "📧 Send a proactive retention email with a loyalty discount (5–10%). "
        "Consider highlighting service upgrades relevant to their current plan."
    )
else:
    st.error(
        "📞 Escalate to account manager immediately. "
        "Offer a contract upgrade, significant discount, or personalised bundle."
    )

# ── Model-level reports (if available) ───────────────────────────────────────
st.divider()
st.subheader("📊 Model Reports")

report_cols = st.columns(3)
report_files = {
    "ROC Curves": REPORTS / "roc_curves.png",
    "PR Curves": REPORTS / "pr_curves.png",
    "Cost Analysis": REPORTS / "cost_analysis.png",
}

for col, (title, path) in zip(report_cols, report_files.items()):
    with col:
        if path.exists():
            st.image(str(path), caption=title, use_container_width=True)
        else:
            st.info(f"{title} not generated yet.\nRun `python train.py` first.")
