"""api.py — FastAPI backend for the Churn Prediction web app.

Run from the project root:
    uvicorn app.api:app --reload --port 8000

Endpoints:
    GET  /health                → model status
    POST /predict               → single customer prediction
    GET  /portfolio/summary     → dataset-wide risk analysis
    GET  /portfolio/customers   → paginated customer list with scores
    POST /whatif                → what-if scenario comparison
    POST /campaign/optimize     → budget-constrained campaign optimizer
    GET  /models/comparison     → all model results
    GET  /models/diversity      → pairwise diversity data
    GET  /models/calibration    → calibration summary
    GET  /models/des            → DES results
    GET  /reports/{filename}    → serve report images
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from Common.feature_engineering import engineer_features

ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")
DATA_DIR = Path("Data")

# ── Load artifacts ───────────────────────────────────────────────────────────

model = joblib.load(ARTIFACTS / "best_model.joblib")
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
feature_names: list[str] = json.loads((ARTIFACTS / "feature_names.json").read_text())
model_info: dict = json.loads((ARTIFACTS / "best_model_info.json").read_text())

THRESHOLD = float(model_info.get("threshold_f1", 0.5))
COST_FN = float(model_info.get("cost_fn", 500))
COST_FP = float(model_info.get("cost_fp", 50))


# ── Precompute portfolio scores (cached at startup) ─────────────────────────

def _load_portfolio() -> pd.DataFrame:
    """Load the full dataset and score every customer."""
    df = pd.read_csv(DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Drop customerID/Churn, then apply feature engineering
    feature_cols = [c for c in df.columns if c.lower() not in ("customerid", "churn")]
    X_raw = engineer_features(df[feature_cols].copy())
    X_transformed = preprocessor.transform(X_raw)
    probas = model.predict_proba(X_transformed)[:, 1]

    df = df.copy()
    df["churn_probability"] = probas
    df["risk_tier"] = pd.cut(
        probas,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    df["predicted_churn"] = (probas >= THRESHOLD).astype(int)
    return df


PORTFOLIO = _load_portfolio()


# ── SHAP helper ──────────────────────────────────────────────────────────────

def _shap_top_factors(X: np.ndarray, n: int = 8) -> list[dict]:
    try:
        import shap

        inner = model.named_steps["model"]
        explainer = shap.TreeExplainer(inner)
        sv = explainer(X)[0].values
        if sv.ndim > 1:
            sv = sv[:, 1]

        top_idx = np.argsort(np.abs(sv))[::-1][:n]
        return [
            {
                "feature": feature_names[i],
                "shap_value": round(float(sv[i]), 4),
                "direction": "increases churn" if sv[i] > 0 else "decreases churn",
            }
            for i in top_idx
        ]
    except Exception:
        return []


def _risk_level(p: float) -> str:
    if p < 0.3:
        return "low"
    if p < 0.6:
        return "medium"
    return "high"


def _recommendation(p: float) -> str:
    if p < 0.3:
        return "No action needed. Monitor quarterly."
    if p < 0.6:
        return "Send proactive retention email with loyalty discount (5-10%)."
    return "Escalate to account manager. Offer contract upgrade or significant discount."


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0.0, examples=[29.85])
    TotalCharges: float = Field(..., ge=0.0, examples=[29.85])
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class WhatIfRequest(BaseModel):
    original: CustomerFeatures
    modified: CustomerFeatures


class CampaignRequest(BaseModel):
    budget: float = Field(..., gt=0, examples=[50000])
    offer_cost: float = Field(50.0, gt=0, examples=[50])
    min_probability: float = Field(0.0, ge=0.0, le=1.0)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "model": model_info.get("name"),
        "threshold": THRESHOLD,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "total_customers": len(PORTFOLIO),
    }


@app.post("/predict", tags=["Prediction"])
def predict(customer: CustomerFeatures):
    try:
        df = pd.DataFrame([customer.model_dump(exclude={"threshold"})])
        df = engineer_features(df)
        X = preprocessor.transform(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}")

    proba = float(model.predict_proba(X)[0, 1])
    t = customer.threshold if customer.threshold is not None else THRESHOLD

    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": proba >= t,
        "risk_level": _risk_level(proba),
        "threshold_used": round(t, 4),
        "top_risk_factors": _shap_top_factors(X),
        "recommended_action": _recommendation(proba),
        "business_context": {
            "cost_of_false_negative": COST_FN,
            "cost_of_false_positive": COST_FP,
            "expected_cost": COST_FP if proba >= t else round(proba * COST_FN, 2),
        },
    }


# ── Portfolio ────────────────────────────────────────────────────────────────

@app.get("/portfolio/summary", tags=["Portfolio"])
def portfolio_summary():
    df = PORTFOLIO
    total = len(df)
    actual_churn = int((df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})).sum())

    tiers = df["risk_tier"].value_counts().to_dict()
    probas = df["churn_probability"]

    # Histogram bins
    hist_counts, hist_edges = np.histogram(probas, bins=20, range=(0, 1))

    return {
        "total_customers": total,
        "actual_churn_count": actual_churn,
        "actual_churn_rate": round(actual_churn / total, 4),
        "predicted_churn_count": int(df["predicted_churn"].sum()),
        "risk_tiers": {k: int(v) for k, v in tiers.items()},
        "avg_probability": round(float(probas.mean()), 4),
        "total_expected_cost": round(float(probas.sum() * COST_FN), 0),
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": [round(float(e), 2) for e in hist_edges.tolist()],
        },
        "monthly_charges_by_risk": {
            tier: round(float(df[df["risk_tier"] == tier]["MonthlyCharges"].mean()), 2)
            for tier in ["low", "medium", "high"]
            if tier in df["risk_tier"].values
        },
        "contract_distribution": {
            tier: df[df["risk_tier"] == tier]["Contract"].value_counts().to_dict()
            for tier in ["low", "medium", "high"]
            if tier in df["risk_tier"].values
        },
    }


@app.get("/portfolio/customers", tags=["Portfolio"])
def portfolio_customers(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    risk_tier: Optional[str] = Query(None),
    sort_by: str = Query("churn_probability"),
    sort_desc: bool = Query(True),
):
    df = PORTFOLIO.copy()

    if risk_tier and risk_tier in ("low", "medium", "high"):
        df = df[df["risk_tier"] == risk_tier]

    ascending = not sort_desc
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]

    records = []
    for _, row in page_df.iterrows():
        records.append({
            "customerID": row.get("customerID", ""),
            "gender": row.get("gender", ""),
            "tenure": int(row.get("tenure", 0)),
            "Contract": row.get("Contract", ""),
            "MonthlyCharges": float(row.get("MonthlyCharges", 0)),
            "TotalCharges": float(row.get("TotalCharges", 0)),
            "InternetService": row.get("InternetService", ""),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "risk_tier": str(row["risk_tier"]),
            "predicted_churn": bool(row["predicted_churn"]),
            "actual_churn": row.get("Churn", "Unknown"),
        })

    return {
        "customers": records,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


# ── What-If ──────────────────────────────────────────────────────────────────

@app.post("/whatif", tags=["What-If"])
def whatif(req: WhatIfRequest):
    try:
        df_orig = engineer_features(pd.DataFrame([req.original.model_dump(exclude={"threshold"})]))
        df_mod = engineer_features(pd.DataFrame([req.modified.model_dump(exclude={"threshold"})]))
        X_orig = preprocessor.transform(df_orig)
        X_mod = preprocessor.transform(df_mod)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    p_orig = float(model.predict_proba(X_orig)[0, 1])
    p_mod = float(model.predict_proba(X_mod)[0, 1])

    return {
        "original": {
            "churn_probability": round(p_orig, 4),
            "risk_level": _risk_level(p_orig),
            "top_factors": _shap_top_factors(X_orig),
        },
        "modified": {
            "churn_probability": round(p_mod, 4),
            "risk_level": _risk_level(p_mod),
            "top_factors": _shap_top_factors(X_mod),
        },
        "delta": round(p_mod - p_orig, 4),
        "recommendation": _recommendation(p_mod),
    }


# ── Campaign Optimizer ───────────────────────────────────────────────────────

@app.post("/campaign/optimize", tags=["Campaign"])
def campaign_optimize(req: CampaignRequest):
    df = PORTFOLIO.copy()
    probas = df["churn_probability"].values

    # Filter by min probability
    mask = probas >= req.min_probability
    df_eligible = df[mask].copy()
    eligible_probas = df_eligible["churn_probability"].values

    # Sort by expected value of intervention: p(churn) * cost_fn - offer_cost
    expected_savings = eligible_probas * COST_FN - req.offer_cost
    df_eligible = df_eligible.copy()
    df_eligible["expected_savings"] = expected_savings
    df_eligible = df_eligible[df_eligible["expected_savings"] > 0]
    df_eligible = df_eligible.sort_values("expected_savings", ascending=False)

    # Constrain by budget
    max_offers = int(req.budget // req.offer_cost)
    targeted = df_eligible.head(max_offers)

    total_campaign_cost = len(targeted) * req.offer_cost
    expected_retained = float(targeted["churn_probability"].sum())
    expected_savings_total = expected_retained * COST_FN
    net_roi = expected_savings_total - total_campaign_cost

    # Build targeting summary
    if len(targeted) > 0:
        tier_counts = targeted["risk_tier"].value_counts().to_dict()
        contract_counts = targeted["Contract"].value_counts().to_dict()
    else:
        tier_counts = {}
        contract_counts = {}

    # ROI curve: what happens at different budget levels
    roi_curve = []
    for budget_pct in range(5, 105, 5):
        budget_level = req.budget * budget_pct / 100
        n_offers = int(budget_level // req.offer_cost)
        subset = df_eligible.head(n_offers)
        cost = len(subset) * req.offer_cost
        savings = float(subset["churn_probability"].sum()) * COST_FN
        roi_curve.append({
            "budget_pct": budget_pct,
            "budget": round(budget_level, 0),
            "customers_targeted": len(subset),
            "expected_savings": round(savings, 0),
            "campaign_cost": round(cost, 0),
            "net_roi": round(savings - cost, 0),
        })

    top_targets = []
    for _, row in targeted.head(20).iterrows():
        top_targets.append({
            "customerID": row.get("customerID", ""),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "risk_tier": str(row["risk_tier"]),
            "Contract": row.get("Contract", ""),
            "MonthlyCharges": float(row.get("MonthlyCharges", 0)),
            "expected_savings": round(float(row["expected_savings"]), 2),
        })

    return {
        "summary": {
            "total_eligible": len(df_eligible),
            "customers_targeted": len(targeted),
            "campaign_cost": round(total_campaign_cost, 0),
            "expected_retained": round(expected_retained, 1),
            "expected_savings": round(expected_savings_total, 0),
            "net_roi": round(net_roi, 0),
            "roi_pct": round(net_roi / total_campaign_cost * 100, 1) if total_campaign_cost > 0 else 0,
        },
        "tier_breakdown": {k: int(v) for k, v in tier_counts.items()},
        "contract_breakdown": {k: int(v) for k, v in contract_counts.items()},
        "roi_curve": roi_curve,
        "top_targets": top_targets,
    }


# ── Model Observatory ────────────────────────────────────────────────────────

@app.get("/models/comparison", tags=["Models"])
def models_comparison():
    results = json.loads((ARTIFACTS / "results.json").read_text())
    rows = []
    for name, r in results.items():
        d = r["test"]
        rows.append({
            "name": name,
            "roc_auc": d["default"].get("roc_auc"),
            "pr_auc": d["default"].get("pr_auc"),
            "f1": d["f1"].get("f1"),
            "recall": d["f1"].get("recall"),
            "precision": d["f1"].get("precision"),
            "accuracy": d["default"].get("accuracy"),
            "biz_threshold": r.get("business", {}).get("threshold"),
            "biz_cost": r.get("business", {}).get("cost"),
        })
    rows.sort(key=lambda x: x["f1"] or 0, reverse=True)
    return {"models": rows, "best_model": model_info.get("name")}


@app.get("/models/diversity", tags=["Models"])
def models_diversity():
    csv_path = REPORTS / "diversity_pairwise.csv"
    if not csv_path.exists():
        return {"pairs": [], "subset": []}

    pairs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "q_statistic": round(float(row["q_statistic"]), 4),
                "disagreement": round(float(row["disagreement"]), 4),
                "double_fault": round(float(row["double_fault"]), 4),
            })

    subset_info = json.loads((ARTIFACTS / "diverse_subset.json").read_text())

    return {"pairs": pairs, "subset": subset_info}


@app.get("/models/calibration", tags=["Models"])
def models_calibration():
    csv_path = REPORTS / "calibration_summary.csv"
    if not csv_path.exists():
        return {"entries": []}

    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "model": row["model"],
                "variant": row["variant"],
                "ece": round(float(row["ece"]), 6),
                "brier_score": round(float(row["brier_score"]), 6),
            })

    cal_stack = {}
    cal_stack_path = ARTIFACTS / "calibrated_stacking_results.json"
    if cal_stack_path.exists():
        cal_stack = json.loads(cal_stack_path.read_text())

    return {"entries": entries, "calibrated_stacking": cal_stack}


@app.get("/models/des", tags=["Models"])
def models_des():
    des_path = ARTIFACTS / "des_results.json"
    if not des_path.exists():
        return {"methods": []}

    raw = json.loads(des_path.read_text())
    methods = []
    for name, m in raw.items():
        methods.append({
            "name": name,
            "f1": round(m.get("f1", 0), 4),
            "recall": round(m.get("recall", 0), 4),
            "precision": round(m.get("precision", 0), 4),
            "roc_auc": round(m.get("roc_auc", 0), 4),
            "accuracy": round(m.get("accuracy", 0), 4),
        })

    return {"methods": methods}


# ── Revenue Impact ──────────────────────────────────────────────────────────

def _expected_lifetime_months(contract: str) -> float:
    """Estimate remaining lifetime by contract type (conservative)."""
    mapping = {"Month-to-month": 6, "One year": 18, "Two year": 30}
    return mapping.get(contract, 12)


@app.get("/revenue/impact", tags=["Revenue"])
def revenue_impact():
    df = PORTFOLIO.copy()

    # Revenue at risk per customer: p(churn) × monthly_charges × expected_lifetime
    df["lifetime_months"] = df["Contract"].map(
        lambda c: _expected_lifetime_months(c)
    )
    df["annual_revenue"] = df["MonthlyCharges"] * 12
    df["revenue_at_risk"] = (
        df["churn_probability"] * df["MonthlyCharges"] * df["lifetime_months"]
    )

    total_revenue_at_risk = float(df["revenue_at_risk"].sum())
    total_annual_revenue = float(df["annual_revenue"].sum())

    # Top 50 at-risk customers by revenue
    top50 = df.nlargest(50, "revenue_at_risk")
    top50_revenue = float(top50["revenue_at_risk"].sum())
    top50_annual = float(top50["annual_revenue"].sum())

    top_customers = []
    for _, row in top50.iterrows():
        top_customers.append({
            "customerID": row.get("customerID", ""),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "risk_tier": str(row["risk_tier"]),
            "Contract": row.get("Contract", ""),
            "MonthlyCharges": float(row["MonthlyCharges"]),
            "annual_revenue": round(float(row["annual_revenue"]), 2),
            "lifetime_months": int(row["lifetime_months"]),
            "revenue_at_risk": round(float(row["revenue_at_risk"]), 2),
        })

    # Revenue at risk by tier
    tier_revenue = {}
    for tier in ["low", "medium", "high"]:
        subset = df[df["risk_tier"] == tier]
        if len(subset) > 0:
            tier_revenue[tier] = {
                "count": len(subset),
                "total_annual_revenue": round(float(subset["annual_revenue"].sum()), 0),
                "revenue_at_risk": round(float(subset["revenue_at_risk"].sum()), 0),
            }

    # Revenue at risk by contract type
    contract_revenue = {}
    for contract in df["Contract"].unique():
        subset = df[df["Contract"] == contract]
        contract_revenue[contract] = {
            "count": len(subset),
            "revenue_at_risk": round(float(subset["revenue_at_risk"].sum()), 0),
            "avg_churn_prob": round(float(subset["churn_probability"].mean()), 4),
        }

    # Revenue at risk distribution (histogram)
    rar_values = df["revenue_at_risk"].values
    hist_counts, hist_edges = np.histogram(
        rar_values[rar_values > 0], bins=20
    )

    return {
        "total_revenue_at_risk": round(total_revenue_at_risk, 0),
        "total_annual_revenue": round(total_annual_revenue, 0),
        "pct_revenue_at_risk": round(
            total_revenue_at_risk / total_annual_revenue * 100, 1
        )
        if total_annual_revenue > 0
        else 0,
        "top50_revenue_at_risk": round(top50_revenue, 0),
        "top50_annual_revenue": round(top50_annual, 0),
        "top_customers": top_customers,
        "tier_revenue": tier_revenue,
        "contract_revenue": contract_revenue,
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": [round(float(e), 2) for e in hist_edges.tolist()],
        },
    }


# ── A/B Test Simulator ─────────────────────────────────────────────────────

class ABTestRequest(BaseModel):
    strategy_a_name: str = Field("20% Discount", examples=["20% Discount"])
    strategy_a_cost: float = Field(50.0, gt=0, examples=[50])
    strategy_a_effectiveness: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Fraction of at-risk customers retained by strategy A",
    )
    strategy_b_name: str = Field("Free Tech Support", examples=["Free Tech Support"])
    strategy_b_cost: float = Field(25.0, gt=0, examples=[25])
    strategy_b_effectiveness: float = Field(
        0.2, ge=0.0, le=1.0,
        description="Fraction of at-risk customers retained by strategy B",
    )
    min_probability: float = Field(0.3, ge=0.0, le=1.0)
    budget: float = Field(50000, gt=0)


@app.post("/abtest/simulate", tags=["A/B Test"])
def abtest_simulate(req: ABTestRequest):
    df = PORTFOLIO.copy()
    at_risk = df[df["churn_probability"] >= req.min_probability].copy()
    at_risk["lifetime_months"] = at_risk["Contract"].map(
        lambda c: _expected_lifetime_months(c)
    )
    at_risk["revenue_at_risk"] = (
        at_risk["churn_probability"]
        * at_risk["MonthlyCharges"]
        * at_risk["lifetime_months"]
    )

    total_at_risk = len(at_risk)

    def _simulate_strategy(
        name: str, cost_per: float, effectiveness: float
    ) -> dict:
        max_offers = int(req.budget // cost_per)
        # Prioritise by revenue at risk
        targeted = at_risk.nlargest(min(max_offers, total_at_risk), "revenue_at_risk")

        campaign_cost = len(targeted) * cost_per
        retained_count = effectiveness * len(targeted)
        revenue_saved = float(targeted["revenue_at_risk"].sum()) * effectiveness
        net_roi = revenue_saved - campaign_cost

        # Break down by risk tier
        tier_counts = targeted["risk_tier"].value_counts().to_dict()
        contract_counts = targeted["Contract"].value_counts().to_dict()

        return {
            "name": name,
            "cost_per_customer": cost_per,
            "effectiveness": effectiveness,
            "customers_targeted": len(targeted),
            "campaign_cost": round(campaign_cost, 0),
            "estimated_retained": round(retained_count, 1),
            "revenue_saved": round(revenue_saved, 0),
            "net_roi": round(net_roi, 0),
            "roi_pct": round(net_roi / campaign_cost * 100, 1) if campaign_cost > 0 else 0,
            "tier_breakdown": {k: int(v) for k, v in tier_counts.items()},
            "contract_breakdown": {k: int(v) for k, v in contract_counts.items()},
        }

    strat_a = _simulate_strategy(req.strategy_a_name, req.strategy_a_cost, req.strategy_a_effectiveness)
    strat_b = _simulate_strategy(req.strategy_b_name, req.strategy_b_cost, req.strategy_b_effectiveness)

    # Sensitivity: vary effectiveness from 10% to 50% for both
    sensitivity = []
    for eff_pct in range(10, 55, 5):
        eff = eff_pct / 100
        max_a = int(req.budget // req.strategy_a_cost)
        max_b = int(req.budget // req.strategy_b_cost)
        tgt_a = at_risk.nlargest(min(max_a, total_at_risk), "revenue_at_risk")
        tgt_b = at_risk.nlargest(min(max_b, total_at_risk), "revenue_at_risk")

        rev_a = float(tgt_a["revenue_at_risk"].sum()) * eff
        rev_b = float(tgt_b["revenue_at_risk"].sum()) * eff
        cost_a = len(tgt_a) * req.strategy_a_cost
        cost_b = len(tgt_b) * req.strategy_b_cost

        sensitivity.append({
            "effectiveness_pct": eff_pct,
            "strategy_a_roi": round(rev_a - cost_a, 0),
            "strategy_b_roi": round(rev_b - cost_b, 0),
        })

    winner = strat_a["name"] if strat_a["net_roi"] > strat_b["net_roi"] else strat_b["name"]

    return {
        "total_at_risk": total_at_risk,
        "strategy_a": strat_a,
        "strategy_b": strat_b,
        "winner": winner,
        "sensitivity": sensitivity,
    }


# ── Reports (serve images) ───────────────────────────────────────────────────

@app.get("/reports/{filename}", tags=["Reports"])
def get_report(filename: str):
    safe_name = Path(filename).name
    path = REPORTS / safe_name
    if not path.exists() or not path.suffix in (".png", ".csv", ".json"):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(path))


# ── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
