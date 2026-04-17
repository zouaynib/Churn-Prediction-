#  Ensemble Learning for Customer Churn Prediction  
### A Diversity-Driven Approach with Business Decision Support

An end-to-end machine learning and decision intelligence platform for telecom churn prediction that combines **ensemble learning**, **model diversity analysis**, **probability calibration**, and **cost-sensitive optimization**.

Built as both a **research project** and a **production-style web platform** with real deployment architecture.

---


In telecom, acquiring a new customer can cost **5–7× more** than retaining one.

With a churn rate of **26.5%**, companies lose significant lifetime revenue every year.

The key challenge is not only predicting churn accurately—but making **reliable, actionable, and cost-effective decisions** from those predictions.

### Research Question

> Can diversity-aware ensemble learning outperform individual models and deliver reliable, business-ready churn predictions?

---

# 📊 Dataset

**Telco Customer Churn Dataset**

- **7,043 customers**
- **21 raw features**
- **1 binary target:** Churn (Yes / No)

### Class Distribution

- **5,174 retained**
- **1,869 churned**
- **26.5% churn rate**

### Data Split

- Train: **70%**
- Validation: **10%**
- Test: **20%**

---

# 🧠 Feature Engineering

Beyond the raw dataset, 9 additional business-driven features were engineered.

### Engineered Features

- `AvgMonthlyCharge`
- `ChargeIncrease`
- `MonthlyToTotalRatio`
- `tenure_bin` (4 segments)
- `IsNewCustomer` (≤ 6 months)
- `NumServices`
- `HasSecurity`
- `HasSupport`
- `AutoPay`

### Final Pipeline

```text
21 Raw Features
+ 9 Engineered Features
= 30 Features Ready for Modeling
```

### Preprocessing

- One-hot encoding for categorical variables
- Scaling for numeric features
- Reusable fitted preprocessing pipeline

---

# 🏗️ Model Architecture

This project compares **14 models** across **4 learning families**.

## 1️⃣ Tree-Based Models

- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- XGBoost
- LightGBM
- Bagging

## 2️⃣ Linear Models

- Logistic Regression

## 3️⃣ Kernel Models

- SVM (RBF)

## 4️⃣ Neural Models

- MLP (64 → 32)

---

# 🤝 Ensemble Methods

## Stacking

- 6 base learners
- Logistic Regression meta-model

## Soft Voting

- 6 models

## Hard Voting

- 7 models

---

# 🔬 Core Innovation: Diversity-Driven Ensembles

Most churn projects combine models blindly.

This project studies whether models make **different errors**, because ensembles only improve when members are diverse.

### Key Metrics Used

- Q-Statistic
- Disagreement Rate
- Double-Fault Rate

---

# 💡 Major Finding

Tree-based models were highly redundant:

```text
Q-Statistic > 0.95
```

After adding Linear, Kernel, and Neural learners:

```text
Q drops to 0.663
```

Meaningfully different error patterns created stronger ensembles.

---

# 🏆 Best Insight

The maximally diverse pair:

```text
{Extra Trees + Logistic Regression}
```

achieved:

```text
ROC-AUC = 0.863
```

This matched the full 6-model stacking ensemble using only **2 models**.

### Conclusion:

> Diversity matters more than quantity.

---

# 📈 Results

| Model | F1 | Recall | Precision | ROC-AUC | PR-AUC |
|------|----|--------|-----------|--------|--------|
| Calibrated Stacking | **0.756** | 0.744 | 0.586 | **0.863** | 0.698 |
| Random Forest | 0.658 | 0.779 | 0.570 | 0.861 | 0.699 |
| Stacking | 0.647 | 0.784 | 0.551 | 0.863 | 0.703 |
| Logistic Regression | 0.634 | **0.817** | 0.518 | 0.861 | **0.808** |
| SVM | 0.630 | 0.623 | 0.638 | 0.845 | 0.632 |

---

# 🎯 Probability Calibration

High AUC is not enough.  
Businesses need probabilities they can trust.

### Example: Stacking Model

```text
ECE: 0.152 → 0.025
```

Using isotonic calibration (**6× improvement**)

### Best Native Calibration

```text
MLP: ECE = 0.014
```

### Why It Matters

If the model predicts 80% churn risk, that probability should reflect reality.

---

# 💰 Cost-Sensitive Threshold Optimization

The default threshold of **0.50** is rarely optimal.

### Business Assumptions

- False Negative (missed churner): **$500**
- False Positive (unnecessary intervention): **$50**

### Optimal Thresholds

- F1-optimal: **0.397**
- Business-optimal: **0.14**

### Outcome

✅ **26% reduction in business cost**

---

# 🌐 Production Web Platform

A complete deployment system was built around the models.

## Backend

- FastAPI
- 16 endpoints
- Saved joblib model artifacts

## Frontend

- Next.js
- React
- shadcn/ui
- Recharts

---

# 🖥️ Platform Pages (10 Modules)

## Customer Intelligence

- Executive Summary
- Portfolio Dashboard
- Customer Lookup
- Watch List
- Segment Explorer

## Retention & Analytics Tools

- What-If Simulator
- Campaign Optimizer
- A/B Test Simulator
- Revenue Impact
- Model Observatory

---

# 🧪 Example API Endpoints

```text
GET  /health
POST /predict
GET  /portfolio/summary
GET  /portfolio/customers
POST /whatif
POST /campaign/optimize
GET  /models/comparison
GET  /models/diversity
GET  /models/calibration
GET  /models/des
```

---

# 🗂️ Project Structure

```text
Churn-Prediction/
├── backend/              # FastAPI API
├── frontend/             # Next.js dashboard
├── models/               # Training pipelines
├── common/               # Shared utilities
├── artifacts/            # Saved models + preprocessors
├── reports/              # ROC / PR / Cost / Calibration plots
├── train.py              # Full training pipeline
├── tune.py               # Hyperparameter tuning
└── README.md
```

---

# 🚀 Quick Start

## Clone Repo

```bash
git clone <your-repo-url>
cd Churn-Prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python train.py
```

## Run Backend

```bash
uvicorn backend.api:app --reload
```

## Run Frontend

```bash
npm install
npm run dev
```

---

# 📌 Key Contributions

### ✅ Diversity-Driven Ensemble Design

2 models matched a 6-model stack.

### ✅ 4-Family Benchmark

Tree, Linear, Kernel, Neural compared on one pipeline.

### ✅ Production Calibration

6× probability reliability improvement.

### ✅ Cost-Sensitive Deployment

26% business cost reduction.

### ✅ Full-Stack Platform

14 models, 16 APIs, 10 pages.

---

# 🔭 Future Work

- Temporal customer behavior features
- SMOTE / ADASYN imbalance handling
- Larger DSEL for Dynamic Ensemble Selection
- Optuna hyperparameter search
- Cloud deployment
- Real-time inference pipelines
- Drift monitoring

---

# ⭐ If You Like This Project

Give it a star and connect with me.
