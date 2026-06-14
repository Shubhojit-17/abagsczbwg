# SentinelIQ - LLM-Powered Identity Risk Analyst

## 🛡️ Overview

SentinelIQ is an enterprise-grade **Identity Security Analytics Platform** that detects identity sprawl, stale accounts, excessive privileges, privilege abuse, and anomalous access patterns using a combination of:

1. **Rule-Based Detection** - 7 specialized security rules
2. **Machine Learning** - Isolation Forest anomaly detection
3. **Context Intelligence** - Role exceptions, calendar awareness, contractor rules
4. **LLM Explainability** - AI-powered investigation reports
5. **Interactive Dashboard** - 5-page Streamlit analytics dashboard
6. **Privilege Graph** - NetworkX-based relationship visualization
7. **Compliance Mapping** - NIST AC-2 and GDPR Article 32

---

## 🏗️ Architecture

```
Data CSVs → Validation → Feature Engineering → Rule Engine + ML → Risk Scoring → Context Layer → LLM Analysis → Dashboard
```

### Risk Scoring Formula
```
Final Risk = 40% Rule Score + 35% ML Score + 25% Context Score
```

### Risk Levels
| Score Range | Level |
|------------|-------|
| 0-30 | LOW |
| 31-60 | MEDIUM |
| 61-80 | HIGH |
| 81-100 | CRITICAL |

---

## 📁 Project Structure

```
sentineliq/
├── data/                    # Input CSV data files
├── src/
│   ├── ingestion/          # Data loading & validation
│   ├── features/           # Feature engineering
│   ├── rules/              # 7 detection rules
│   ├── ml/                 # Isolation Forest anomaly detection
│   ├── context/            # Context-aware adjustments
│   ├── scoring/            # Risk score calculation
│   ├── graph/              # Privilege graph & blast radius
│   ├── explainability/     # LLM analysis & recommendations
│   ├── reporting/          # Report generation
│   └── evaluation/         # Detection quality metrics
├── api/                    # FastAPI backend
├── dashboard/              # Streamlit dashboard
├── tests/                  # Unit tests
├── outputs/                # Generated reports
├── pipeline.py             # Main orchestration engine
└── requirements.txt        # Dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd sentineliq
pip install -r requirements.txt
```

### 2. Place Data Files

Copy your CSV files to the `data/` directory:
- `data/identity_users.csv`
- `data/identity_events.csv`

### 3. Run Analysis Pipeline

```bash
python pipeline.py --data-dir data --output-dir outputs
```

### 4. Start Dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Start API Server

```bash
cd api
uvicorn main:app --reload --port 8000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Run full analysis pipeline |
| GET | `/summary` | Executive dashboard summary |
| GET | `/top-risks` | Top N risky users |
| GET | `/report/{user_id}` | Full investigation report |
| GET | `/users` | List users with filters |
| GET | `/findings` | Get risk findings |
| GET | `/graph` | Privilege graph data |
| GET | `/graph/blast-radius/{user_id}` | Blast radius analysis |
| GET | `/metrics` | Evaluation metrics |
| POST | `/feedback` | Submit analyst feedback |
| GET | `/health` | Health check |

---

## 📊 Dashboard Pages

1. **Executive Overview** - KPIs, risk distribution, department heatmap
2. **Top Risks** - Filterable risk table with severity indicators
3. **User Investigation** - Deep-dive with risk gauge, findings, LLM analysis
4. **Privilege Graph** - Interactive NetworkX visualization
5. **False Positive Review** - Approve/Dismiss/Challenge workflow

---

## 🧠 Detection Rules

| # | Rule | Weight | Description |
|---|------|--------|-------------|
| 1 | Stale Privileged Account | 25 | Admin inactive > 30 days |
| 2 | Excessive Privileges | 20 | More access than role requires |
| 3 | Off-Hours Admin Activity | 15 | Admin operations at night |
| 4 | Cross-Department Access | 15 | Accessing other dept resources |
| 5 | Privilege Escalation | 25 | Non-admin performing admin ops |
| 6 | Service Account Misuse | 20 | Anomalous service account behavior |
| 7 | Bulk Data Export | 30 | Potential data exfiltration |

---

## 🤖 ML Model

- **Algorithm**: Isolation Forest
- **Contamination**: 15%
- **Features**: 13 behavioral indicators
- **Output**: Anomaly score (0-100) + binary anomaly label

---

## 🧪 Running Tests

```bash
cd sentineliq
pytest tests/ -v
```

---

## 🐳 Docker Support

```bash
docker build -t sentineliq .
docker run -p 8000:8000 -p 8501:8501 sentineliq
```

---

## 📈 Evaluation Targets

| Metric | Target |
|--------|--------|
| Precision | > 85% |
| Recall | > 80% |
| F1 Score | > 0.75 |
| ROC-AUC | > 0.80 |

---

## 🔑 LLM Configuration (Optional)

Set environment variable for LLM-powered analysis:
```bash
export OPENAI_API_KEY=your-key-here
```

Without an API key, the system uses intelligent rule-based fallback explanations.

---

## 📜 Compliance Mapping

- **NIST SP 800-53 AC-2**: Account Management
- **NIST SP 800-53 AC-6**: Least Privilege
- **GDPR Article 32**: Security of Processing
- **SOX Section 404**: Access Controls

---

## 👥 Built for Hackathon

Target Score: **85+ Points**

| Category | Points | Our Approach |
|----------|--------|--------------|
| Detection Quality | 30 | 7 rules + ML + context |
| Explainability | 25 | LLM + compliance mapping |
| Actionability | 20 | Recommendations + SLAs |
| Code Quality | 15 | Modular + typed + tested |
| Bonus Features | 10 | Graph + feedback + blast radius |
