"""
SentinelIQ FastAPI Backend.
Provides REST API endpoints for identity risk analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

from pipeline import SentinelIQPipeline

# Global pipeline instance
pipeline: Optional[SentinelIQPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    pipeline = SentinelIQPipeline(data_dir="data", output_dir="outputs")
    yield
    # Cleanup (if needed) on shutdown


app = FastAPI(
    title="SentinelIQ API",
    description="Identity Security Analytics Platform - REST API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class AnalyzeRequest(BaseModel):
    data_dir: str = "data"
    output_dir: str = "outputs"


class AnalyzeResponse(BaseModel):
    status: str
    total_users: int
    total_events: int
    total_findings: int
    risk_distribution: Dict[str, int]
    data_quality: float


class UserRiskResponse(BaseModel):
    user_id: str
    username: str
    department: str
    risk_score: float
    risk_level: str
    confidence: float


class InvestigationResponse(BaseModel):
    report: Dict


class SummaryResponse(BaseModel):
    overview: Dict
    risk_distribution: Dict
    department_risk: Dict
    top_risks: List[Dict]


class FeedbackRequest(BaseModel):
    user_id: str
    finding_id: Optional[str] = None
    action: str  # "approve", "dismiss", "challenge"
    reason: Optional[str] = None


# --- In-memory feedback store ---
feedback_store: List[Dict] = []


# --- API Endpoints ---

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Run full analysis pipeline.
    
    This triggers the complete SentinelIQ analysis:
    - Data ingestion & validation
    - Feature engineering
    - Rule engine evaluation
    - ML anomaly detection
    - Risk scoring
    - Report generation
    """
    global pipeline
    try:
        pipeline = SentinelIQPipeline(
            data_dir=request.data_dir,
            output_dir=request.output_dir
        )
        results = pipeline.run_full_pipeline()

        return AnalyzeResponse(
            status="success",
            total_users=results["total_users"],
            total_events=results["total_events"],
            total_findings=results["total_findings"],
            risk_distribution=results["risk_distribution"],
            data_quality=results["data_quality"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", response_model=SummaryResponse)
async def get_summary():
    """Get executive dashboard summary."""
    if pipeline is None or pipeline.scores_df is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run. POST /analyze first.")

    summary = pipeline.report_generator.generate_executive_summary(
        pipeline.scores_df, pipeline.rule_findings
    )

    return SummaryResponse(
        overview=summary["overview"],
        risk_distribution=summary["risk_distribution"],
        department_risk=summary["department_risk"],
        top_risks=summary["top_risks"],
    )


@app.get("/top-risks", response_model=List[UserRiskResponse])
async def get_top_risks(n: int = Query(default=10, le=50)):
    """Get top N highest-risk users."""
    if pipeline is None or pipeline.scores_df is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run. POST /analyze first.")

    top_risks = pipeline.get_top_risks(n)
    return [UserRiskResponse(**r) for r in top_risks]


@app.get("/report/{user_id}", response_model=InvestigationResponse)
async def get_user_report(user_id: str):
    """Get full investigation report for a specific user."""
    if pipeline is None or pipeline.scores_df is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run. POST /analyze first.")

    report = pipeline.get_user_investigation(user_id)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])

    return InvestigationResponse(report=report)


@app.get("/users")
async def list_users(
    department: Optional[str] = None,
    risk_level: Optional[str] = None,
    min_score: float = 0,
    max_score: float = 100,
    limit: int = 50,
):
    """List users with optional filters."""
    if pipeline is None or pipeline.scores_df is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run. POST /analyze first.")

    df = pipeline.scores_df.copy()

    if department:
        df = df[df["department"] == department]
    if risk_level:
        df = df[df["risk_level"] == risk_level]

    df = df[(df["final_risk_score"] >= min_score) & (df["final_risk_score"] <= max_score)]
    df = df.nlargest(limit, "final_risk_score")

    return df[["user_id", "username", "department", "job_title",
               "privilege_level", "final_risk_score", "risk_level", "confidence"]].to_dict("records")


@app.get("/findings")
async def get_findings(
    user_id: Optional[str] = None,
    rule: Optional[str] = None,
    severity: Optional[str] = None,
):
    """Get risk findings with optional filters."""
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run.")

    findings = pipeline.rule_findings

    if user_id:
        findings = [f for f in findings if f.get("user_id") == user_id]
    if rule:
        findings = [f for f in findings if f.get("rule") == rule]
    if severity:
        findings = [f for f in findings if f.get("severity") == severity]

    return findings


@app.get("/graph")
async def get_graph_data():
    """Get privilege graph data for visualization."""
    if pipeline is None or pipeline.graph is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run.")

    return pipeline.privilege_graph.export_for_visualization()


@app.get("/graph/blast-radius/{user_id}")
async def get_blast_radius(user_id: str):
    """Get blast radius analysis for a user."""
    if pipeline is None or pipeline.graph is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run.")

    result = pipeline.privilege_graph.get_blast_radius(user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/metrics")
async def get_metrics():
    """Get evaluation metrics."""
    if pipeline is None or pipeline.scores_df is None:
        raise HTTPException(status_code=400, detail="Analysis not yet run.")

    return pipeline.metrics_calculator.evaluate(pipeline.scores_df)


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit analyst feedback on a finding (False Positive review)."""
    feedback_store.append({
        "user_id": request.user_id,
        "finding_id": request.finding_id,
        "action": request.action,
        "reason": request.reason,
    })
    return {"status": "success", "message": f"Feedback recorded: {request.action}"}


@app.get("/feedback")
async def get_feedback():
    """Get all analyst feedback."""
    return feedback_store


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None and pipeline.scores_df is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
