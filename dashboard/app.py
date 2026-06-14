"""
SentinelIQ Streamlit Dashboard.
Interactive 5-page dashboard for identity risk analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

from pipeline import SentinelIQPipeline

# Page configuration
st.set_page_config(
    page_title="SentinelIQ - Identity Security Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid;
    }
    .critical { border-left-color: #FF4444; }
    .high { border-left-color: #FF8C00; }
    .medium { border-left-color: #FFD700; }
    .low { border-left-color: #00CC00; }
    .stMetricValue { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load and run the analysis pipeline (cached)."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    pipeline = SentinelIQPipeline(data_dir=data_dir, output_dir=output_dir)
    pipeline.run_full_pipeline()
    return pipeline


def main():
    """Main dashboard application."""
    # Sidebar
    st.sidebar.title("🛡️ SentinelIQ")
    st.sidebar.markdown("**Identity Security Analytics**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Executive Overview", "⚠️ Top Risks", "🔍 User Investigation",
         "🕸️ Privilege Graph", "✅ False Positive Review"]
    )

    # Load pipeline
    with st.spinner("Loading SentinelIQ Analysis Engine..."):
        try:
            pipeline = load_pipeline()
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")
            st.info("Make sure data files are in the 'data/' directory.")
            return

    scores_df = pipeline.scores_df
    findings = pipeline.rule_findings

    # Route to page
    if "Executive" in page:
        page_executive_overview(scores_df, findings, pipeline)
    elif "Top Risks" in page:
        page_top_risks(scores_df, findings)
    elif "Investigation" in page:
        page_user_investigation(scores_df, findings, pipeline)
    elif "Privilege Graph" in page:
        page_privilege_graph(pipeline)
    elif "False Positive" in page:
        page_false_positive_review(scores_df, findings)


def page_executive_overview(scores_df: pd.DataFrame, findings: list, pipeline):
    """Executive Overview Dashboard Page."""
    st.title("📊 Executive Overview")
    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Users", len(scores_df))
    with col2:
        critical = len(scores_df[scores_df["risk_level"] == "CRITICAL"])
        st.metric("🔴 Critical", critical)
    with col3:
        high = len(scores_df[scores_df["risk_level"] == "HIGH"])
        st.metric("🟠 High Risk", high)
    with col4:
        st.metric("Total Findings", len(findings))
    with col5:
        avg_score = scores_df["final_risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_score:.1f}")

    st.markdown("---")

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = scores_df["risk_level"].value_counts()
        colors = {"CRITICAL": "#FF4444", "HIGH": "#FF8C00", "MEDIUM": "#FFD700", "LOW": "#00CC00"}
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map=colors,
            hole=0.4,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Department Risk Heatmap")
        dept_risk = scores_df.groupby("department")["final_risk_score"].agg(["mean", "count"]).reset_index()
        dept_risk.columns = ["Department", "Avg Risk Score", "User Count"]
        fig = px.bar(
            dept_risk.sort_values("Avg Risk Score", ascending=False),
            x="Department", y="Avg Risk Score",
            color="Avg Risk Score",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            scores_df, x="final_risk_score",
            nbins=20, color_discrete_sequence=["#4C78A8"],
            labels={"final_risk_score": "Risk Score"},
        )
        fig.add_vline(x=60, line_dash="dash", line_color="red",
                      annotation_text="High Risk Threshold")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Findings by Rule Type")
        rule_counts = pd.DataFrame(findings).groupby("rule").size().reset_index(name="count")
        fig = px.bar(
            rule_counts.sort_values("count", ascending=True),
            x="count", y="rule", orientation="h",
            color="count", color_continuous_scale="Reds",
        )
        fig.update_layout(height=300, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Evaluation Metrics
    st.markdown("---")
    st.subheader("🎯 Detection Quality Metrics")
    metrics = pipeline.metrics_calculator.evaluate(scores_df)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    with mcol2:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    with mcol3:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    with mcol4:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")


def page_top_risks(scores_df: pd.DataFrame, findings: list):
    """Top Risks Page."""
    st.title("⚠️ Top Risk Users")
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_filter = st.selectbox(
            "Department", ["All"] + sorted(scores_df["department"].unique().tolist())
        )
    with col2:
        risk_filter = st.selectbox(
            "Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
        )
    with col3:
        n_results = st.slider("Show Top N", 5, 50, 20)

    # Apply filters
    filtered = scores_df.copy()
    if dept_filter != "All":
        filtered = filtered[filtered["department"] == dept_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]

    # Display table
    top_risks = filtered.nlargest(n_results, "final_risk_score")

    display_cols = ["user_id", "username", "department", "job_title",
                    "privilege_level", "final_risk_score", "risk_level", "confidence"]
    available_cols = [c for c in display_cols if c in top_risks.columns]

    st.dataframe(
        top_risks[available_cols].style.apply(
            lambda x: ['background-color: #FF444433' if v == "CRITICAL"
                       else 'background-color: #FF8C0033' if v == "HIGH"
                       else 'background-color: #FFD70033' if v == "MEDIUM"
                       else '' for v in x],
            subset=["risk_level"] if "risk_level" in available_cols else []
        ),
        use_container_width=True,
        height=600,
    )

    # Summary stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered Users", len(filtered))
    with col2:
        st.metric("Avg Score", f"{filtered['final_risk_score'].mean():.1f}")
    with col3:
        st.metric("Max Score", f"{filtered['final_risk_score'].max():.1f}")


def page_user_investigation(scores_df: pd.DataFrame, findings: list, pipeline):
    """User Investigation Page."""
    st.title("🔍 User Investigation")
    st.markdown("---")

    # User selector
    user_options = scores_df.sort_values("final_risk_score", ascending=False)
    user_display = [
        f"{row['username']} ({row['user_id']}) - Score: {row['final_risk_score']:.0f}"
        for _, row in user_options.head(50).iterrows()
    ]

    selected = st.selectbox("Select User to Investigate", user_display)

    if selected:
        user_id = selected.split("(")[1].split(")")[0]

        # Get investigation report
        with st.spinner("Generating investigation report..."):
            report = pipeline.get_user_investigation(user_id)

        if "error" in report:
            st.error(report["error"])
            return

        # User Profile
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("👤 User Profile")
            profile = report.get("user_profile", {})
            st.write(f"**User ID:** {profile.get('user_id', '')}")
            st.write(f"**Username:** {profile.get('username', '')}")
            st.write(f"**Department:** {profile.get('department', '')}")
            st.write(f"**Job Title:** {profile.get('job_title', '')}")
            st.write(f"**Privilege Level:** {profile.get('privilege_level', '')}")

        with col2:
            st.subheader("📊 Risk Assessment")
            risk = report.get("risk_assessment", {})
            score = risk.get("final_risk_score", 0)
            level = risk.get("risk_level", "LOW")

            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "#00CC00"},
                        {'range': [30, 60], 'color': "#FFD700"},
                        {'range': [60, 80], 'color': "#FF8C00"},
                        {'range': [80, 100], 'color': "#FF4444"},
                    ],
                },
                title={'text': f"Risk Level: {level}"},
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        # Findings
        st.markdown("---")
        st.subheader("🚨 Detected Findings")
        user_findings = report.get("findings", [])
        if user_findings:
            for finding in user_findings:
                severity = finding.get("severity", "MEDIUM")
                icon = "🔴" if severity == "CRITICAL" else "🟠" if severity == "HIGH" else "🟡"
                with st.expander(f"{icon} {finding.get('rule', '')} ({severity})"):
                    st.write(finding.get("description", ""))
                    st.write(f"**Score:** {finding.get('score', 0)}")
                    st.json(finding.get("evidence", {}))
        else:
            st.info("No specific rule findings for this user.")

        # LLM Analysis
        st.markdown("---")
        st.subheader("🤖 AI Investigation Report")
        llm = report.get("llm_analysis", {})
        if llm:
            st.write(f"**Executive Summary:** {llm.get('executive_summary', 'N/A')}")
            st.write(f"**Why Risky:** {llm.get('why_risky', 'N/A')}")
            st.write(f"**Business Impact:** {llm.get('business_impact', 'N/A')}")
            st.write(f"**Escalation:** {llm.get('escalation_level', 'N/A')}")

            if llm.get("recommended_actions"):
                st.write("**Recommended Actions:**")
                for action in llm["recommended_actions"]:
                    st.write(f"  • {action}")

        # Blast Radius
        st.markdown("---")
        st.subheader("💥 Blast Radius Analysis")
        blast = report.get("blast_radius", {})
        if blast and "error" not in blast:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Blast Score", blast.get("blast_radius_score", 0))
            with col2:
                st.metric("Systems at Risk", len(blast.get("systems_at_risk", [])))
            with col3:
                st.metric("Risk Level", blast.get("risk_assessment", "N/A"))

            if blast.get("systems_at_risk"):
                st.write("**Systems at Risk:**", ", ".join(blast["systems_at_risk"]))


def page_privilege_graph(pipeline):
    """Privilege Graph Visualization Page."""
    st.title("🕸️ Privilege Graph")
    st.markdown("---")

    graph_data = pipeline.privilege_graph.export_for_visualization()
    stats = pipeline.privilege_graph.get_graph_stats()

    # Graph stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", stats["total_nodes"])
    with col2:
        st.metric("Total Edges", stats["total_edges"])
    with col3:
        st.metric("User Nodes", stats["user_nodes"])
    with col4:
        st.metric("System Nodes", stats["system_nodes"])

    st.markdown("---")

    # Department filter for graph
    dept_filter = st.selectbox(
        "Filter by Department",
        ["All"] + sorted(pipeline.scores_df["department"].unique().tolist())
    )

    # Create network visualization using Plotly
    import networkx as nx

    G = pipeline.privilege_graph.graph

    # Filter if department selected
    if dept_filter != "All":
        dept_users = set(
            pipeline.scores_df[pipeline.scores_df["department"] == dept_filter]["user_id"]
        )
        subgraph_nodes = set()
        for user in dept_users:
            if user in G:
                subgraph_nodes.add(user)
                subgraph_nodes.update(G.successors(user))
        G_vis = G.subgraph(subgraph_nodes)
    else:
        # For "All", show top risk users + their connections
        top_users = pipeline.scores_df.nlargest(20, "final_risk_score")["user_id"].tolist()
        subgraph_nodes = set()
        for user in top_users:
            if user in G:
                subgraph_nodes.add(user)
                subgraph_nodes.update(G.successors(user))
        G_vis = G.subgraph(subgraph_nodes) if subgraph_nodes else G

    # Layout
    pos = nx.spring_layout(G_vis, seed=42, k=2)

    # Create edge traces
    edge_x, edge_y = [], []
    for edge in G_vis.edges():
        x0, y0 = pos.get(edge[0], (0, 0))
        x1, y1 = pos.get(edge[1], (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none'
    )

    # Create node traces by type
    node_traces = []
    for node_type, color, symbol in [
        ("user", "#4C78A8", "circle"),
        ("system", "#F58518", "diamond"),
        ("resource", "#54A24B", "square"),
        ("department", "#E45756", "triangle-up"),
    ]:
        nodes = [n for n, d in G_vis.nodes(data=True) if d.get("node_type") == node_type]
        if not nodes:
            continue

        node_x = [pos.get(n, (0, 0))[0] for n in nodes]
        node_y = [pos.get(n, (0, 0))[1] for n in nodes]
        labels = [G_vis.nodes[n].get("label", n) for n in nodes]

        node_traces.append(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            marker=dict(size=12 if node_type == "user" else 10, color=color, symbol=symbol),
            text=labels, textposition="top center",
            textfont=dict(size=8),
            name=node_type.capitalize(),
            hoverinfo='text',
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title="Identity-Privilege Relationship Graph"
    )
    st.plotly_chart(fig, use_container_width=True)


def page_false_positive_review(scores_df: pd.DataFrame, findings: list):
    """False Positive Review Page."""
    st.title("✅ False Positive Review")
    st.markdown("---")
    st.markdown("Review flagged findings and provide analyst feedback.")

    # Initialize feedback in session state
    if "feedback" not in st.session_state:
        st.session_state.feedback = []

    # Show findings for review
    high_findings = [f for f in findings if f.get("severity") in ["CRITICAL", "HIGH"]]

    if not high_findings:
        st.info("No high-severity findings to review.")
        return

    for i, finding in enumerate(high_findings[:20]):
        severity = finding.get("severity", "MEDIUM")
        icon = "🔴" if severity == "CRITICAL" else "🟠"

        with st.expander(
            f"{icon} {finding.get('username', 'Unknown')} - {finding.get('rule', '')} "
            f"(Score: {finding.get('score', 0)})"
        ):
            st.write(f"**Description:** {finding.get('description', '')}")
            st.write(f"**Recommendation:** {finding.get('recommendation', '')}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("✅ Approve", key=f"approve_{i}"):
                    st.session_state.feedback.append({
                        "user_id": finding.get("user_id"),
                        "rule": finding.get("rule"),
                        "action": "approved",
                    })
                    st.success("Finding approved")
            with col2:
                if st.button("❌ Dismiss", key=f"dismiss_{i}"):
                    st.session_state.feedback.append({
                        "user_id": finding.get("user_id"),
                        "rule": finding.get("rule"),
                        "action": "dismissed",
                    })
                    st.warning("Finding dismissed as false positive")
            with col3:
                if st.button("🔄 Challenge", key=f"challenge_{i}"):
                    st.session_state.feedback.append({
                        "user_id": finding.get("user_id"),
                        "rule": finding.get("rule"),
                        "action": "challenged",
                    })
                    st.info("Finding challenged for re-review")
            with col4:
                if st.button("📝 Note", key=f"note_{i}"):
                    st.text_input("Add note:", key=f"note_input_{i}")

    # Feedback Summary
    st.markdown("---")
    st.subheader("📋 Feedback Summary")
    if st.session_state.feedback:
        feedback_df = pd.DataFrame(st.session_state.feedback)
        st.dataframe(feedback_df, use_container_width=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Approved", len(feedback_df[feedback_df["action"] == "approved"]))
        with col2:
            st.metric("Dismissed", len(feedback_df[feedback_df["action"] == "dismissed"]))
        with col3:
            st.metric("Challenged", len(feedback_df[feedback_df["action"] == "challenged"]))
    else:
        st.info("No feedback submitted yet.")


if __name__ == "__main__":
    main()
