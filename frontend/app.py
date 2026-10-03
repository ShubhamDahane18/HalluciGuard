"""
HalluciGuard — Streamlit Dashboard
Communicates with FastAPI backend via REST.
"""
from __future__ import annotations
import os
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List

# ─── Config ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="HalluciGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0.25rem 0 0 0;
    }
    .badge-supported {
        background: #166534;
        color: #86efac;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-uncertain {
        background: #713f12;
        color: #fde68a;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-unsupported {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .claim-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .claim-card:hover { border-color: #64748b; }
    .answer-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-top: 0.5rem;
    }
    .evidence-box {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #93c5fd;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def api_get(path: str) -> Dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict, timeout: int = 300) -> Dict | None:
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to backend. "
            "Start it with: `uvicorn backend.main:app --reload --port 8000`"
        )
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The LLM may still be loading — try again.")
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"❌ Backend error: {detail}")
    return None


def status_badge(status: str) -> str:
    badges = {
        "supported": '<span class="badge-supported">✅ Supported</span>',
        "uncertain": '<span class="badge-uncertain">⚠️ Uncertain</span>',
        "unsupported": '<span class="badge-unsupported">❌ Unsupported</span>',
    }
    return badges.get(status, status)


def confidence_bar(confidence: float, status: str) -> str:
    pct = int(confidence * 100)
    colors = {"supported": "#22c55e", "uncertain": "#f59e0b", "unsupported": "#ef4444"}
    color = colors.get(status, "#94a3b8")
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    return f"`{bar}` **{pct}%**"


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ HalluciGuard")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")

    verification_method = st.selectbox(
        "Verification Method",
        ["adaptive", "embedding", "nli", "llm"],
        index=0,
        help="adaptive: cascading strategy (proposed). Others are baselines.",
    )

    st.markdown("---")

    # Health check
    st.markdown("### 🔌 Backend Status")
    health = api_get("/health")
    if health:
        st.success("Backend: Online")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Docs indexed", health.get("documents_indexed", 0))
        with col2:
            ollama_ok = health.get("ollama_available", False)
            st.metric("Ollama", "✅" if ollama_ok else "❌")
        if not ollama_ok:
            st.warning("⚠️ Ollama not detected. Start with `ollama serve`.")
    else:
        st.error("Backend: Offline")
        st.info("Run: `uvicorn backend.main:app --reload --port 8000`")

    st.markdown("---")
    st.markdown("### 📖 Methods")
    st.markdown(
        """
**Adaptive** (proposed)  
Cascades: Embedding → NLI → LLM

**Embedding**  
Cosine similarity baseline

**NLI**  
DeBERTa entailment baseline

**LLM**  
LLM-as-judge baseline
"""
    )


# ─── Main Header ─────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="main-header">
    <h1>🛡️ HalluciGuard</h1>
    <p>Claim-Level LLM Hallucination Detection & Evaluation Framework</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Input ──────────────────────────────────────────────────────────────────

question = st.text_area(
    "Enter your question",
    placeholder="Who founded Tesla? / What happened during Apollo 11?",
    height=90,
    key="question_input",
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🔍 Generate & Analyze", use_container_width=True)
with col_info:
    st.info(
        f"Method: **{verification_method.upper()}** | "
        "The system will generate an answer, extract claims, retrieve evidence, and verify each claim."
    )

# ─── Analysis ────────────────────────────────────────────────────────────────

if analyze_btn:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Generating answer and analyzing claims… this may take a minute"):
        payload = {"question": question.strip(), "verification_method": verification_method}
        response = api_post("/analyze", payload)

    if not response:
        st.stop()

    # Cache result in session state for persistence
    st.session_state["last_response"] = response
    st.session_state["last_question"] = question.strip()

# Render results from session state
if "last_response" in st.session_state:
    response = st.session_state["last_response"]
    summary = response["summary"]
    claims = response["claims"]

    # ─── Answer ────────────────────────────────────────────────────────
    st.markdown("### 📝 Generated Answer")
    st.markdown(
        f'<div class="answer-box">{response["answer"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Model: `{response['model']}`")

    st.markdown("---")

    # ─── Metrics ────────────────────────────────────────────────────────
    st.markdown("### 📊 Hallucination Report")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Total Claims", summary["total_claims"])
    with m2:
        st.metric("✅ Supported", summary["supported"], delta=f"{summary['supported_pct']}%")
    with m3:
        st.metric("⚠️ Uncertain", summary["uncertain"], delta=f"{summary['uncertain_pct']}%")
    with m4:
        st.metric("❌ Unsupported", summary["unsupported"], delta=f"{summary['unsupported_pct']}%" if summary["unsupported"] > 0 else None, delta_color="inverse")
    with m5:
        st.metric("Faithfulness", f"{summary['faithfulness_score']:.0%}")
    with m6:
        st.metric("Hallucination Rate", f"{summary['hallucination_rate']:.0%}")

    # Efficiency metrics (for adaptive mode)
    if summary.get("llm_calls", 0) > 0 or summary.get("nli_calls", 0) > 0:
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.metric("Embedding Calls", summary.get("embedding_calls", 0))
        with ec2:
            st.metric("NLI Calls", summary.get("nli_calls", 0))
        with ec3:
            st.metric("LLM Calls", summary.get("llm_calls", 0))
        with ec4:
            total = summary["total_claims"]
            llm = summary.get("llm_calls", 0)
            avoidance = (1 - llm / total) * 100 if total > 0 else 100
            st.metric("LLM Avoidance", f"{avoidance:.0f}%")

    st.markdown("---")

    # ─── Visualizations ──────────────────────────────────────────────────
    col_chart1, col_chart2, col_chart3 = st.columns(3)

    with col_chart1:
        # Status distribution donut
        labels = ["Supported", "Uncertain", "Unsupported"]
        values = [summary["supported"], summary["uncertain"], summary["unsupported"]]
        colors = ["#22c55e", "#f59e0b", "#ef4444"]
        fig_donut = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors),
                textfont=dict(color="white"),
            )
        )
        fig_donut.update_layout(
            title="Claim Status Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            legend=dict(font=dict(color="white")),
            showlegend=True,
            height=280,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        # Confidence histogram
        confs = [c["confidence"] for c in claims]
        fig_hist = px.histogram(
            x=confs,
            nbins=10,
            labels={"x": "Confidence"},
            title="Confidence Distribution",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font=dict(color="white"),
            height=280,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart3:
        # Verification methods used (adaptive)
        methods_used: Dict[str, int] = {}
        for c in claims:
            m = c["verification_method"]
            methods_used[m] = methods_used.get(m, 0) + 1

        fig_bar = px.bar(
            x=list(methods_used.keys()),
            y=list(methods_used.values()),
            labels={"x": "Method", "y": "Claims"},
            title="Verification Methods Used",
            color_discrete_sequence=["#6366f1"],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font=dict(color="white"),
            height=280,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ─── Claim Table ─────────────────────────────────────────────────────
    st.markdown("### 🔬 Claim-Level Verification")

    table_data = []
    for c in claims:
        table_data.append({
            "ID": c["claim_id"],
            "Claim": c["claim"],
            "Status": c["status"].upper(),
            "Confidence": f"{c['confidence']:.0%}",
            "Method": c["verification_method"],
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Claim Detail Cards ───────────────────────────────────────────────
    st.markdown("### 📋 Claim Details")

    # Group by status for better UX
    unsupported_claims = [c for c in claims if c["status"] == "unsupported"]
    uncertain_claims = [c for c in claims if c["status"] == "uncertain"]
    supported_claims = [c for c in claims if c["status"] == "supported"]

    ordered_claims = unsupported_claims + uncertain_claims + supported_claims

    for claim in ordered_claims:
        status = claim["status"]
        conf = claim["confidence"]
        icons = {"supported": "✅", "uncertain": "⚠️", "unsupported": "❌"}
        icon = icons.get(status, "")

        with st.expander(
            f"{icon} Claim {claim['claim_id']}: {claim['claim'][:80]}{'…' if len(claim['claim']) > 80 else ''}",
            expanded=(status != "supported"),
        ):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**Claim:** {claim['claim']}")
                st.markdown(
                    f"**Status:** {status_badge(status)}", unsafe_allow_html=True
                )
                st.markdown(f"**Confidence:** {confidence_bar(conf, status)}")
                st.markdown(f"**Method:** `{claim['verification_method']}`")

                if claim.get("embedding_score") is not None:
                    st.markdown(f"**Embedding Score:** `{claim['embedding_score']:.3f}`")

            with col_b:
                # NLI probabilities
                if claim.get("nli"):
                    nli = claim["nli"]
                    probs = nli.get("probabilities", {})
                    if probs:
                        fig_nli = go.Figure(
                            go.Bar(
                                x=list(probs.keys()),
                                y=list(probs.values()),
                                marker_color=["#22c55e", "#f59e0b", "#ef4444"],
                            )
                        )
                        fig_nli.update_layout(
                            title="NLI Probs",
                            height=180,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white", size=10),
                        )
                        st.plotly_chart(fig_nli, use_container_width=True)

            # LLM Judge
            if claim.get("llm_judge") and claim["llm_judge"].get("used"):
                llm = claim["llm_judge"]
                st.markdown("---")
                st.markdown("**🤖 LLM Judge Verdict**")
                col_lbl, col_conf = st.columns(2)
                with col_lbl:
                    st.markdown(
                        f"{status_badge(llm['label'])}",
                        unsafe_allow_html=True,
                    )
                with col_conf:
                    st.markdown(f"Confidence: `{llm['confidence']:.0%}`")
                if llm.get("reason"):
                    st.markdown(
                        f"> *{llm['reason']}*",
                    )

            # Evidence
            evidence = claim.get("evidence", [])
            if evidence:
                st.markdown("---")
                st.markdown(f"**📚 Retrieved Evidence** ({len(evidence)} chunks)")
                for i, ev in enumerate(evidence[:3]):
                    source = ev.get("metadata", {}).get("source", "unknown")
                    score = ev.get("score", 0)
                    st.markdown(
                        f'<div class="evidence-box">'
                        f'<b>Source:</b> {source} | <b>Score:</b> {score:.3f}<br><br>'
                        f'{ev["text"][:300]}{"…" if len(ev["text"]) > 300 else ""}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

else:
    # Placeholder on first load
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1:** Enter a question and click Analyze")
    with col2:
        st.info("**Step 2:** HalluciGuard extracts and verifies each claim")
    with col3:
        st.info("**Step 3:** Review per-claim hallucination analysis")
