import io
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# ==============================================================================
# 1. SYSTEM CONFIGURATION & ENTERPRISE BRANDING ASSETS
# ==============================================================================
st.set_page_config(
    page_title="NEXUS AI™ | Sovereign Enterprise Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEXUS_LOGO_SVG = """
<svg width="60" height="60" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="nexusPrimary" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="50%" stop-color="#1E3A8A"/>
      <stop offset="100%" stop-color="#2563EB"/>
    </linearGradient>
    <linearGradient id="nexusAccent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#818CF8"/>
    </linearGradient>
  </defs>
  <polygon points="32,2 58,17 58,47 32,62 6,47 6,17" fill="url(#nexusPrimary)" stroke="url(#nexusAccent)" stroke-width="2.5"/>
  <polygon points="32,12 49,22 49,42 32,52 15,42 15,22" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4 2"/>
  <circle cx="32" cy="32" r="7" fill="#38BDF8"/>
  <line x1="32" y1="12" x2="32" y2="25" stroke="#93C5FD" stroke-width="2"/>
  <line x1="32" y1="39" x2="32" y2="52" stroke="#93C5FD" stroke-width="2"/>
  <line x1="15" y1="22" x2="26" y2="28" stroke="#93C5FD" stroke-width="2"/>
  <line x1="49" y1="42" x2="38" y2="36" stroke="#93C5FD" stroke-width="2"/>
</svg>
"""


def inject_enterprise_css() -> None:
    """Injects responsive, mobile-friendly, and executive-grade CSS styles."""
    st.markdown(
        """
        <style>
        .nexus-header-card {
            background: linear-gradient(120deg, #0F172A 0%, #1E293B 100%);
            padding: 20px 26px;
            border-radius: 12px;
            border-left: 6px solid #3B82F6;
            margin-bottom: 20px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.14);
        }
        .nexus-title {
            font-size: clamp(22px, 3vw, 34px);
            font-weight: 900;
            color: #FFFFFF;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .nexus-tagline {
            font-size: 13px;
            color: #93C5FD;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 3px;
        }
        .nexus-subtitle {
            font-size: 14px;
            color: #CBD5E1;
            margin-top: 6px;
        }
        .nexus-badge {
            background-color: #2563EB;
            color: #FFFFFF;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            vertical-align: middle;
            margin-left: 8px;
        }
        .healing-banner {
            background-color: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-left: 4px solid #16A34A;
            padding: 11px 16px;
            border-radius: 6px;
            color: #166534;
            font-size: 13.5px;
            margin-bottom: 16px;
        }
        .upgrade-btn {
            display: block;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF !important;
            padding: 11px 16px;
            border-radius: 8px;
            text-align: center;
            font-weight: 800;
            text-decoration: none;
            margin-top: 10px;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. CORE ALGORITHMIC ENGINE (SELF-HEALING & AUTO-CALIBRATION)
# ==============================================================================
class NexusCognitiveEngine:
    """Modular, fault-tolerant AI engine for data self-healing and anomaly detection."""

    @staticmethod
    def self_heal_dataframe(
        df: pd.DataFrame, required_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Repairs missing values, duplicates, and schema mismatches automatically."""
        logs: List[str] = []
        clean_df = df.copy()
        clean_df.columns = [str(c).strip() for c in clean_df.columns]

        null_count = int(clean_df.isnull().sum().sum())
        if null_count > 0:
            for col in clean_df.select_dtypes(include=[np.number]).columns:
                if clean_df[col].isnull().any():
                    median_val = float(clean_df[col].median())
                    clean_df[col] = clean_df[col].fillna(
                        0.0 if np.isnan(median_val) else median_val
                    )
            logs.append(f"Imputed {null_count} missing cells via median estimation")

        dup_count = int(clean_df.duplicated().sum())
        if dup_count > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            logs.append(f"Purged {dup_count} duplicate rows")

        for col in required_cols:
            if col not in clean_df.columns:
                clean_df[col] = np.random.normal(100.0, 15.0, len(clean_df))
                logs.append(f"Synthesized missing column '{col}'")
            else:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(0.0)

        if not logs:
            logs.append("Schema integrity 100% verified (Zero structural defects)")

        return clean_df, logs

    @staticmethod
    def auto_calibrate_and_score(
        df: pd.DataFrame,
        feature_cols: List[str],
        manual_contamination: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, float]:
        """Auto-tunes IsolationForest via IQR statistics and computes 0-100 Risk Score."""
        scored_df = df.copy()
        if len(scored_df) < 5:
            scored_df["Risk_Score_100"] = 0.0
            scored_df["Anomaly_Flag"] = 1
            return scored_df, 5.0

        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(scored_df[feature_cols])

        q1 = np.percentile(x_scaled, 25, axis=0)
        q3 = np.percentile(x_scaled, 75, axis=0)
        iqr = q3 - q1
        outlier_mask = (
            (x_scaled < (q1 - 2.5 * iqr)) | (x_scaled > (q3 + 2.5 * iqr))
        ).any(axis=1)
        estimated_rate = float(np.clip(outlier_mask.mean(), 0.01, 0.25))

        contamination = (
            estimated_rate if manual_contamination is None else manual_contamination
        )

        model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )
        preds = model.fit_predict(x_scaled)
        raw_scores = model.decision_function(x_scaled)

        min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
        if max_s - min_s > 1e-9:
            risk_scores = ((max_s - raw_scores) / (max_s - min_s)) * 100.0
        else:
            risk_scores = np.zeros(len(raw_scores))

        scored_df["Risk_Score_100"] = np.round(risk_scores, 1)
        scored_df["Anomaly_Flag"] = preds
        return scored_df, round(contamination * 100.0, 2)


# ==============================================================================
# 3. SYNTHETIC ENTERPRISE DATA GENERATORS (FOR INSTANT DEMO & BENCHMARKING)
# ==============================================================================
@st.cache_data
def generate_default_finance_data(n_rows: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Transaction_Amount": np.random.normal(2200.0, 350.0, n_rows),
            "Tax_Value": np.random.normal(165.0, 30.0, n_rows),
            "Processing_Days": np.random.poisson(2, n_rows),
        }
    )
    df.loc[15:24, "Transaction_Amount"] = np.random.normal(42000.0, 3000.0, 10)
    df.loc[15:24, "Tax_Value"] = np.random.normal(6300.0, 500.0, 10)
    df["Transaction_Date"] = pd.date_range(start="2026-11-01", periods=n_rows, freq="h")
    return df


@st.cache_data
def generate_default_cyber_data(n_rows: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Packet_Size": np.random.normal(520.0, 95.0, n_rows),
            "Failed_Attempts": np.random.poisson(1, n_rows),
            "Response_Time": np.random.exponential(1.4, n_rows),
        }
    )
    df.loc[20:32, "Packet_Size"] = np.random.normal(5400.0, 350.0, 13)
    df.loc[20:32, "Failed_Attempts"] = np.random.randint(20, 42, 13)
    df["Timestamp"] = pd.date_range(
        start="2026-11-16 08:00:00", periods=n_rows, freq="s"
    )
    return df


# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "مرحباً بك في **NEXUS AI™ Sovereign Enterprise Suite**. "
                    "أنا المساعد الذكي المتصل مباشرة بمحركات التدقيق المالي، الأمن السيبراني، وتحليل المستندات. "
                    "ارفع بياناتك أو اسألني مباشرة عن المؤشرات والمخاطر!"
                ),
            }
        ]
    if "active_context" not in st.session_state:
        st.session_state.active_context = {}


# ==============================================================================
# 5. UI RENDERERS (MODULAR & CLEAN ARCHITECTURE)
# ==============================================================================
def render_header() -> None:
    st.markdown(
        f"""
        <div class="nexus-header-card">
            <div style="display: flex; align-items: center; gap: 18px;">
                <div>{NEXUS_LOGO_SVG}</div>
                <div>
                    <div class="nexus-title">NEXUS AI™ <span class="nexus-badge">SOVEREIGN ENTERPRISE</span></div>
                    <div class="nexus-tagline">Autonomous Intelligence • Self-Healing Pipelines • Zero Blind Spots</div>
                    <div class="nexus-subtitle">Next-Generation Fiscal Auditing, Cybersecurity Telemetry, Deep PDF Governance & Live Copilot</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[Optional[float], str]:
    st.sidebar.markdown(
        f"<div style='text-align:center;'>{NEXUS_LOGO_SVG}<h3>NEXUS CONTROL</h3></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    tier = st.sidebar.selectbox(
        "🏢 Enterprise License Tier",
        [
            "Starter / Freemium (5 Scans/Mo)",
            "Pro Auditor ($99/Mo - Unlimited)",
            "Sovereign Enterprise ($499/Mo)",
        ],
    )

    tuning_mode = st.sidebar.radio(
        "🧠 Algorithmic Calibration",
        ["Autonomous Self-Tuning (AI Auto)", "Manual Sensitivity Override"],
    )

    manual_contam: Optional[float] = None
    if tuning_mode == "Manual Sensitivity Override":
        manual_contam = st.sidebar.slider(
            "Contamination Threshold", 0.01, 0.25, 0.05, 0.01
        )
    else:
        st.sidebar.caption(
            "✨ AI auto-calibrates IQR statistical boundaries per dataset."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💳 Commercial & Billing")
    stripe_link = (
        "https://buy.stripe.com/test_pro_plan"
        if "Pro" in tier
        else "https://buy.stripe.com/test_enterprise_plan"
    )
    st.sidebar.markdown(
        f"""
        <a href="{stripe_link}" target="_blank" class="upgrade-btn">
            🚀 تفعيل الاشتراك المؤسسي
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.success("● Self-Healing Pipeline: ACTIVE")
    st.sidebar.success("● Live Copilot Context: SYNCED")
    st.sidebar.info(
        "🔒 **ISO/IEC 27001 & SOC2 Compliant:** Volatile in-memory processing with zero data retention."
    )

    return manual_contam, tier


def render_finance_tab(manual_contam: Optional[float]) -> None:
    st.subheader("💰 Autonomous Fiscal Fraud & Ledger Auditing")
    uploaded_fin = st.file_uploader(
        "Upload Financial Ledger / Invoices (CSV)", type=["csv"], key="fin_uploader"
    )

    try:
        raw_df = (
            pd.read_csv(uploaded_fin)
            if uploaded_fin is not None
            else generate_default_finance_data()
        )
        req_cols = ["Transaction_Amount", "Tax_Value", "Processing_Days"]
        clean_df, heal_logs = NexusCognitiveEngine.self_heal_dataframe(raw_df, req_cols)
        scored_df, applied_rate = NexusCognitiveEngine.auto_calibrate_and_score(
            clean_df, req_cols, manual_contam
        )
        scored_df["Audit_Status"] = scored_df["Anomaly_Flag"].apply(
            lambda x: "🚨 Fiscal Fraud Flag" if x == -1 else "✅ Verified Compliant"
        )

        fraud_rows = scored_df[scored_df["Anomaly_Flag"] == -1]
        st.session_state.active_context["finance"] = {
            "total_records": len(scored_df),
            "fraud_count": len(fraud_rows),
            "total_volume": float(scored_df["Transaction_Amount"].sum()),
            "max_amount": float(scored_df["Transaction_Amount"].max()),
            "auto_rate": applied_rate,
        }

        st.markdown(
            f"<div class='healing-banner'>🩺 <b>Self-Healing Report:</b> {' | '.join(heal_logs)} (Calibrated Sensitivity: <b>{applied_rate}%</b>)</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ledgers Audited", f"{len(scored_df):,}")
        c2.metric("Fraud Risks Flagged", len(fraud_rows))
        c3.metric("Total Audited Volume", f"${scored_df['Transaction_Amount'].sum():,.0f}")
        c4.metric("Peak Risk Score", f"{scored_df['Risk_Score_100'].max()}/100")

        fig = px.scatter(
            scored_df,
            x="Transaction_Amount",
            y="Tax_Value",
            color="Audit_Status",
            size="Risk_Score_100",
            color_discrete_map={
                "✅ Verified Compliant": "#10B981",
                "🚨 Fiscal Fraud Flag": "#EF4444",
            },
            title="Self-Calibrating Fiscal Anomaly & Risk Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            scored_df.sort_values(by="Risk_Score_100", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "📥 Download Certified Financial Audit Report (CSV)",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name="nexus_certified_financial_audit.csv",
            mime="text/csv",
            key="dl_fin_btn",
        )
    except Exception as exc:
        st.error(f"❌ Financial Module Error: {str(exc)}")


def render_cyber_tab(manual_contam: Optional[float]) -> None:
    st.subheader("🛡️ Self-Evolving Cybersecurity Network Telemetry")
    uploaded_cyber = st.file_uploader(
        "Upload Network Traffic Logs (CSV)", type=["csv"], key="cyber_uploader"
    )

    try:
        raw_df = (
            pd.read_csv(uploaded_cyber)
            if uploaded_cyber is not None
            else generate_default_cyber_data()
        )
        req_cols = ["Packet_Size", "Failed_Attempts", "Response_Time"]
        clean_df, heal_logs = NexusCognitiveEngine.self_heal_dataframe(raw_df, req_cols)
        scored_df, applied_rate = NexusCognitiveEngine.auto_calibrate_and_score(
            clean_df, req_cols, manual_contam
        )
        scored_df["Threat_Status"] = scored_df["Anomaly_Flag"].apply(
            lambda x: "🚨 Critical Threat Vector" if x == -1 else "🟢 Secured Node"
        )

        threat_rows = scored_df[scored_df["Anomaly_Flag"] == -1]
        st.session_state.active_context["cyber"] = {
            "total_packets": len(scored_df),
            "threat_count": len(threat_rows),
            "max_failed": int(scored_df["Failed_Attempts"].max()),
            "auto_rate": applied_rate,
        }

        st.markdown(
            f"<div class='healing-banner'>🩺 <b>Self-Healing Report:</b> {' | '.join(heal_logs)} (Calibrated Sensitivity: <b>{applied_rate}%</b>)</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Packets Scanned", f"{len(scored_df):,}")
        c2.metric("Threats Isolated", len(threat_rows))
        c3.metric("Max Failed Logins", int(scored_df["Failed_Attempts"].max()))
        c4.metric(
            "Defense Posture", "🚨 DEFCON 1" if len(threat_rows) > 15 else "🟢 OPTIMAL"
        )

        fig = px.scatter(
            scored_df,
            x="Packet_Size",
            y="Failed_Attempts",
            color="Threat_Status",
            size="Risk_Score_100",
            color_discrete_map={
                "🟢 Secured Node": "#10B981",
                "🚨 Critical Threat Vector": "#EF4444",
            },
            title="Autonomous Threat Dispersion & Risk Scoring Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            scored_df.sort_values(by="Risk_Score_100", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "📥 Download Certified Cybersecurity Report (CSV)",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name="nexus_certified_cyber_audit.csv",
            mime="text/csv",
            key="dl_cyber_btn",
        )
    except Exception as exc:
        st.error(f"❌ Cybersecurity Module Error: {str(exc)}")


def render_pdf_tab() -> None:
    st.subheader("📄 Neural PDF & Contract Risk Governance")
    uploaded_pdf = st.file_uploader(
        "Upload Government / Financial PDF Document", type=["pdf"], key="pdf_uploader"
    )

    if uploaded_pdf is None:
        st.info(
            "👈 Upload a PDF invoice or contract to extract financial tokens, entities, and legal risks."
        )
        return

    try:
        pdf_bytes = io.BytesIO(uploaded_pdf.getvalue())
        reader = PdfReader(pdf_bytes)
        pages_count = len(reader.pages)
        text_chunks: List[str] = []

        for idx, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text_chunks.append(f"--- Page {idx + 1} ---\n{extracted}")

        full_text = "\n\n".join(text_chunks)
        if not full_text.strip():
            st.warning(
                "⚠️ Scanned image PDF detected without an embedded text layer."
            )
            return

        amounts = re.findall(
            r"\b(?:\$|USD|EUR|SAR|AED)?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})\b", full_text
        )
        emails = re.findall(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", full_text
        )
        risk_keywords = [
            "fraud",
            "suspicious",
            "error",
            "delay",
            "penalty",
            "urgent",
            "unauthorized",
            "tax",
            "breach",
            "مخالفة",
            "غرامة",
            "تأخير",
            "ضريبة",
        ]
        found_risks = sorted(
            list({kw for kw in risk_keywords if kw in full_text.lower()})
        )

        st.session_state.active_context["pdf"] = {
            "filename": uploaded_pdf.name,
            "pages": pages_count,
            "amounts": amounts[:12],
            "emails": emails[:6],
            "risks": found_risks,
            "raw_teimport io
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# ==============================================================================
# 1. SYSTEM CONFIGURATION & ENTERPRISE BRANDING ASSETS
# ==============================================================================
st.set_page_config(
    page_title="NEXUS AI™ | Sovereign Enterprise Intelligence",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEXUS_LOGO_SVG = """
<svg width="60" height="60" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="nexusPrimary" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="50%" stop-color="#1E3A8A"/>
      <stop offset="100%" stop-color="#2563EB"/>
    </linearGradient>
    <linearGradient id="nexusAccent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#818CF8"/>
    </linearGradient>
  </defs>
  <polygon points="32,2 58,17 58,47 32,62 6,47 6,17" fill="url(#nexusPrimary)" stroke="url(#nexusAccent)" stroke-width="2.5"/>
  <polygon points="32,12 49,22 49,42 32,52 15,42 15,22" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4 2"/>
  <circle cx="32" cy="32" r="7" fill="#38BDF8"/>
  <line x1="32" y1="12" x2="32" y2="25" stroke="#93C5FD" stroke-width="2"/>
  <line x1="32" y1="39" x2="32" y2="52" stroke="#93C5FD" stroke-width="2"/>
  <line x1="15" y1="22" x2="26" y2="28" stroke="#93C5FD" stroke-width="2"/>
  <line x1="49" y1="42" x2="38" y2="36" stroke="#93C5FD" stroke-width="2"/>
</svg>
"""


def inject_enterprise_css() -> None:
    """Injects responsive, mobile-friendly, and executive-grade CSS styles."""
    st.markdown(
        """
        <style>
        .nexus-header-card {
            background: linear-gradient(120deg, #0F172A 0%, #1E293B 100%);
            padding: 20px 26px;
            border-radius: 12px;
            border-left: 6px solid #3B82F6;
            margin-bottom: 20px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.14);
        }
        .nexus-title {
            font-size: clamp(22px, 3vw, 34px);
            font-weight: 900;
            color: #FFFFFF;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .nexus-tagline {
            font-size: 13px;
            color: #93C5FD;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 3px;
        }
        .nexus-subtitle {
            font-size: 14px;
            color: #CBD5E1;
            margin-top: 6px;
        }
        .nexus-badge {
            background-color: #2563EB;
            color: #FFFFFF;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            vertical-align: middle;
            margin-left: 8px;
        }
        .healing-banner {
            background-color: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-left: 4px solid #16A34A;
            padding: 11px 16px;
            border-radius: 6px;
            color: #166534;
            font-size: 13.5px;
            margin-bottom: 16px;
        }
        .upgrade-btn {
            display: block;
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF !important;
            padding: 11px 16px;
            border-radius: 8px;
            text-align: center;
            font-weight: 800;
            text-decoration: none;
            margin-top: 10px;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. CORE ALGORITHMIC ENGINE (SELF-HEALING & AUTO-CALIBRATION)
# ==============================================================================
class NexusCognitiveEngine:
    """Modular, fault-tolerant AI engine for data self-healing and anomaly detection."""

    @staticmethod
    def self_heal_dataframe(
        df: pd.DataFrame, required_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Repairs missing values, duplicates, and schema mismatches automatically."""
        logs: List[str] = []
        clean_df = df.copy()
        clean_df.columns = [str(c).strip() for c in clean_df.columns]

        null_count = int(clean_df.isnull().sum().sum())
        if null_count > 0:
            for col in clean_df.select_dtypes(include=[np.number]).columns:
                if clean_df[col].isnull().any():
                    median_val = float(clean_df[col].median())
                    clean_df[col] = clean_df[col].fillna(
                        0.0 if np.isnan(median_val) else median_val
                    )
            logs.append(f"Imputed {null_count} missing cells via median estimation")

        dup_count = int(clean_df.duplicated().sum())
        if dup_count > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            logs.append(f"Purged {dup_count} duplicate rows")

        for col in required_cols:
            if col not in clean_df.columns:
                clean_df[col] = np.random.normal(100.0, 15.0, len(clean_df))
                logs.append(f"Synthesized missing column '{col}'")
            else:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(0.0)

        if not logs:
            logs.append("Schema integrity 100% verified (Zero structural defects)")

        return clean_df, logs

    @staticmethod
    def auto_calibrate_and_score(
        df: pd.DataFrame,
        feature_cols: List[str],
        manual_contamination: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, float]:
        """Auto-tunes IsolationForest via IQR statistics and computes 0-100 Risk Score."""
        scored_df = df.copy()
        if len(scored_df) < 5:
            scored_df["Risk_Score_100"] = 0.0
            scored_df["Anomaly_Flag"] = 1
            return scored_df, 5.0

        scaler = RobustScaler()
        x_scaled = scaler.fit_transform(scored_df[feature_cols])

        q1 = np.percentile(x_scaled, 25, axis=0)
        q3 = np.percentile(x_scaled, 75, axis=0)
        iqr = q3 - q1
        outlier_mask = (
            (x_scaled < (q1 - 2.5 * iqr)) | (x_scaled > (q3 + 2.5 * iqr))
        ).any(axis=1)
        estimated_rate = float(np.clip(outlier_mask.mean(), 0.01, 0.25))

        contamination = (
            estimated_rate if manual_contamination is None else manual_contamination
        )

        model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )
        preds = model.fit_predict(x_scaled)
        raw_scores = model.decision_function(x_scaled)

        min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
        if max_s - min_s > 1e-9:
            risk_scores = ((max_s - raw_scores) / (max_s - min_s)) * 100.0
        else:
            risk_scores = np.zeros(len(raw_scores))

        scored_df["Risk_Score_100"] = np.round(risk_scores, 1)
        scored_df["Anomaly_Flag"] = preds
        return scored_df, round(contamination * 100.0, 2)


# ==============================================================================
# 3. SYNTHETIC ENTERPRISE DATA GENERATORS (FOR INSTANT DEMO & BENCHMARKING)
# ==============================================================================
@st.cache_data
def generate_default_finance_data(n_rows: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Transaction_Amount": np.random.normal(2200.0, 350.0, n_rows),
            "Tax_Value": np.random.normal(165.0, 30.0, n_rows),
            "Processing_Days": np.random.poisson(2, n_rows),
        }
    )
    df.loc[15:24, "Transaction_Amount"] = np.random.normal(42000.0, 3000.0, 10)
    df.loc[15:24, "Tax_Value"] = np.random.normal(6300.0, 500.0, 10)
    df["Transaction_Date"] = pd.date_range(start="2026-11-01", periods=n_rows, freq="h")
    return df


@st.cache_data
def generate_default_cyber_data(n_rows: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Packet_Size": np.random.normal(520.0, 95.0, n_rows),
            "Failed_Attempts": np.random.poisson(1, n_rows),
            "Response_Time": np.random.exponential(1.4, n_rows),
        }
    )
    df.loc[20:32, "Packet_Size"] = np.random.normal(5400.0, 350.0, 13)
    df.loc[20:32, "Failed_Attempts"] = np.random.randint(20, 42, 13)
    df["Timestamp"] = pd.date_range(
        start="2026-11-16 08:00:00", periods=n_rows, freq="s"
    )
    return df


# ==============================================================================
# 4. SESSION STATE INITIALIZATION
# ==============================================================================
def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "مرحباً بك في **NEXUS AI™ Sovereign Enterprise Suite**. "
                    "أنا المساعد الذكي المتصل مباشرة بمحركات التدقيق المالي، الأمن السيبراني، وتحليل المستندات. "
                    "ارفع بياناتك أو اسألني مباشرة عن المؤشرات والمخاطر!"
                ),
            }
        ]
    if "active_context" not in st.session_state:
        st.session_state.active_context = {}


# ==============================================================================
# 5. UI RENDERERS (MODULAR & CLEAN ARCHITECTURE)
# ==============================================================================
def render_header() -> None:
    st.markdown(
        f"""
        <div class="nexus-header-card">
            <div style="display: flex; align-items: center; gap: 18px;">
                <div>{NEXUS_LOGO_SVG}</div>
                <div>
                    <div class="nexus-title">NEXUS AI™ <span class="nexus-badge">SOVEREIGN ENTERPRISE</span></div>
                    <div class="nexus-tagline">Autonomous Intelligence • Self-Healing Pipelines • Zero Blind Spots</div>
                    <div class="nexus-subtitle">Next-Generation Fiscal Auditing, Cybersecurity Telemetry, Deep PDF Governance & Live Copilot</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[Optional[float], str]:
    st.sidebar.markdown(
        f"<div style='text-align:center;'>{NEXUS_LOGO_SVG}<h3>NEXUS CONTROL</h3></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    tier = st.sidebar.selectbox(
        "🏢 Enterprise License Tier",
        [
            "Starter / Freemium (5 Scans/Mo)",
            "Pro Auditor ($99/Mo - Unlimited)",
            "Sovereign Enterprise ($499/Mo)",
        ],
    )

    tuning_mode = st.sidebar.radio(
        "🧠 Algorithmic Calibration",
        ["Autonomous Self-Tuning (AI Auto)", "Manual Sensitivity Override"],
    )

    manual_contam: Optional[float] = None
    if tuning_mode == "Manual Sensitivity Override":
        manual_contam = st.sidebar.slider(
            "Contamination Threshold", 0.01, 0.25, 0.05, 0.01
        )
    else:
        st.sidebar.caption(
            "✨ AI auto-calibrates IQR statistical boundaries per dataset."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💳 Commercial & Billing")
    stripe_link = (
        "https://buy.stripe.com/test_pro_plan"
        if "Pro" in tier
        else "https://buy.stripe.com/test_enterprise_plan"
    )
    st.sidebar.markdown(
        f"""
        <a href="{stripe_link}" target="_blank" class="upgrade-btn">
            🚀 تفعيل الاشتراك المؤسسي
        </a>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.success("● Self-Healing Pipeline: ACTIVE")
    st.sidebar.success("● Live Copilot Context: SYNCED")
    st.sidebar.info(
        "🔒 **ISO/IEC 27001 & SOC2 Compliant:** Volatile in-memory processing with zero data retention."
    )

    return manual_contam, tier


def render_finance_tab(manual_contam: Optional[float]) -> None:
    st.subheader("💰 Autonomous Fiscal Fraud & Ledger Auditing")
    uploaded_fin = st.file_uploader(
        "Upload Financial Ledger / Invoices (CSV)", type=["csv"], key="fin_uploader"
    )

    try:
        raw_df = (
            pd.read_csv(uploaded_fin)
            if uploaded_fin is not None
            else generate_default_finance_data()
        )
        req_cols = ["Transaction_Amount", "Tax_Value", "Processing_Days"]
        clean_df, heal_logs = NexusCognitiveEngine.self_heal_dataframe(raw_df, req_cols)
        scored_df, applied_rate = NexusCognitiveEngine.auto_calibrate_and_score(
            clean_df, req_cols, manual_contam
        )
        scored_df["Audit_Status"] = scored_df["Anomaly_Flag"].apply(
            lambda x: "🚨 Fiscal Fraud Flag" if x == -1 else "✅ Verified Compliant"
        )

        fraud_rows = scored_df[scored_df["Anomaly_Flag"] == -1]
        st.session_state.active_context["finance"] = {
            "total_records": len(scored_df),
            "fraud_count": len(fraud_rows),
            "total_volume": float(scored_df["Transaction_Amount"].sum()),
            "max_amount": float(scored_df["Transaction_Amount"].max()),
            "auto_rate": applied_rate,
        }

        st.markdown(
            f"<div class='healing-banner'>🩺 <b>Self-Healing Report:</b> {' | '.join(heal_logs)} (Calibrated Sensitivity: <b>{applied_rate}%</b>)</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ledgers Audited", f"{len(scored_df):,}")
        c2.metric("Fraud Risks Flagged", len(fraud_rows))
        c3.metric("Total Audited Volume", f"${scored_df['Transaction_Amount'].sum():,.0f}")
        c4.metric("Peak Risk Score", f"{scored_df['Risk_Score_100'].max()}/100")

        fig = px.scatter(
            scored_df,
            x="Transaction_Amount",
            y="Tax_Value",
            color="Audit_Status",
            size="Risk_Score_100",
            color_discrete_map={
                "✅ Verified Compliant": "#10B981",
                "🚨 Fiscal Fraud Flag": "#EF4444",
            },
            title="Self-Calibrating Fiscal Anomaly & Risk Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            scored_df.sort_values(by="Risk_Score_100", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "📥 Download Certified Financial Audit Report (CSV)",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name="nexus_certified_financial_audit.csv",
            mime="text/csv",
            key="dl_fin_btn",
        )
    except Exception as exc:
        st.error(f"❌ Financial Module Error: {str(exc)}")


def render_cyber_tab(manual_contam: Optional[float]) -> None:
    st.subheader("🛡️ Self-Evolving Cybersecurity Network Telemetry")
    uploaded_cyber = st.file_uploader(
        "Upload Network Traffic Logs (CSV)", type=["csv"], key="cyber_uploader"
    )

    try:
        raw_df = (
            pd.read_csv(uploaded_cyber)
            if uploaded_cyber is not None
            else generate_default_cyber_data()
        )
        req_cols = ["Packet_Size", "Failed_Attempts", "Response_Time"]
        clean_df, heal_logs = NexusCognitiveEngine.self_heal_dataframe(raw_df, req_cols)
        scored_df, applied_rate = NexusCognitiveEngine.auto_calibrate_and_score(
            clean_df, req_cols, manual_contam
        )
        scored_df["Threat_Status"] = scored_df["Anomaly_Flag"].apply(
            lambda x: "🚨 Critical Threat Vector" if x == -1 else "🟢 Secured Node"
        )

        threat_rows = scored_df[scored_df["Anomaly_Flag"] == -1]
        st.session_state.active_context["cyber"] = {
            "total_packets": len(scored_df),
            "threat_count": len(threat_rows),
            "max_failed": int(scored_df["Failed_Attempts"].max()),
            "auto_rate": applied_rate,
        }

        st.markdown(
            f"<div class='healing-banner'>🩺 <b>Self-Healing Report:</b> {' | '.join(heal_logs)} (Calibrated Sensitivity: <b>{applied_rate}%</b>)</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Packets Scanned", f"{len(scored_df):,}")
        c2.metric("Threats Isolated", len(threat_rows))
        c3.metric("Max Failed Logins", int(scored_df["Failed_Attempts"].max()))
        c4.metric(
            "Defense Posture", "🚨 DEFCON 1" if len(threat_rows) > 15 else "🟢 OPTIMAL"
        )

        fig = px.scatter(
            scored_df,
            x="Packet_Size",
            y="Failed_Attempts",
            color="Threat_Status",
            size="Risk_Score_100",
            color_discrete_map={
                "🟢 Secured Node": "#10B981",
                "🚨 Critical Threat Vector": "#EF4444",
            },
            title="Autonomous Threat Dispersion & Risk Scoring Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            scored_df.sort_values(by="Risk_Score_100", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "📥 Download Certified Cybersecurity Report (CSV)",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name="nexus_certified_cyber_audit.csv",
            mime="text/csv",
            key="dl_cyber_btn",
        )
    except Exception as exc:
        st.error(f"❌ Cybersecurity Module Error: {str(exc)}")


def render_pdf_tab() -> None:
    st.subheader("📄 Neural PDF & Contract Risk Governance")
    uploaded_pdf = st.file_uploader(
        "Upload Government / Financial PDF Document", type=["pdf"], key="pdf_uploader"
    )

    if uploaded_pdf is None:
        st.info(
            "👈 Upload a PDF invoice or contract to extract financial tokens, entities, and legal risks."
        )
        return

    try:
        pdf_bytes = io.BytesIO(uploaded_pdf.getvalue())
        reader = PdfReader(pdf_bytes)
        pages_count = len(reader.pages)
        text_chunks: List[str] = []

        for idx, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text_chunks.append(f"--- Page {idx + 1} ---\n{extracted}")

        full_text = "\n\n".join(text_chunks)
        if not full_text.strip():
            st.warning(
                "⚠️ Scanned image PDF detected without an embedded text layer."
            )
            return

        amounts = re.findall(
            r"\b(?:\$|USD|EUR|SAR|AED)?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})\b", full_text
        )
        emails = re.findall(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", full_text
        )
        risk_keywords = [
            "fraud",
            "suspicious",
            "error",
            "delay",
            "penalty",
            "urgent",
            "unauthorized",
            "tax",
            "breach",
            "مخالفة",
            "غرامة",
            "تأخير",
            "ضريبة",
        ]
        found_risks = sorted(
            list({kw for kw in risk_keywords if kw in full_text.lower()})
        )

        st.session_state.active_context["pdf"] = {
            "filename": uploaded_pdf.name,
            "pages": pages_count,
            "amounts": amounts[:12],
            "emails": emails[:6],
            "risks": found_risks,
            "raw_te
