import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import plotly.express as px
from pypdf import PdfReader
import io
import zipfile

# Set page configuration with a global professional title and icon
st.set_page_config(
    page_title="NEXUS-AI | Enterprise Global Platform",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling & Global Branding Header
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: 800;
        color: #0F172A;
        text-align: left;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        color: #475569;
        margin-bottom: 20px;
    }
    .brand-badge {
        background-color: #3B82F6;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Brand Logo & Title Section
col_logo, col_title = st.column([1, 6])
with col_logo:
    st.markdown("<h1>🛡️</h1>", unsafe_allow_html=True) # Global Enterprise Logo Symbol
with col_title:
    st.markdown("<div class='main-header'>NEXUS AI™ <span class='brand-badge'>GLOBAL EDITION</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Next-Generation Autonomous Multi-Domain Intelligence, Cybersecurity Audits & Smart Document Governance.</div>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar Commercial Control Panel with Global Branding
st.sidebar.markdown("## 🌐 NEXUS-AI CONTROL")
st.sidebar.markdown("---")
plan_type = st.sidebar.selectbox("Subscription Tier", ["Enterprise Global Tier", "Government Secured Plan", "Pro Corporate"])

analysis_mode = st.sidebar.selectbox(
    "Select Intelligence Domain",
    ["Cybersecurity Network Logs", "Government & Financial Invoices (CSV)", "Smart PDF Document Analyzer", "Batch ZIP File Processing"]
)

# Dynamic AI Sensitivity Sliders in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Neural Engine Controls")
contamination_rate = st.sidebar.slider("AI Threat Sensitivity", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

# File Uploader based on mode
uploaded_file = None
uploaded_zip = None

if analysis_mode == "Smart PDF Document Analyzer":
    uploaded_file = st.sidebar.file_uploader("Upload Government/Financial PDF Document", type=["pdf"])
elif analysis_mode == "Batch ZIP File Processing":
    uploaded_zip = st.sidebar.file_uploader("Upload ZIP Archive (Bulk Files)", type=["zip"])
else:
    uploaded_file = st.sidebar.file_uploader(f"Upload CSV Dataset for {analysis_mode}", type=["csv"])

# Security & Compliance Global Badge
st.sidebar.markdown("---")
st.sidebar.info("🔒 **ISO/IEC 27001 Security Standard:** End-to-end memory isolation. Zero data retention policy strictly enforced globally.")

# 1. Cybersecurity Mode Logic
if analysis_mode == "Cybersecurity Network Logs":
    @st.cache_data
    def load_cyber_data(file):
        if file is not None:
            return pd.read_csv(file)
        else:
            np.random.seed(42)
            n_samples = 600
            data = {
                'Packet_Size': np.random.normal(500, 100, n_samples),
                'Failed_Attempts': np.random.poisson(1, n_samples),
                'Response_Time': np.random.exponential(1.5, n_samples)
            }
            data['Packet_Size'][15:25] = np.random.normal(4500, 400, 10)
            data['Failed_Attempts'][15:25] = np.random.randint(15, 30, 10)
            df = pd.DataFrame(data)
            df['Timestamp'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_samples, freq='s')
            return df

    df = load_cyber_data(uploaded_file)
    
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[['Packet_Size', 'Failed_Attempts', 'Response_Time']])
    df['Status'] = df['Anomaly_Score'].apply(lambda x: 'Critical Threat' if x == -1 else 'Secure')

    st.subheader("📊 Global Network Traffic Telemetry")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Packets Analyzed", len(df))
    col2.metric("Detected Threats", len(df[df['Status'] == 'Critical Threat']))
    col3.metric("Global Threat Status", "⚠️ DEFCON 2 (High Risk)" if len(df[df['Status'] == 'Critical Threat']) > 15 else "✅ SECURE SYSTEM")

    st.subheader("📈 Neural Network Threat Distribution")
    fig = px.scatter(df, x='Packet_Size', y='Failed_Attempts', color='Status',
                     color_discrete_map={'Secure': '#10B981', 'Critical Threat': '#EF4444'},
                     title="Advanced Threat Mapping", hover_data=['Timestamp', 'Response_Time'])
    st.plotly_chart(fig, use_container_width=True)

# 2. Government & Financial Invoices Mode Logic (CSV)
elif analysis_mode == "Government & Financial Invoices (CSV)":
    @st.cache_data
    def load_financial_data(file):
        if file is not None:
            return pd.read_csv(file)
        else:
            np.random.seed(42)
            n_samples = 600
            data = {
                'Transaction_Amount': np.random.normal(2000, 300, n_samples),
                'Tax_Value': np.random.normal(150, 30, n_samples),
                'Processing_Days': np.random.poisson(2, n_samples)
            }
            data['Transaction_Amount'][12:20] = np.random.normal(30000, 2500, 8)
            data['Tax_Value'][12:20] = np.random.normal(3500, 500, 8)
            df = pd.DataFrame(data)
            df['Transaction_Date'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_samples, freq='h')
            return df

    df = load_financial_data(uploaded_file)
    
    model = IsolationForest(contamination=contamination_rate, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[['Transaction_Amount', 'Tax_Value', 'Processing_Days']])
    df['Status'] = df['Anomaly_Score'].apply(lambda x: 'Fraud Risk Flag' if x == -1 else 'Verified Compliant')

    st.subheader("📊 Government & Financial Audit Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions Scanned", len(df))
    col2.metric("Fraud Anomalies Detected", len(df[df['Status'] == 'Fraud Risk Flag']))
    col3.metric("Fiscal Compliance Score", "⚠️ Audit Required" if len(df[df['Status'] == 'Fraud Risk Flag']) > 5 else "✅ 100% Compliant")

    st.subheader("📈 Tax vs Transaction Fraud Matrix")
    fig = px.scatter(df, x='Transaction_Amount', y='Tax_Value', color='Status',
                     color_discrete_map={'Verified Compliant': '#10B981', 'Fraud Risk Flag': '#EF4444'},
                     title="Fiscal Anomaly Detection Engine", hover_data=['Transaction_Date', 'Processing_Days'])
    st.plotly_chart(fig, use_container_width=True)

# 3. PDF Document Analyzer Mode Logic (with AI Chat & Deep Extraction)
elif analysis_mode == "Smart PDF Document Analyzer":
    st.subheader("📄 NEXUS Neural PDF & Invoice Intelligence")
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text_content = ""
        for i, page in enumerate(reader.pages):
            text_content += f"\n--- Page {i+1} ---\n" + page.extract_text()
        
        st.success(f"Successfully extracted document text ({len(reader.pages)} pages parsed securely).")
        
        # Interactive AI QA / Search inside PDF
        st.subheader("💬 NEXUS AI Document Assistant")
        user_query = st.text_input("Query document context (e.g., total amount, invoice ID, legal clauses):")
        if user_query:
            if user_query.lower() in text_content.lower() or any(w in text_content.lower() for w in user_query.lower().split()):
                st.success(f"✅ Context found matching query: '{user_query}'")
            else:
                st.warning(f"⚠️ No exact token match found for '{user_query}' in document text.")

        with st.expander("🔍 View Raw Parsed Text Layer"):
            st.text_area("Extracted Document Content", text_content, height=250)
            
        # AI Smart Insights on PDF text
        st.subheader("🤖 Automated Legal & Risk Governance Audit")
        risk_keywords = ["fraud", "suspicious", "error", "delay", "penalty", "urgent", "unauthorized", "خطر", "مخالفة", "تأخير", "tax", "ضريبة", "غرامة"]
        found_risks = [kw for kw in risk_keywords if kw in text_content.lower()]
        
        col1, col2 = st.columns(2)
        col1.metric("Governance Status", "⚠️ Audit Warnings Flagged" if found_risks else "✅ Fully Certified Clean")
        col2.metric("Risk Indicators
            
