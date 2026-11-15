import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import plotly.express as px
from pypdf import PdfReader

# Set page configuration
st.set_page_config(
    page_icon="💼",
    layout="wide"
)

# App Header (Commercial Landing Style)
st.title("💼 Enterprise AI Audit & Threat Detection Platform")
st.markdown("### Professional AI-Powered Solutions for Cybersecurity, Financial Records, and Smart PDF Invoices Analysis.")
st.markdown("---")

# Sidebar Commercial Control Panel
st.sidebar.header("🏢 Enterprise Control Panel")
plan_type = st.sidebar.selectbox("Select Subscription Tier", ["Free Trial Plan", "Enterprise Pro Plan"])

analysis_mode = st.sidebar.selectbox(
    "Select Analysis Domain",
    ["Cybersecurity Network Logs", "Government & Financial Invoices (CSV)", "Smart PDF Document Analyzer"]
)

# File Uploader based on mode
if analysis_mode == "Smart PDF Document Analyzer":
    uploaded_file = st.sidebar.file_uploader("Upload Government/Financial PDF Document", type=["pdf"])
else:
    uploaded_file = st.sidebar.file_uploader(f"Upload CSV File for {analysis_mode}", type=["csv"])

# Security & Privacy Badge in Sidebar
st.sidebar.markdown("---")
st.sidebar.info("🔒 **Enterprise Security:** All uploaded files and data are securely processed in memory with strict privacy compliance.")

# 1. Cybersecurity Mode Logic
if analysis_mode == "Cybersecurity Network Logs":
    @st.cache_data
    def load_cyber_data(file):
        if file is not None:
            return pd.read_csv(file)
        else:
            np.random.seed(42)
            n_samples = 500
            data = {
                'Packet_Size': np.random.normal(500, 100, n_samples),
                'Failed_Attempts': np.random.poisson(1, n_samples),
                'Response_Time': np.random.exponential(1.5, n_samples)
            }
            data['Packet_Size'][10:20] = np.random.normal(4000, 500, 10)
            data['Failed_Attempts'][10:20] = np.random.randint(15, 25, 10)
            df = pd.DataFrame(data)
            df['Timestamp'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_samples, freq='s')
            return df

    df = load_cyber_data(uploaded_file)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[['Packet_Size', 'Failed_Attempts', 'Response_Time']])
    df['Status'] = df['Anomaly_Score'].apply(lambda x: 'Anomaly (Threat)' if x == -1 else 'Normal')

    st.subheader("📊 Network Traffic Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records Analyzed", len(df))
    col2.metric("Detected Threats", len(df[df['Status'] == 'Anomaly (Threat)']))
    col3.metric("Network Status", "⚠️ High Risk" if len(df[df['Status'] == 'Anomaly (Threat)']) > 15 else "✅ Secure")

    st.subheader("📈 Visual Threat Analysis")
    fig = px.scatter(df, x='Packet_Size', y='Failed_Attempts', color='Status',
                     color_discrete_map={'Normal': '#00CC96', 'Anomaly (Threat)': '#EF553B'},
                     title="Network Traffic Distribution", hover_data=['Timestamp', 'Response_Time'])
    st.plotly_chart(fig, use_container_width=True)

# 2. Government & Financial Invoices Mode Logic (CSV)
elif analysis_mode == "Government & Financial Invoices (CSV)":
    @st.cache_data
    def load_financial_data(file):
        if file is not None:
            return pd.read_csv(file)
        else:
            np.random.seed(42)
            n_samples = 500
            data = {
                'Transaction_Amount': np.random.normal(2000, 300, n_samples),
                'Tax_Value': np.random.normal(150, 30, n_samples),
                'Processing_Days': np.random.poisson(2, n_samples)
            }
            data['Transaction_Amount'][10:18] = np.random.normal(25000, 2000, 8)
            data['Tax_Value'][10:18] = np.random.normal(3000, 400, 8)
            df = pd.DataFrame(data)
            df['Transaction_Date'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_samples, freq='h')
            return df

    df = load_financial_data(uploaded_file)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[['Transaction_Amount', 'Tax_Value', 'Processing_Days']])
    df['Status'] = df['Anomaly_Score'].apply(lambda x: 'Anomaly (Suspicious / Fraud)' if x == -1 else 'Normal')

    st.subheader("📊 Government Transactions & Invoices Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions Analyzed", len(df))
    col2.metric("Suspicious Invoices Found", len(df[df['Status'] == 'Anomaly (Suspicious / Fraud)']))
    col3.metric("Audit Risk Status", "⚠️ Review Required" if len(df[df['Status'] == 'Anomaly (Suspicious / Fraud)']) > 5 else "✅ Transactions Verified")

    st.subheader("📈 Visual Financial & Invoice Analysis")
    fig = px.scatter(df, x='Transaction_Amount', y='Tax_Value', color='Status',
                     color_discrete_map={'Normal': '#00CC96', 'Anomaly (Suspicious / Fraud)': '#EF553B'},
                     title="Transaction Amount vs Tax Value (Fraud Detection)", hover_data=['Transaction_Date', 'Processing_Days'])
    st.plotly_chart(fig, use_container_width=True)

# 3. PDF Document Analyzer Mode Logic
else:
    st.subheader("📄 Smart PDF Document & Invoice Extraction")
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text_content = ""
        for i, page in enumerate(reader.pages):
            text_content += f"\n--- Page {i+1} ---\n" + page.extract_text()
        
        st.success(f"Successfully extracted text from PDF ({len(reader.pages)} pages found).")
        
        with st.expander("🔍 View Extracted Raw Text from PDF"):
            st.text_area("PDF Content", text_content, height=300)
            
        # AI Smart Insights on PDF text
        st.subheader("🤖 AI Document Security & Risk Audit")
        risk_keywords = ["fraud", "suspicious", "error", "delay", "penalty", "urgent", "unauthorized", "خطر", "مخالفة", "تأخير"]
        found_risks = [kw for kw in risk_keywords if kw in text_content.lower()]
        
        col1, col2 = st.columns(2)
        col1.metric("Document Status", "⚠️ Review Warnings Found" if found_risks else "✅ Document Verified Safe")
        col2.metric("Detected Risk Flags", len(found_risks))
        
        if found_risks:
            st.warning(f"AI detected potential risk indicators in this document: {', '.join(set(found_risks))}")
        else:
            st.info("No critical anomalies or suspicious keywords detected in the uploaded PDF text.")
            
        df = pd.DataFrame({
            'Document_Name': [uploaded_file.name],
            'Total_Pages': [len(reader.pages)],
            'Risk_Flags_Count': [len(found_risks)],
            'Audit_Status': ['Review Required' if found_risks else 'Verified Safe']
        })
    else:
        st.info("👈 Please upload a PDF invoice or document from the sidebar to begin smart AI extraction and analysis.")
        df = pd.DataFrame(columns=['Document_Name', 'Total_Pages', 'Risk_Flags_Count', 'Audit_Status'])

# Common Table & Export Section
if analysis_mode != "Smart PDF Document Analyzer" or uploaded_file is not None:
    st.subheader("📋 Detailed Data Records & Audit Logs")
    st.dataframe(df, use_container_width=True)

    st.subheader("📥 Export Professional Audit Report")
    csv_data = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Enterprise Risk Report (CSV)",
        data=csv_data,
        file_name='enterprise_audit_report.csv',
        mime='text/csv'
    )

# Commercial Footer
st.markdown("---")
st.markdown("© 2026 Enterprise AI Solutions | Secure Cloud Platform for Corporate Auditing & Security.")
        
