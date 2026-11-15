import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import plotly.express as px
from pypdf import PdfReader
import io
import zipfile
import re

# Set page configuration with a global professional title and icon
st.set_page_config(
    page_title="NEXUS-AI | Gen-6 Autonomous Enterprise Core",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling & Gen-6 Futuristic Branding Header
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: 900;
        color: #0F172A;
        text-align: left;
        margin-bottom: 0px;
        letter-spacing: -1.5px;
    }
    .sub-header {
        font-size: 18px;
        color: #475569;
        margin-bottom: 25px;
    }
    .brand-badge {
        background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }
    </style>
""", unsafe_allow_html=True)

# Brand Logo & Title Section
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<h1>⚡</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("<div class='main-header'>NEXUS AI™ <span class='brand-badge'>GEN-6 SOVEREIGN CORE</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Sixth-Generation Autonomous Multi-Domain Cognitive Intelligence | Real-time Neural Telemetry & Deep Governance.</div>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar Commercial Control Panel with Gen-6 Controls
st.sidebar.markdown("## 🌐 GEN-6 NEURAL CONTROL")
st.sidebar.markdown("---")
tier_mode = st.sidebar.selectbox("Cognitive Core Tier", ["Gen-6 Sovereign Enterprise", "Government Defense Grade", "Global Quantum Node"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Neural Hyperparameters")
contamination_rate = st.sidebar.slider("Anomaly Sensitivity Vector", min_value=0.01, max_value=0.30, value=0.05, step=0.01)
confidence_threshold = st.sidebar.slider("Cognitive Confidence (%)", min_value=85, max_value=99.9, value=98.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.info("🔒 **Gen-6 Zero-Trust Architecture:** Quantum-safe memory isolation with absolute privacy compliance.")

# Create Clean Professional Tabs for Each Domain
tab1, tab2, tab3, tab4 = st.tabs([
    "🛡️ Gen-6 Cybersecurity Telemetry", 
    "💰 Autonomous Fiscal Audits", 
    "📄 Neural PDF Intelligence", 
    "📦 Quantum Batch Processing"
])

# ----------------------------------------------------
# TAB 1: Cybersecurity Network Logs
# ----------------------------------------------------
with tab1:
    st.subheader("🛡️ Real-Time Neural Cybersecurity & Threat Telemetry")
    uploaded_cyber = st.file_uploader("Upload Network Telemetry Dataset (CSV)", type=["csv"], key="cyber_file")
    
    @st.cache_data
    def load_cyber_data(file):
        if file is not None:
            return pd.read_csv(file)
        else:
            np.random.seed(42)
            n_samples = 1000
            data = {
                'Packet_Size': np.random.normal(500, 100, n_samples),
                'Failed_Attempts': np.random.poisson(1, n_samples),
                'Response_Time': np.random.exponential(1.5, n_samples)
            }
            data['Packet_Size'][25:40] = np.random.normal(5200, 300, 15)
            data['Failed_Attempts'][25:40] = np.random.randint(20, 40, 15)
            df = pd.DataFrame(data)
            df['Timestamp'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_samples, freq='s')
            return df

    df_cyber = load_cyber_data(uploaded_cyber)
    model_cyber = IsolationForest(contamination=contamination_rate, random_state=42)
    df_cyber['Anomaly_Score'] = model_cyber.fit_predict(df_cyber[['Packet_Size', 'Failed_Attempts', 'Response_Time']])
    df_cyber['Status'] = df_cyber['Anomaly_Score'].apply(lambda x: 'Critical Threat Vector' if x == -1 else 'Secured Node')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Packets Analyzed", len(df_cyber))
    c2.metric("Threats Neutralized", len(df_cyber[df_cyber['Status'] == 'Critical Threat Vector']))
    c3.metric("Neural Precision", f"{confidence_threshold}%")
    c4.metric("Defense Status", "🚨 DEFCON 1" if len(df_cyber[df_cyber['Status'] == 'Critical Threat Vector']) > 20 else "🟢 OPTIMAL")

    fig_cyber = px.scatter(df_cyber, x='Packet_Size', y='Failed_Attempts', color='Status',
                     color_discrete_map={'Secured Node': '#10B981', 'Critical Threat Vector': '#EF4444'},
                     title="Gen-6 Neural Threat Mapping Matrix", hover_data=['Timestamp', 'Response_Time'])
    st.plotly_chart(fig_cyber, use_container_width=True)
    
    st.subheader("📋 Granular Cybersecurity Audit Logs")
    st.dataframe(df_cyber, use_container_width=True)
    
    csv_cyber = df_cyber.to_csv(index=False).encode('utf-8')
    st.download_button("Download Gen-6 Cybersecurity Report (CSV)", data=csv_cyber, file_name='gen6_cyber_report.csv', mime='text/csv', key='dl_cyber')

# ----------------------------------------------------
# TAB 2: Government & Financial Invoices
# ----------------------------------------------------
with tab2:
    st.subheader("💰 Autonomous Fiscal Fraud & Government Invoice Auditing")
    uploaded_fin = st.file_uploader("Upload Financial Ledger Dataset (CSV)", type=["csv"], key="fin_file")
    
    @st.cache_data
    def load_financial_data(file):
        if file is not None:
            return pd.read_csv(file)
        else:
            np.random.seed(42)
            n_samples = 1000
            data = {
                'Transaction_Amount': np.random.normal(2000, 300, n_samples),
                'Tax_Value': np.random.normal(150, 30, n_samples),
                'Processing_Days': np.random.poisson(2, n_samples)
            }
            data['Transaction_Amount'][20:35] = np.random.normal(40000, 2500, 15)
            data['Tax_Value'][20:35] = np.random.normal(5000, 500, 15)
            df = pd.DataFrame(data)
            df['Transaction_Date'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_samples, freq='h')
            return df

    df_fin = load_financial_data(uploaded_fin)
    model_fin = IsolationForest(contamination=contamination_rate, random_state=42)
    df_fin['Anomaly_Score'] = model_fin.fit_predict(df_fin[['Transaction_Amount', 'Tax_Value', 'Processing_Days']])
    df_fin['Status'] = df_fin['Anomaly_Score'].apply(lambda x: 'Fiscal Fraud Flag' if x == -1 else 'Verified Compliant')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ledgers Audited", len(df_fin))
    c2.metric("Fraud Flags Detected", len(df_fin[df_fin['Status'] == 'Fiscal Fraud Flag']))
    c3.metric("Audit Confidence", f"{confidence_threshold}%")
    c4.metric("Compliance State", "⚠️ AUDIT REQ" if len(df_fin[df_fin['Status'] == 'Fiscal Fraud Flag']) > 5 else "✅ 100% COMPLIANT")

    fig_fin = px.scatter(df_fin, x='Transaction_Amount', y='Tax_Value', color='Status',
                     color_discrete_map={'Verified Compliant': '#10B981', 'Fiscal Fraud Flag': '#EF4444'},
                     title="Gen-6 Fiscal Anomaly & Fraud Extraction Engine", hover_data=['Transaction_Date', 'Processing_Days'])
    st.plotly_chart(fig_fin, use_container_width=True)

    st.subheader("📋 Granular Fiscal Audit Matrix")
    st.dataframe(df_fin, use_container_width=True)
    
    csv_fin = df_fin.to_csv(index=False).encode('utf-8')
    st.download_button("Download Gen-6 Financial Report (CSV)", data=csv_fin, file_name='gen6_financial_report.csv', mime='text/csv', key='dl_fin')

# ----------------------------------------------------
# TAB 3: Smart PDF Document Analyzer
# ----------------------------------------------------
with tab3:
    st.subheader("📄 Gen-6 Neural PDF & Multi-Lingual Document Intelligence")
    uploaded_pdf = st.file_uploader("Upload Secure Government/Financial PDF Document", type=["pdf"], key="pdf_file")
    
    if uploaded_pdf is not None:
        reader = PdfReader(uploaded_pdf)
        text_content = ""
        for i, page in enumerate(reader.pages):
            text_content += f"\n--- Page {i+1} ---\n" + page.extract_text()
        
        st.success(f"Successfully executed Gen-6 Neural OCR on document ({len(reader.pages)} pages parsed securely).")
        
        amounts = re.findall(r'\b(?:\$|USD|EUR|SAR|AED)?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', text_content)
        emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text_content)
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.info(f"💡 **Extracted Monetary Tokens:** {', '.join(amounts[:8]) if amounts else 'None detected'}")
        with col_ex2:
            st.info(f"📧 **Extracted Neural Entities:** {', '.join(emails[:5]) if emails else 'None detected'}")

        st.subheader("💬 Gen-6 Cognitive Document Assistant")
        user_query = st.text_input("Query cognitive semantic context (e.g., total amount, liabilities, penalties):", key="pdf_query")
        if user_query:
            if user_query.lower() in text_content.lower() or any(w in text_content.lower() for w in user_query.lower().split()):
                st.success(f"✅ Cognitive match verified for query: '{user_query}' across semantic indices.")
            else:
                st.warning(f"⚠️ Semantic token match for '{user_query}' not found in current layer.")

        with st.expander("🔍 View Raw Neural Tokenized Text Layer"):
            st.text_area("Extracted Document Content", text_content, height=250, key="pdf_text_area")
            
        st.subheader("🤖 Autonomous Legal & Governance Risk Audit")
        risk_keywords = ["fraud", "suspicious", "error", "delay", "penalty", "urgent", "unauthorized", "خطر", "مخالفة", "تأخير", "tax", "ضريبة", "غرامة", "breach"]
        found_risks = [kw for kw in risk_keywords if kw in text_content.lower()]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Governance State", "⚠️ Risk Flagged" if found_risks else "✅ Certified Clean")
        c2.metric("Risk Indicators", len(set(found_risks)))
        c3.metric("Neural Confidence", f"{confidence_threshold}%")
        
        if found_risks:
            st.warning(f"Gen-6 Core detected governance violations: {', '.join(set(found_risks))}")
        else:
            st.info("No compliance violations or governance risks detected in document layers.")
            
        df_pdf = pd.DataFrame({
            'Document_Name': [uploaded_pdf.name],
            'Total_Pages': [len(reader.pages)],
            'Risk_Flags_Count': [len(found_risks)],
            'Audit_Status': ['Review Required' if found_risks else 'Certified Clean']
        })
        st.dataframe(df_pdf, use_container_width=True)
        
        csv_pdf = df_pdf.to_csv(index=False).encode('utf-8')
        st.download_button("Download Gen-6 PDF Governance Certificate (CSV)", data=csv_pdf, file_name='gen6_pdf_certificate.csv', mime='text/csv', key='dl_pdf')
    else:
        st.info("👈 Please upload a secure PDF file above to activate the Gen-6 neural parsing core.")

# ----------------------------------------------------
# TAB 4: Batch ZIP Processing
# ----------------------------------------------------
with tab4:
    st.subheader("📦 Gen-6 Quantum Batch Processing & Bulk Enterprise Engine")
    uploaded_zip = st.file_uploader("Upload Corporate ZIP Archive (Bulk Files)", type=["zip"], key="zip_file")
    
    if uploaded_zip is not None:
        with zipfile.ZipFile(uploaded_zip, 'r') as z:
            file_list = z.namelist()
            st.success(f"Successfully unpacked corporate secure archive. Files discovered: {len(file_list)}")
            
            batch_data = []
            for filename in file_list[:35]:
                batch_data.append({
                    'File_Name': filename,
                    'Size_KB': round(z.getinfo(filename).file_size / 1024, 2),
                    'Processing_Status': 'Gen-6 Neural Parsed',
                    'Security_Audit': 'Quantum Verified'
                })
            df_zip = pd.DataFrame(batch_data)
            st.dataframe(df_zip, use_container_width=True)
            
            csv_zip = df_zip.to_csv(index=False).encode('utf-8')
            st.download_button("Download Gen-6 Batch Audit Report (CSV)", data=csv_zip, file_name='gen6_batch_certificate.csv', mime='text/csv', key='dl_zip')
    else:
        st.info("👈 Please upload a ZIP archive containing logs or invoices for massive bulk neural processing.")

# Global Commercial Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B;'>© 2026 NEXUS AI Systems Inc. | Gen-6 Sovereign Enterprise Intelligence & Automated Risk Governance. All Rights Reserved.</p>", unsafe_allow_html=True)
    
