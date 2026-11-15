import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Cyber Anomaly & Threat Detector",
    page_icon="🛡️",
    layout="wide"
)

# App Header
st.title("🛡️ Cyber Anomaly & Threat Detection System")
st.markdown("An AI-powered security tool to detect malicious network activities and anomalies using Machine Learning.")

# Sidebar Controls
st.sidebar.header("Configuration Panel")
st.sidebar.markdown("Upload your network logs or use generated sample data.")

# File Uploader for Real Data
uploaded_file = st.sidebar.file_uploader("Upload Network Logs (.csv)", type=["csv"])

# Data Generation or Loading
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Generate realistic sample data if no file is uploaded
        np.random.seed(42)
        n_samples = 500
        data = {
            'Packet_Size': np.random.normal(500, 100, n_samples),
            'Failed_Attempts': np.random.poisson(1, n_samples),
            'Response_Time': np.random.exponential(1.5, n_samples)
        }
        # Inject some synthetic anomalies (attacks)
        data['Packet_Size'][10:20] = np.random.normal(4000, 500, 10)
        data['Failed_Attempts'][10:20] = np.random.randint(15, 25, 10)
        
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_samples, freq='s')
    return df

df = load_data(uploaded_file)

# Machine Learning Model (Isolation Forest)
model = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_Score'] = model.fit_predict(df[['Packet_Size', 'Failed_Attempts', 'Response_Time']])
df['Status'] = df['Anomaly_Score'].apply(lambda x: 'Anomaly (Threat)' if x == -1 else 'Normal')

# Main Metrics Dashboard
st.subheader("📊 Network Traffic Overview")
col1, col2, col3 = st.columns(3)

total_records = len(df)
total_threats = len(df[df['Status'] == 'Anomaly (Threat)'])
network_status = "⚠️ High Risk Detected" if total_threats > 15 else "✅ Relatively Secure"

col1.metric("Total Records Analyzed", total_records)
col2.metric("Detected Threats / Anomalies", total_threats)
col3.metric("Overall Network Status", network_status)

# Visualizing Data with Plotly
st.subheader("📈 Visual Threat Analysis")
fig = px.scatter(
    df, 
    x='Packet_Size', 
    y='Failed_Attempts', 
    color='Status',
    color_discrete_map={'Normal': '#00CC96', 'Anomaly (Threat)': '#EF553B'},
    title="Network Traffic Distribution & Anomaly Detection",
    hover_data=['Timestamp', 'Response_Time']
)
st.plotly_chart(fig, use_container_width=True)

# Displaying Data Tables
st.subheader("📋 Filtered Network Logs")
st.dataframe(df, use_container_width=True)

# Export Report Feature
st.subheader("📥 Export Threat Report")
anomalies_df = df[df['Status'] == 'Anomaly (Threat)']

csv_data = anomalies_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Threat Report (CSV)",
    data=csv_data,
    file_name='cyber_threat_report.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("🔒 **Cyber Anomaly Detector Platform** | Built for Global Security Solutions")
