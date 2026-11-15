import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="NEXUS AI™ Enterprise", page_icon="⚡", layout="wide")

st.title("⚡ NEXUS AI™ - Enterprise Core")
st.markdown("### منصة الذكاء الاصطناعي للتدقيق والأمن السيبراني")

tab1, tab2 = st.tabs(["📊 التدقيق المالي", "🛡️ الأمن السيبراني"])

with tab1:
    st.subheader("تحليل المعاملات المالية واكتشاف الاحتيال")
    uploaded = st.file_uploader("رفع ملف CSV للمعاملات", type=["csv"])
    
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'Transaction_Amount': np.random.normal(2000, 300, 100),
            'Tax_Value': np.random.normal(150, 20, 100)
        })
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = model.fit_predict(df[['Transaction_Amount', 'Tax_Value']])
    df['Status'] = df['Anomaly'].apply(lambda x: 'مشبوح / احتيال' if x == -1 else 'سليم ومتوافق')
    
    st.dataframe(df, use_container_width=True)
    fig = px.scatter(df, x='Transaction_Amount', y='Tax_Value', color='Status', title="تحليل الشذوذ المالي")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("تحليل حركات الشبكة والأمن السيبراني")
    np.random.seed(42)
    df_cyber = pd.DataFrame({
        'Packet_Size': np.random.normal(500, 50, 100),
        'Failed_Attempts': np.random.poisson(1, 100)
    })
    model_c = IsolationForest(contamination=0.05, random_state=42)
    df_cyber['Status'] = model_c.fit_predict(df_cyber)
    df_cyber['Status'] = df_cyber['Status'].apply(lambda x: 'تهديد حرج' if x == -1 else 'آمن')
    
    st.dataframe(df_cyber, use_container_width=True)
    fig_c = px.scatter(df_cyber, x='Packet_Size', y='Failed_Attempts', color='Status', title="مراقبة الحزم السيبرانية")
    st.plotly_chart(fig_c, use_container_width=True)
    
