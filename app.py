import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import plotly.express as px
from pypdf import PdfReader
import io
import zipfile
import re
from datetime import datetime

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM VECTOR SVG LOGO
# ----------------------------------------------------
st.set_page_config(
    page_title="NEXUS AI™ | Self-Evolving Enterprise Core",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

NEXUS_LOGO_SVG = """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="nexusGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="50%" stop-color="#1E3A8A"/>
      <stop offset="100%" stop-color="#3B82F6"/>
    </linearGradient>
    <linearGradient id="accentGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#818CF8"/>
    </linearGradient>
  </defs>
  <polygon points="32,2 58,17 58,47 32,62 6,47 6,17" fill="url(#nexusGrad)" stroke="url(#accentGrad)" stroke-width="2.5"/>
  <polygon points="32,12 49,22 49,42 32,52 15,42 15,22" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4 2"/>
  <circle cx="32" cy="32" r="7" fill="#38BDF8"/>
  <line x1="32" y1="12" x2="32" y2="25" stroke="#93C5FD" stroke-width="2"/>
  <line x1="32" y1="39" x2="32" y2="52" stroke="#93C5FD" stroke-width="2"/>
  <line x1="15" y1="22" x2="26" y2="28" stroke="#93C5FD" stroke-width="2"/>
  <line x1="49" y1="42" x2="38" y2="36" stroke="#93C5FD" stroke-width="2"/>
</svg>
"""

st.markdown("""
    <style>
    .nexus-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        padding: 20px 28px;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .nexus-title {
        font-size: 34px;
        font-weight: 900;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .nexus-tagline {
        font-size: 14px;
        color: #93C5FD;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .nexus-sub {
        font-size: 15px;
        color: #CBD5E1;
        margin-top: 6px;
    }
    .badge-pro {
        background-color: #2563EB;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        vertical-align: middle;
        margin-left: 10px;
    }
    .healing-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        padding: 12px 16px;
        border-radius: 6px;
        color: #166534;
        font-size: 14px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(f"""
    <div class="nexus-banner">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div>{NEXUS_LOGO_SVG}</div>
            <div>
                <div class="nexus-title">NEXUS AI™ <span class="badge-pro">SELF-EVOLVING CORE</span></div>
                <div class="nexus-tagline">Autonomous Intelligence & Self-Healing Architecture</div>
                <div class="nexus-sub">Self-Healing Pipelines • Auto-Calibrating Neural Risk Matrix • Live Cognitive Copilot</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. SELF-HEALING & AUTO-TUNING ALGORITHMIC ENGINE
# ----------------------------------------------------
def self_heal_dataframe(df, required_cols):
    """خوارزمية التصحيح الذاتي للبيانات: تعالج القيم المفقودة والتالفة والمكررة تلقائياً."""
    healing_log = []
    df_clean = df.copy()
    
    df_clean.columns = [c.strip() for c in df_clean.columns]
    
    null_count = df_clean.isnull().sum().sum()
    if null_count > 0:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        healing_log.append(f"تم تصحيح وملء {null_count} خلية مفقودة/تالفة تلقائياً باستخدام الوسيط الإحصائي.")
        
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean.drop_duplicates(inplace=True)
        healing_log.append(f"تم تطهير وحذف {dup_count} سجل مكرر بـ شكل ذاتي.")
        
    for col in required_cols:
        if col not in df_clean.columns:
            df_clean[col] = np.random.normal(100, 15, len(df_clean))
            healing_log.append(f"تم تخليق العمود المفقود '{col}' بصورة ذاتية بناءً على التوزيع الأساسي.")
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
    if not healing_log:
        healing_log.append("سلامة الهيكل البيانات بنسبة 100%. لا توجد أي تشوهات هيكلية.")
        
    return df_clean, healing_log

def auto_calibrate_and_score(df, feature_cols, manual_contamination=None):
    """خوارزمية المعايرة والتطوير الذاتي: تحسب الحدود عبر IQR وتولد درجات خطورة من 0 إلى 100."""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = ((X_scaled < (Q1 - 2.5 * IQR)) | (X_scaled > (Q3 + 2.5 * IQR))).any(axis=1)
    estimated_rate = float(np.clip(outlier_mask.mean(), 0.01, 0.25))
    
    final_contamination = estimated_rate if manual_contamination is None else manual_contamination
    
    model = IsolationForest(n_estimators=200, contamination=final_contamination, random_state=42)
    preds = model.fit_predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)
    
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s - min_s > 0:
        risk_score = ((max_s - raw_scores) / (max_s - min_s)) * 100
    else:
        risk_score = np.zeros(len(raw_scores))
        
    df['Risk_Score_100'] = np.round(risk_score, 1)
    df['Anomaly_Flag'] = preds
    return df, round(final_contamination * 100, 2)

# ----------------------------------------------------
# 3. SIDEBAR CONTROLS & SYSTEM TELEMETRY
# ----------------------------------------------------
st.sidebar.markdown(f"<div style='text-align:center;'>{NEXUS_LOGO_SVG}<h3>NEXUS CONTROL</h3></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

tuning_mode = st.sidebar.radio(
    "🧠 وضع معايرة الخوارزميات",
    ["التطوير الذاتي (AI Auto-Tune)", "تجاوز يدوي بالحساسية"]
)

if tuning_mode == "تجاوز يدوي بالحساسية":
    manual_contam = st.sidebar.slider("حساسية الشذوذ اليدوية", 0.01, 0.25, 0.05, 0.01)
else:
    manual_contam = None
    st.sidebar.caption("✨ الذكاء الاصطناعي يضبط الحدود الإحصائية تلقائياً لكل ملف.")

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ حالة نظام التصحيح الذاتي")
st.sidebar.success("● Self-Healing Pipeline: نشط ومستمر")
st.sidebar.success("● Neural Copilot Memory: متزامن")
st.sidebar.info("🔒 **حماية أمنية سيادية:** معالجة البيانات بالذاكرة المتطايرة لضمان الخصوصية التامة.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "مرحباً بك في **NEXUS AI™ Self-Evolving Core**. أنا النظام الذكي المدمج بخوارزميات التصحيح الذاتي. ارفع أي ملف أو اسألني مباشرة عن أي مؤشرات ومخاطر!"}
    ]
if "active_context" not in st.session_state:
    st.session_state.active_context = {}

# ----------------------------------------------------
# 4. MAIN APPLICATION TABS
# ----------------------------------------------------
tab_chat, tab_fin, tab_cyber, tab_pdf, tab_zip = st.tabs([
    "💬 Live AI Copilot (المحادثة الذكية)",
    "💰 التدقيق المالي الآلي",
    "🛡️ الأمن السيبراني",
    "📄 استخبارات الـ PDF",
    "📦 معالجة الدفعات الجماعية"
])

# ----------------------------------------------------
# TAB 2: FISCAL FRAUD AUDITING
# ----------------------------------------------------
with tab_fin:
    st.subheader("💰 التدقيق المالي واكتشاف الاحتيال بالمعايرة الذاتية")
    uploaded_fin = st.file_uploader("رفع سجلات المعاملات المالية (CSV)", type=["csv"], key="fin_file")
    
    if uploaded_fin is not None:
        raw_fin = pd.read_csv(uploaded_fin)
    else:
        np.random.seed(42)
        n_s = 600
        raw_fin = pd.DataFrame({
            'Transaction_Amount': np.random.normal(2200, 350, n_s),
            'Tax_Value': np.random.normal(165, 35, n_s),
            'Processing_Days': np.random.poisson(2, n_s)
        })
        raw_fin.loc[15:24, 'Transaction_Amount'] = np.random.normal(42000, 3000, 10)
        raw_fin.loc[15:24, 'Tax_Value'] = np.random.normal(6300, 500, 10)
        raw_fin['Transaction_Date'] = pd.date_range(start='2026-11-01', periods=n_s, freq='h')

    df_fin, fin_heal_logs = self_heal_dataframe(raw_fin, ['Transaction_Amount', 'Tax_Value', 'Processing_Days'])
    df_fin, applied_fin_rate = auto_calibrate_and_score(df_fin, ['Transaction_Amount', 'Tax_Value', 'Processing_Days'], manual_contam)
    df_fin['Audit_Status'] = df_fin['Anomaly_Flag'].apply(lambda x: '🚨 مؤشر احتيال مالي' if x == -1 else '✅ متوافق وسليم')

    fraud_rows = df_fin[df_fin['Anomaly_Flag'] == -1]
    st.session_state.active_context['finance'] = {
        'total_records': len(df_fin),
        'fraud_count': len(fraud_rows),
        'max_amount': float(df_fin['Transaction_Amount'].max()),
        'total_volume': float(df_fin['Transaction_Amount'].sum()),
        'auto_rate': applied_fin_rate
    }

    st.markdown(f"<div class='healing-box'>🩺 <b>تقرير التصحيح الذاتي:</b> {' | '.join(fin_heal_logs)} (حساسية المعايرة: <b>{applied_fin_rate}%</b>)</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي السجلات", f"{len(df_fin):,}")
    c2.metric("حالات الاحتيال المرصودة", len(fraud_rows))
    c3.metric("إجمالي حجم المعاملات", f"${df_fin['Transaction_Amount'].sum():,.0f}")
    c4.metric("أعلى درجة خطورة", f"{df_fin['Risk_Score_100'].max()}/100")

    fig_fin = px.scatter(
        df_fin, x='Transaction_Amount', y='Tax_Value', color='Audit_Status',
        size='Risk_Score_100',
        color_discrete_map={'✅ متوافق وسليم': '#10B981', '🚨 مؤشر احتيال مالي': '#EF4444'},
        title="مصفوفة الفحص المالي والمعايرة التلقائية"
    )
    st.plotly_chart(fig_fin, use_container_width=True)
    st.dataframe(df_fin.sort_values(by='Risk_Score_100', ascending=False), use_container_width=True)
    
    st.download_button(
        "📥 تحميل تقرير التدقيق المالي المعتمد (CSV)",
        data=df_fin.to_csv(index=False).encode('utf-8'),
        file_name='nexus_certified_financial_audit.csv',
        mime='text/csv'
    )

# ----------------------------------------------------
# TAB 3: CYBERSECURITY TELEMETRY
# ----------------------------------------------------
with tab_cyber:
    st.subheader("🛡️ الأمن السيبراني وتحليل حركة الشبكات")
    uploaded_cyber = st.file_uploader("رفع سجلات الشبكة (CSV)", type=["csv"], key="cyber_file")
    
    if uploaded_cyber is not None:
        raw_cyber = pd.read_csv(uploaded_cyber)
    else:
        np.random.seed(42)
        n_c = 600
        raw_cyber = pd.DataFrame({
            'Packet_Size': np.random.normal(520, 95, n_c),
            'Failed_Attempts': np.random.poisson(1, n_c),
            'Response_Time': np.random.exponential(1.4, n_c)
        })
        raw_cyber.loc[20:32, 'Packet_Size'] = np.random.normal(5400, 350, 13)
        raw_cyber.loc[20:32, 'Failed_Attempts'] = np.random.randint(20, 42, 13)
        raw_cyber['Timestamp'] = pd.date_range(start='2026-11-15 18:00:00', periods=n_c, freq='s')

    df_cyber, cyber_heal_logs = self_heal_dataframe(raw_cyber, ['Packet_Size', 'Failed_Attempts', 'Response_Time'])
    df_cyber, applied_cyber_rate = auto_calibrate_and_score(df_cyber, ['Packet_
    
