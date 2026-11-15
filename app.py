import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime

# إعدادات الصفحة
st.set_page_config(
    page_title="نظام كشف الشذوذ والأمن السيبراني",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ نظام كشف الشذوذ والأنشطة المشبوهة (Cyber Anomaly Detector)")
st.markdown("---")

# الشريط الجانبي للإعدادات والتحكم
st.sidebar.header("⚙️ إعدادات النموذج والبيانات")

# خيار توليد بيانات وهمية أو رفع ملف
data_source = st.sidebar.radio("اختر مصدر البيانات:", ["توليد بيانات شبكة افتراضية", "رفع ملف CSV"])

if data_source == "توليد بيانات شبكة افتراضية":
    st.sidebar.subheader("خصائص التوليد")
    n_samples = st.sidebar.slider("عدد سجلات حركة المرور:", 100, 2000, 500, 100)
    contamination_rate = st.sidebar.slider("نسبة الشذوذ المتوقعة (Contamination):", 0.01, 0.2, 0.05, 0.01)
    
    # توليد بيانات شبكة افتراضية (حجم الحزم، عدد المحاولات، زمن الاستجابة)
    np.random.seed(42)
    packet_size = np.random.normal(500, 100, n_samples)
    failed_attempts = np.random.poisson(1, n_samples)
    response_time = np.random.exponential(0.5, n_samples)
    
    # حقن بعض القيم الشاذة قصداً
    n_anomalies = int(n_samples * contamination_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    packet_size[anomaly_indices] = np.random.uniform(2500, 5000, n_anomalies)
    failed_attempts[anomaly_indices] = np.random.randint(10, 25, n_anomalies)
    
    df = pd.DataFrame({
        'Packet_Size': packet_size,
        'Failed_Attempts': failed_attempts,
        'Response_Time': response_time
    })
    
    # إضافة طابع زمني للسجلات
    df['Timestamp'] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * n_samples

else:
    uploaded_file = st.sidebar.file_uploader("اختر ملف CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("الرجاء رفع ملف CSV أو اختيار توليد بيانات افتراضية من الشريط الجانبي.")
        st.stop()

# عرض عينة من البيانات الأصلية
st.subheader("📊 عينة من سجلات الشبكة المرصودة")
st.dataframe(df.head(10), use_container_width=True)

# تدريب نموذج Isolation Forest لكشف الشذوذ
model = IsolationForest(contamination=contamination_rate, random_state=42)
df['Anomaly_Score'] = model.fit_predict(df[['Packet_Size', 'Failed_Attempts', 'Response_Time']])

# تحويل النتائج إلى تسميات واضحة (-1 يعني شذوذ/خطر، 1 يعني طبيعي)
df['Status'] = df['Anomaly_Score'].apply(lambda x: 'مشبوه (Anomaly)' if x == -1 else 'طبيعي (Normal)')

# إحصائيات عامة
col1, col2, col3 = st.columns(3)
total_records = len(df)
anomalies_count = len(df[df['Status'] == 'مشبوه (Anomaly)'])
normal_count = total_records - anomalies_count

col1.metric("إجمالي السجلات المفحوصة", total_records)
col2.metric("التهديدات / الأنشطة الشاذة", anomalies_count, delta_color="inverse")
col3.metric("الحالة العامة للشبكة", "آمنة نسبياً" if anomalies_count < (total_records * 0.1) else "تحذير: نشاط عالي!")

st.markdown("---")

# تصور البيانات مرئياً باستخدام Plotly
st.subheader("📈 التحليل البصري لحركة المرور")
fig = px.scatter(
    df, 
    x='Packet_Size', 
    y='Failed_Attempts', 
    color='Status',
    color_discrete_map={'طبيعي (Normal)': '#00CC96', 'مشبوه (Anomaly)': '#EF553B'},
    title="توزيع حركة المرور واكتشاف القيم الشاذة",
    labels={'Packet_Size': 'حجم الحزمة (بايت)', 'Failed_Attempts': 'محاولات الفشل'}
)
st.plotly_chart(fig, use_container_width=True)

# عرض السجلات المشبوهة وحدها
if anomalies_count > 0:
    st.subheader("⚠️ سجلات الأنشطة المشبوهة المكتشفة")
    st.dataframe(df[df['Status'] == 'مشبوه (Anomaly)'], use_container_width=True)

    # زر لتصدير التقرير كملف CSV
    csv_data = df[df['Status'] == 'مشبوه (Anomaly)'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 تحميل تقرير التهديدات (CSV)",
        data=csv_data,
        file_name='security_anomalies_report.csv',
        mime='text/csv',
    )
else:
    st.success("لم يتم رصد أي أنشطة شاذة أو هجمات بناءً على المعايير الحالية.")
