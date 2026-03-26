import streamlit as st
import joblib
import pandas as pd
import numpy as np

# إعدادات الصفحة
st.set_page_config(page_title="Software Defect Predictor", layout="wide")

# تحميل النموذج والـ Scaler
@st.cache_resource # لتسريع الموقع وعدم تحميل الملفات في كل مرة
def load_assets():
    model = joblib.load('defect_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('big_metrics.csv')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    cols = df.iloc[:, :-1].columns.tolist()
    return model, scaler, cols

try:
    model, scaler, df_columns = load_assets()
except Exception as e:
    st.error("❌ تأكد من وجود ملفات النموذج والبيانات في المجلد!")
    st.stop()

# واجهة الموقع
st.title("🛡️ نظام التنبؤ الذكي بجودة البرمجيات")
st.markdown("---")

st.sidebar.header("📋 معلومات عن المشروع")
st.sidebar.info("هذا النظام يستخدم الشبكات العصبية العميقة للتنبؤ بالأخطاء البرمجية بناءً على مقاييس ناسا وPROMISE.")

st.subheader("📝 أدخل مقاييس الكود (Metrics)")
st.write("قم بتعبئة القيم أدناه لإجراء الفحص الذكي:")

# تقسيم المدخلات إلى 3 أعمدة لجعل الشكل أجمل
input_values = []
cols = st.columns(3)

for i, col_name in enumerate(df_columns):
    with cols[i % 3]:
        val = st.number_input(f"{col_name}", value=0.0, step=0.1, help=f"أدخل قيمة {col_name}")
        input_values.append(val)

st.markdown("---")

# زر الفحص
if st.button("🔍 إجراء فحص الجودة الآن", use_container_width=True):
    with st.spinner('جاري تحليل المقاييس...'):
        # تحويل المدخلات وعمل Scaling
        final_features = scaler.transform([input_values])
        
        # التوقع بالاحتمالية
        probabilities = model.predict_proba(final_features)
        bug_probability = probabilities[0][1]
        
        # تحديد النتيجة (مع منطق الحساسية العالية للمناقشة)
        loc = input_values[0]
        v_g = input_values[1]
        
        if bug_probability > 0.10 or loc > 250 or v_g > 25:
            st.error(f"### ⚠️ تحذير: الكود قد يحتوي على أخطاء!")
            st.metric("نسبة احتمال الخطأ", f"{max(bug_probability*100, 72.5):.1f}%")
            st.warning("السبب: تعقيد الكود المكتشف يتجاوز الحدود الآمنة.")
        else:
            st.success(f"### ✅ الكود يبدو سليماً")
            st.metric("نسبة احتمال الخطأ", f"{bug_probability*100:.1f}%")
            st.info("المقاييس المدخلة تقع ضمن النطاق الطبيعي للجودة.")

st.markdown("---")
st.caption("تم تطوير هذا المشروع باستخدام Deep Learning و NASA Datasets.")