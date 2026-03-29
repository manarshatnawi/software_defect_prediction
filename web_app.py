import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. إعدادات الصفحة
st.set_page_config(page_title="Software Quality AI", page_icon="🛡️", layout="wide")

# 2. تحميل الموديل والمقياس الجديدين
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('defect_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"خطأ في تحميل الموديل: {e}")
        return None, None

model, scaler = load_assets()

# 3. واجهة المستخدم
st.title("🛡️ Software Quality AI Predictor")
st.markdown("تحليل جودة البرمجيات بناءً على مقاييس التصميم (OO Metrics)")
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📊 أدخلي مقاييس الكود (OO Metrics)")
    st.info("يرجى إدخال القيم المستخرجة من تحليل الكود:")
    
    # تأكدي أن هذا الترتيب هو نفس ترتيب الأعمدة في ملف الـ CSV الذي تدرب عليه الموديل
    cbo = st.number_input("CBO (Coupling Between Objects)", value=0.0, step=1.0)
    wmc = st.number_input("WMC (Weighted Methods per Class)", value=0.0, step=1.0)
    dit = st.number_input("DIT (Depth of Inheritance Tree)", value=0.0, step=1.0)
    rfc = st.number_input("RFC (Response For a Class)", value=0.0, step=1.0)
    lcom = st.number_input("LCOM (Lack of Cohesion in Methods)", value=0.0, step=1.0)
    total_methods = st.number_input("Total Methods", value=0.0, step=1.0)
    total_fields = st.number_input("Total Fields", value=0.0, step=1.0)

with col2:
    st.subheader("🔍 نتيجة فحص الجودة")
    
    if st.button("Predict Quality 🚀"):
        if model and scaler:
            # تجهيز البيانات بنفس ترتيب التدريب (7 مقاييس)
            input_data = np.array([[cbo, wmc, dit, rfc, lcom, total_methods, total_fields]])
            
            # توحيد البيانات (Scaling)
            input_scaled = scaler.transform(input_data)
            
            # التنبؤ
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            st.divider()
            
            if prediction[0] == 1:
                st.error(f"🚨 احتمال وجود عيوب (Defect Detected)")
                st.progress(probability)
                st.write(f"نسبة الخطورة المتوقعة: {probability:.2%}")
            else:
                st.success(f"✅ الكود سليم (No Defect)")
                st.progress(1.0 - probability)
                st.write(f"نسبة الأمان المتوقعة: {1.0 - probability:.2%}")
            
            # رسم بياني بسيط للمقارنة
            metrics_df = pd.DataFrame({
                'Metric': ['CBO', 'WMC', 'DIT', 'RFC', 'LCOM'],
                'Value': [cbo, wmc, dit, rfc, lcom]
            })
            fig = px.bar(metrics_df, x='Metric', y='Value', color='Metric', title="Visual Analysis of Design Metrics")
            st.plotly_chart(fig)
        else:
            st.warning("يرجى التأكد من رفع ملفات الموديل (defect_model.pkl) أولاً.")

# 4. التذييل
st.divider()
st.caption("Graduation Project - Software Quality Prediction using MLP Neural Networks")