import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. إعدادات الصفحة والجمالية
st.set_page_config(page_title="Software Quality AI Auditor", page_icon="🛡️", layout="wide")

# 2. تحميل الموديل والسكيلر وقائمة الميزات المشتركة
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('defect_model.pkl')
        scaler = joblib.load('scaler.pkl')
        # تحميل قائمة الأعمدة التي تدرب عليها الموديل لضمان التوافق
        features = joblib.load('features_list.pkl')
        return model, scaler, features
    except Exception as e:
        st.error(f"⚠️ خطأ في تحميل ملفات الموديل: {e}")
        return None, None, None

model, scaler, features_list = load_assets()

# 3. واجهة المستخدم
st.title("🛡️ AI Software Quality & Defect Predictor")
st.markdown(f"هذا النظام مدرب على دمج بيانات **KC2** و **Big Metrics** للتنبؤ بالعيوب البرمجية باستخدام {len(features_list) if features_list else 0} مقياساً هندسياً.")
st.divider()

# تقسيم الشاشة
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 إدخال البيانات للتحليل")
    tab1, tab2 = st.tabs(["إدخال يدوي (Manual)", "رفع ملف (Upload CSV)"])
    
    input_values = []
    
    with tab1:
        if features_list:
            st.info("أدخلي القيم المستخرجة من تحليل الكود:")
            # إنشاء حقول إدخال ديناميكية بناءً على الأعمدة التي تدرب عليها الموديل
            for feat in features_list:
                val = st.number_input(f"المقياس: {feat}", value=0.0, step=0.1, key=feat)
                input_values.append(val)
        else:
            st.warning("لم يتم العثور على قائمة الميزات. تأكدي من تشغيل train_model.py أولاً.")

    with tab2:
        uploaded_file = st.file_input("ارفعي ملف CSV يحتوي على نفس المقاييس:")
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            st.write("معاينة البيانات المرفوعة:")
            st.dataframe(df_upload.head(3))

with col2:
    st.subheader("🔍 نتيجة التنبؤ بالذكاء الاصطناعي")
    
    if st.button("Run AI Audit 🚀"):
        if model and scaler and features_list:
            # تجهيز البيانات للتنبؤ
            final_input = np.array([input_values])
            
            # توحيد البيانات
            input_scaled = scaler.transform(final_input)
            
            # التنبؤ
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            st.divider()
            
            if prediction == 1:
                st.error(f"🚨 نتيجة الفحص: احتمالية وجود عيوب (Defective)")
                st.metric("مستوى الخطورة", f"{probability:.2%}")
                st.progress(probability)
            else:
                st.success(f"✅ نتيجة الفحص: الكود سليم (Clean)")
                st.metric("مستوى الأمان", f"{1-probability:.2%}")
                st.progress(1.0 - probability)
            
            # رسم بياني للميزات المدخلة
            fig_data = pd.DataFrame({'Metric': features_list, 'Value': input_values