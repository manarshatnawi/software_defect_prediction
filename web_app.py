import streamlit as st
import pandas as pd
import joblib
import numpy as np
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Software Quality Auditor", page_icon="🔍", layout="wide")

# 2. دالة تحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('defect_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

# 3. دالة استخراج المقاييس المحدثة (حل مشكلة HalsteadReport)
def extract_metrics(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        h = h_visit(code)
        
        # استخراج المقاييس السبعة المطلوبة للموديل
        cbo = len(raw.loc) / 5  # تقدير الارتباط بناءً على عدد الأسطر
        wmc = sum([obj.complexity for obj in cc]) if cc else 1
        dit = 1.0  # قيمة افتراضية للعمق
        rfc = h.total.bugs * 10 
        lcom = h.total.difficulty
        total_methods = len(cc) if len(cc) > 0 else 1
        
        # الحل الجديد: جمع المعاملات والمتغيرات بدلاً من استخدام نداء 'n' المباشر
        total_fields = h.total.distinct_operators + h.total.distinct_operands
        
        return [cbo, wmc, dit, rfc, lcom, total_methods, total_fields]
    except Exception as e:
        st.error(f"خطأ في تحليل بنية الكود: {e}")
        return None

# 4. واجهة المستخدم
st.title("🔍 AI Software Quality Auditor")
st.markdown("### نظام التنبؤ بجودة البرمجيات باستخدام الشبكات العصبية")
st.divider()

if not model or not scaler:
    st.error("⚠️ ملفات الموديل (defect_model.pkl) غير موجودة على السيرفر. يرجى رفعها لـ GitHub.")
else:
    code_input = st.text_area("Source Code Input (Python):", height=250, placeholder="انسخي الكود البرمجي هنا...")

    if st.button("Analyze Code 🚀"):
        if code_input.strip() == "":
            st.warning("الرجاء إدخال كود لتحليله.")
        else:
            with st.spinner('جاري تحليل الكود واستخراج المقاييس...'):
                metrics = extract_metrics(code_input)
                
                if metrics:
                    # تحويل المقاييس لمصفوفة وتوحيدها
                    input_data = np.array([metrics])
                    input_scaled = scaler.transform(input_data)
                    
                    # التنبؤ
                    prediction = model.predict(input_scaled)
                    probability = model.predict_proba(input_scaled)[0][1]
                    
                    st.divider()
                    
                    # عرض النتائج
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction[0] == 1:
                            st.error(f"🚨 النتيجة: احتمال وجود عيوب (Defect Detected)")
                        else:
                            st.success(f"✅ النتيجة: الكود سليم ومعياري (Clean Code)")
                        
                        st.write(f"**نسبة الثقة في التنبؤ:** {probability if prediction[0]==1 else 1-probability:.2%}")

                    with col2:
                        # عرض المقاييس المستخرجة في جدول
                        st.subheader("📊 الميزات المستخرجة (Features)")
                        metrics_labels = ['CBO', 'WMC', 'DIT', 'RFC', 'LCOM', 'Methods', 'Fields']
                        df_metrics = pd.DataFrame({'Metric': metrics_labels, 'Value': metrics})
                        st.table(df_metrics)

st.divider()
st.caption("Graduation Project © 2026 - Software Quality Prediction System")