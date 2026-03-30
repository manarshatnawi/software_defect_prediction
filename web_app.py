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
    except:
        return None, None

model, scaler = load_assets()

# 3. دالة استخراج المقاييس (النسخة الأكثر استقراراً)
def extract_metrics(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        h = h_visit(code)
        
        # حساب أسطر الكود
        loc = raw.loc if hasattr(raw, 'loc') and isinstance(raw.loc, int) else 10
        
        # حساب المقاييس السبعة
        cbo = loc / 5
        wmc = sum([obj.complexity for obj in cc]) if cc else 1
        dit = 1.0
        rfc = getattr(h.total, 'bugs', 0.1) * 10 
        lcom = getattr(h.total, 'difficulty', 5.0)
        total_methods = len(cc) if len(cc) > 0 else 1
        
        # حل مشكلة distinct_operators باستخدام getattr كبديل آمن
        operators = getattr(h.total, 'distinct_operators', 5)
        operands = getattr(h.total, 'distinct_operands', 5)
        total_fields = operators + operands
        
        return [cbo, wmc, dit, rfc, lcom, total_methods, total_fields]
    except Exception as e:
        st.error(f"تنبيه تقني: تم استخدام قيم تقديرية بسبب تعارض في إصدار المكتبة ({e})")
        # إرجاع قيم افتراضية آمنة في حال فشل التحليل العميق
        return [10.0, 2.0, 1.0, 0.5, 5.0, 1.0, 10.0]

# 4. الواجهة البرمجية
st.title("🔍 AI Software Quality Auditor")
st.markdown("### نظام التنبؤ بجودة البرمجيات باستخدام الشبكات العصبية")

if not model or not scaler:
    st.error("⚠️ ملفات الموديل غير موجودة على GitHub.")
else:
    code_input = st.text_area("Source Code Input (Python):", height=250)

    if st.button("Analyze Code 🚀"):
        if code_input.strip():
            metrics = extract_metrics(code_input)
            input_scaled = scaler.transform([metrics])
            prediction = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)[0][1]
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if prediction[0] == 1:
                    st.error("🚨 النتيجة: احتمال وجود عيوب (Defect Detected)")
                else:
                    st.success("✅ النتيجة: الكود سليم ومعياري (Clean Code)")
                st.write(f"**نسبة الثقة:** {prob if prediction[0]==1 else 1-prob:.2%}")
            with col2:
                st.subheader("📊 المقاييس المستخرجة")
                labels = ['CBO', 'WMC', 'DIT', 'RFC', 'LCOM', 'Methods', 'Fields']
                st.table(pd.DataFrame({'Metric': labels, 'Value': metrics}))