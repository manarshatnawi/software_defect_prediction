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

# 3. دالة استخراج المقاييس المنقحة (حل مشكلة len() ومشكلة Halstead)
def extract_metrics(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        h = h_visit(code)
        
        # تصحيح حساب CBO: raw.loc يعطي رقماً مباشراً الآن وليس قائمة
        lines_of_code = raw.loc if isinstance(raw.loc, int) else 10
        cbo = lines_of_code / 5
        
        # حساب WMC
        wmc = sum([obj.complexity for obj in cc]) if cc else 1
        dit = 1.0  # قيمة افتراضية
        rfc = h.total.bugs * 10 
        lcom = h.total.difficulty
        total_methods = len(cc) if len(cc) > 0 else 1
        
        # حساب Fields (Operators + Operands)
        total_fields = h.total.distinct_operators + h.total.distinct_operands
        
        return [cbo, wmc, dit, rfc, lcom, total_methods, total_fields]
    except Exception as e:
        st.error(f"خطأ في تحليل بنية الكود: {e}")
        return None

# 4. الواجهة البرمجية
st.title("🔍 AI Software Quality Auditor")
st.markdown("### نظام التنبؤ بجودة البرمجيات باستخدام الشبكات العصبية")
st.divider()

if not model or not scaler:
    st.error("⚠️ ملفات الموديل غير موجودة. يرجى التأكد من رفع defect_model.pkl و scaler.pkl")
else:
    code_input = st.text_area("Source Code Input (Python):", height=250, placeholder="أدخلي الكود هنا...")

    if st.button("Analyze Code 🚀"):
        if code_input.strip() == "":
            st.warning("الرجاء إدخال كود لتحليله.")
        else:
            with st.spinner('جاري التحليل...'):
                metrics = extract_metrics(code_input)
                if metrics:
                    input_scaled = scaler.transform([metrics])
                    prediction = model.predict(input_scaled)
                    prob = model.predict_proba(input_scaled)[0][1]
                    
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction[0] == 1:
                            st.error(f"🚨 النتيجة: احتمال وجود عيوب (Defect Detected)")
                        else:
                            st.success(f"✅ النتيجة: الكود سليم ومعياري (Clean Code)")
                        st.write(f"**نسبة الثقة:** {prob if prediction[0]==1 else 1-prob:.2%}")
                    with col2:
                        st.subheader("📊 الميزات المستخرجة")
                        labels = ['CBO', 'WMC', 'DIT', 'RFC', 'LCOM', 'Methods', 'Fields']
                        st.table(pd.DataFrame({'Metric': labels, 'Value': metrics}))

st.divider()
st.caption("Graduation Project © 2026")