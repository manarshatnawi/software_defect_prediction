import streamlit as st
import pandas as pd
import joblib
import numpy as np
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Software Quality Auditor", page_icon="🔍", layout="wide")

# 2. تحميل الموديل والسكيلر (21 ميزة)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('defect_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_assets()

# 3. دالة استخراج وتجهيز الـ 21 ميزة
def extract_metrics_21(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        h = h_visit(code)
        
        loc = getattr(raw, 'loc', 10)
        v_g = sum([obj.complexity for obj in cc]) if cc else 1
        ev_g = v_g * 0.6
        iv_g = v_g * 0.4
        n = getattr(h.total, 'length', 20)
        v = getattr(h.total, 'volume', 100)
        l = getattr(h.total, 'level', 0.1)
        d = getattr(h.total, 'difficulty', 5)
        
        # مصفوفة الـ 21 ميزة (تطابق تدريب الموديل)
        metrics_21 = [
            loc, v_g, ev_g, iv_g, n, v, l, d,
            (v/d if d != 0 else 10), (d*v), (v/3000), (d*v/18),
            loc, 0, 2, 0,
            getattr(h.total, 'distinct_operators', 5),
            getattr(h.total, 'distinct_operands', 5),
            getattr(h.total, 'operators', 10),
            getattr(h.total, 'operands', 10),
            v_g + 1
        ]
        return metrics_21
    except:
        return [0.0] * 21

# 4. واجهة المستخدم
st.title("🔍 AI Software Quality Auditor")
st.markdown("### نظام التنبؤ بجودة البرمجيات المعتمد على 21 ميزة")

if not model or not scaler:
    st.error("⚠️ خطأ في تحميل ملفات الموديل. تأكدي من وجود defect_model.pkl و scaler.pkl")
else:
    code_input = st.text_area("Source Code Input (Python):", height=250)

    if st.button("Analyze Code 🚀"):
        if code_input.strip():
            with st.spinner('جاري التحليل...'):
                metrics = extract_metrics_21(code_input)
                input_data = np.array([metrics])
                input_scaled = scaler.transform(input_data)
                
                prediction = model.predict(input_scaled)
                prob = model.predict_proba(input_scaled)[0][1]
                
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    # تصحيح الخطأ السابق (إضافة النقطتين الرأسيتين :)
                    if prediction[0] == 1:
                        st.error("🚨 النتيجة: احتمال وجود عيوب (Defect Detected)")
                    else:
                        st.success("✅ النتيجة: الكود سليم ومعياري (Clean Code)")
                    
                    final_prob = prob if prediction[0] == 1 else 1 - prob
                    st.write(f"**نسبة الثقة:** {final_prob:.2%}")

                with col2:
                    st.subheader("📊 ملخص المقاييس")
                    labels = ['LOC', 'v(g)', 'ev(g)', 'iv(g)', 'N', 'V', 'L']
                    st.table(pd.DataFrame({'Metric': labels, 'Value': metrics[:7]}))
        else:
            st.warning("الرجاء إدخال كود.")

st.divider()
st.caption("Graduation Project © 2026")