import streamlit as st
import pandas as pd
import joblib
import numpy as np
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Software Quality Auditor", page_icon="🔍", layout="wide")

# 2. دالة تحميل الموديل والسكيلر (النسخة الأصلية لـ 21 ميزة)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('defect_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# 3. دالة استخراج المقاييس وتجهيز الـ 21 ميزة
def extract_metrics_21(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        h = h_visit(code)
        
        # استخراج المقاييس الأساسية (أول 7-8 مقاييس)
        loc = getattr(raw, 'loc', 10)
        v_g = sum([obj.complexity for obj in cc]) if cc else 1
        ev_g = v_g * 0.6  # تقدير للتعقيد الأساسي
        iv_g = v_g * 0.4  # تقدير للتعقيد الداخلي
        n = getattr(h.total, 'length', 20)
        v = getattr(h.total, 'volume', 100)
        l = getattr(h.total, 'level', 0.1)
        d = getattr(h.total, 'difficulty', 5)
        
        # المقاييس المتبقية حتى نصل لـ 21 (نضع قيم افتراضية آمنة)
        # مصفوفة تحتوي على 21 ميزة كما يتوقعها StandardScaler
        # [loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t, lOCode, lOComment, lOBlank, locCodeAndComment, uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount]
        
        metrics_21 = [
            loc, v_g, ev_g, iv_g, n, v, l, d,
            (v/d if d != 0 else 10),  # i (intelligence)
            (d*v),                    # e (effort)
            (v/3000),                 # b (bugs)
            (d*v/18),                 # t (time)
            loc,                      # lOCode (تقدير)
            0,                        # lOComment
            2,                        # lOBlank
            0,                        # locCodeAndComment
            getattr(h.total, 'distinct_operators', 5),
            getattr(h.total, 'distinct_operands', 5),
            getattr(h.total, 'operators', 10),
            getattr(h.total, 'operands', 10),
            v_g + 1                   # branchCount
        ]
        
        return metrics_21
    except Exception as e:
        # في حال حدوث خطأ، نرسل مصفوفة افتراضية من 21 صفراً ليتجنب الموقع الانهيار
        return [0.0] * 21

# 4. واجهة المستخدم
st.title("🔍 AI Software Quality Auditor (Full 21-Feature Mode)")
st.markdown("### نظام التنبؤ بجودة البرمجيات المعتمد على داتا سيت PC1 الكاملة")
st.divider()

if not model or not scaler:
    st.error("⚠️ خطأ: تأكدي أن ملفات الموديل المرفوعة على GitHub هي النسخة التي تدربت على 21 ميزة.")
else:
    code_input = st.text_area("Source Code Input (Python):", height=250, placeholder="أدخلي الكود المراد تحليله...")

    if st.button("Analyze Code 🚀"):
        if code_input.strip():
            with st.spinner('جاري التحليل وفق 21 معياراً للجودة...'):
                metrics = extract_metrics_21(code_input)
                
                # تحويل للمصفوفة وضمان أنها 21 ميزة قبل السكيلر
                input_data = np.array([metrics])
                
                # هنا سيتم التحقق من تطابق الـ 21 ميزة
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                prob = model.predict_proba(input_scaled)[0][1]
                
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction[0]