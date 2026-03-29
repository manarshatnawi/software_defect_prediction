import streamlit as st
import pandas as pd
import joblib
import numpy as np
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

# 1. إعدادات الصفحة
st.set_config(page_title="AI Code Auditor", page_icon="🔍")

# 2. تحميل الموديل والسكيلر
@st.cache_resource
def load_model():
    try:
        model = joblib.load('defect_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# 3. دالة استخراج المقاييس من الكود (تلقائياً)
def extract_metrics(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        h = h_visit(code)
        
        # ترتيب المقاييس السبعة كما تعلمها الموديل في الشاشة السوداء
        # [cbo, wmc, dit, rfc, lcom, totalMethods, totalFields]
        # ملاحظة: سنستخدم قيم تقريبية من Radon لتناسب الـ OO Metrics
        cbo = len(raw.loc) / 10  # تقديري للارتباط
        wmc = sum([c.complexity for c in cc]) if cc else 1
        dit = 1.0 # القيمة الافتراضية للعمق
        rfc = h.total.bugs * 10 
        lcom = h.total.difficulty
        totalMethods = len(cc)
        totalFields = h.total.n / 5
        
        return [cbo, wmc, dit, rfc, lcom, totalMethods, totalFields]
    except Exception as e:
        return None

# 4. الواجهة البرمجية
st.title("🔍 AI Software Quality Auditor")
st.write("ضع الكود الخاص بك هنا، وسيقوم الذكاء الاصطناعي بتحليله فوراً!")

code_input = st.text_area("أدخلي كود بايثون هنا:", height=300, placeholder="def example_function()...")

if st.button("Start AI Audit 🚀"):
    if code_input:
        metrics = extract_metrics(code_input)
        if metrics and model:
            # التنبؤ
            input_scaled = scaler.transform([metrics])
            prediction = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)[0][1]
            
            st.divider()
            if prediction[0] == 1:
                st.error(f"🚨 تحذير: تم اكتشاف احتمالية وجود عيوب (الثقة: {prob:.2%})")
                st.info("نصيحة: الكود يحتوي على تعقيد عالي، يفضل إعادة هيكلته.")
            else:
                st.success(f"✅ ممتاز: الكود سليم ومعياري (الثقة: {1-prob:.2%})")
            
            # عرض المقاييس التي استخرجها البرنامج
            with st.expander("أظهر المقاييس المستخرجة (Technical Details)"):
                labels = ['CBO', 'WMC', 'DIT', 'RFC', 'LCOM', 'Methods', 'Fields']
                st.table(pd.DataFrame({'Metric': labels, 'Value': metrics}))
        else:
            st.error("تأكدي من كتابة كود بايثون صحيح لكي نتمكن من تحليله.")