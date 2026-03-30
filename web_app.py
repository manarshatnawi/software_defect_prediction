import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

# 1. تحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('defect_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features_list.pkl')
        return model, scaler, features
    except:
        return None, None, None

model, scaler, features_list = load_assets()

# 2. دالة استخراج المقاييس من الكود
def extract_metrics_from_python(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        avg_cc = sum([c.complexity for c in cc]) / len(cc) if cc else 1
        h = h_visit(code)
        
        # تحويل النتائج لقاموس (Dictionary)
        # ملاحظة: سنحاول مطابقة أسماء المقاييس مع ما تدرب عليه الموديل
        extracted = {
            'loc': raw.loc,
            'v(g)': avg_cc,
            'ev(g)': avg_cc,
            'iv(g)': avg_cc,
            'n': h.total.n,
            'v': h.total.volume,
            'l': h.total.level,
            'd': h.total.difficulty,
            'i': h.total.intelligence,
            'e': h.total.effort,
            'b': h.total.bugs,
            't': h.total.time,
            'lOCode': raw.lloc,
            'lOComment': raw.comments,
            'lOBlank': raw.blank,
            'locCodeAndComment': raw.comments + raw.lloc,
        }
        return extracted
    except:
        return None

# 3. واجهة المستخدم
st.set_page_config(page_title="AI Code Auditor", page_icon="🔍")
st.title("🔍 AI Code Quality Auditor")
st.markdown("قومي بلصق كود بايثون هنا، وسيقوم النظام باستخراج المقاييس والتنبؤ بالعيوب تلقائياً!")

code_area = st.text_area("Source Code Input:", height=300, placeholder="Paste your python code here...")

if st.button("Analyze Code 🚀"):
    if code_area and model:
        raw_metrics = extract_metrics_from_python(code_area)
        if raw_metrics:
            # ترتيب المقاييس لتمشي مع ترتيب الموديل
            input_data = []
            for feat in features_list:
                input_data.append(raw_metrics.get(feat, 0)) # إذا لم يجد المقياس يضع 0
            
            # التنبؤ
            input_scaled = scaler.transform([input_data])
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"🚨 النتيجة: احتمالية وجود عيوب عالية ({prob:.2%})")
                else:
                    st.success(f"✅ النتيجة: الكود سليم ومعياري ({1-prob:.2%})")
                
                st.write("📊 المقاييس المستخرجة:")
                st.dataframe(pd.Series(raw_metrics).head(10)) # عرض أول 10 مقاييس للتبسيط
                
            with col2:
                # رسم بياني سريع
                fig = px.bar(x=features_list[:10], y=input_data[:10], labels={'x':'Metric', 'y':'Value'})
                st.plotly_chart(fig)
        else:
            st.error("تعذر تحليل الكود. تأكدي أنه كود بايثون صحيح.")