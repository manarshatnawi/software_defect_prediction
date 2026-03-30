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

# 2. دالة استخراج المقاييس مع قيم افتراضية للأكواد القصيرة
def extract_metrics_from_python(code):
    try:
        raw = analyze(code)
        cc = cc_visit(code)
        avg_cc = sum([c.complexity for c in cc]) / len(cc) if cc else 1
        h = h_visit(code)
        
        # تجميع المقاييس في قاموس
        metrics = {
            'loc': raw.loc, 'v(g)': avg_cc, 'ev(g)': avg_cc, 'iv(g)': avg_cc,
            'n': h.total.n, 'v': h.total.volume, 'l': h.total.level,
            'd': h.total.difficulty, 'i': h.total.intelligence,
            'e': h.total.effort, 'b': h.total.bugs, 't': h.total.time,
            'lOCode': raw.lloc, 'lOComment': raw.comments,
            'lOBlank': raw.blank, 'locCodeAndComment': raw.comments + raw.lloc,
        }
        return metrics
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحليل الكود: {e}")
        return None

# 3. واجهة المستخدم
st.set_page_config(page_title="AI Code Auditor", page_icon="🔍", layout="wide")
st.title("🔍 AI Software Quality Auditor")

code_area = st.text_area("Source Code Input (Python):", height=250, placeholder="انسخي الكود هنا...")

if st.button("Analyze Code 🚀"):
    if not code_area:
        st.warning("يرجى إدخال كود أولاً.")
    elif not model:
        st.error("ملفات الموديل غير موجودة على السيرفر.")
    else:
        with st.spinner('جاري التحليل...'):
            res = extract_metrics_from_python(code_area)
            if res:
                # مطابقة الميزات مع ما يطلبه الموديل بالترتيب
                input_values = [res.get(f, 0) for f in features_list]
                
                # التنبؤ
                scaled_data = scaler.transform([input_values])
                prediction = model.predict(scaled_data)[0]
                probability = model.predict_proba(scaled_data)[0][1]

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    if prediction == 1:
                        st.error(f"🚨 احتمالية وجود عيوب: {probability:.2%}")
                    else:
                        st.success(f"✅ الكود يبدو سليماً بنسبة: {1-probability:.2%}")
                    
                    st.write("📊 مقاييس الكود المستخرجة:")
                    st.json(res) # لعرض النتائج بشكل واضح للتأكد
                
                with c2:
                    # رسم بياني توضيحي
                    df_plot = pd.DataFrame({'Metric': features_list[:8], 'Value': input_values[:8]})
                    fig = px.bar(df_plot, x='Metric', y='Value', title="Top 8 Code Metrics")
                    st.plotly_chart(fig)
            else:
                st.error("فشل استخراج المقاييس. تأكدي من صحة الكود.")