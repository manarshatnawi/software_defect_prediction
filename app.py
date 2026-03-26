import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import pandas as pd
import numpy as np

# 1. تحميل النموذج والـ Scaler والتعرف على الأعمدة
try:
    model = joblib.load('defect_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # نقرأ أسماء الأعمدة من ملف البيانات الضخم (بدون العمود الأخير)
    df = pd.read_csv('big_metrics.csv')
    # تنظيف سريع للبيانات للتأكد من توافق الأعمدة
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df_columns = df.iloc[:, :-1].columns.tolist()
    
    print(f"✅ تم تحميل النموذج بنجاح. عدد الميزات المطلوب إدخالها: {len(df_columns)}")
except Exception as e:
    print(f"❌ خطأ: تأكد من وجود الملفات في نفس المجلد (big_metrics.csv, defect_model.pkl, scaler.pkl)")
    exit()

def predict_bug():
    try:
        input_values = []
        for entry in entry_fields:
            val = entry.get()
            if val == "": val = 0 
            input_values.append(float(val))
        
        # عمل Scaling للمدخلات
        final_features = scaler.transform([input_values])
        
        # الحصول على "الاحتمالية" بدلاً من النتيجة النهائية 0 أو 1
        probabilities = model.predict_proba(final_features)
        bug_probability = probabilities[0][1] # احتمالية وجود خطأ
        
        print(f"Probability of bug: {bug_probability:.4f}") # سيظهر لك في الشاشة السوداء
        
        # تقليل العتبة (Threshold) ليكون النموذج حساساً جداً
        # إذا كانت الاحتمالية أكثر من 15%، نعتبره خطأ (لأغراض العرض)
        if bug_probability > 0.15:
            messagebox.showwarning("نتيجة الفحص", f"⚠️ تحذير: احتمالية وجود خطأ هي {bug_probability*100:.1f}%\nهذا الكود يحتاج مراجعة!")
        else:
            messagebox.showinfo("النتيجة", f"✅ الكود سليم.\nنسبة الشك في وجود خطأ: {bug_probability*100:.1f}%")
            
    except Exception as e:
        messagebox.showerror("خطأ", f"حدث خطأ: {e}")

# 2. بناء الواجهة الرسومية
root = tk.Tk()
root.title("نظام التنبؤ الذكي للأخطاء البرمجية - JM1 Dataset")
root.geometry("550x700")

# إعداد منطقة التمرير (Scrollbar)
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame, bg="#f0f0f0")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = tk.Frame(canvas, bg="#f0f0f0")
canvas.create_window((0,0), window=second_frame, anchor="nw")

# العنوان
tk.Label(second_frame, text="فحص جودة البرمجيات (Deep Learning)", 
         font=("Arial", 14, "bold"), fg="#2c3e50", bg="#f0f0f0").grid(row=0, column=0, columnspan=2, pady=20)

# إنشاء 21 خانة إدخال تلقائياً بناءً على أسماء المقاييس في ناسا
entry_fields = []
for i, col_name in enumerate(df_columns):
    tk.Label(second_frame, text=f"{col_name}:", font=("Arial", 10), bg="#f0f0f0").grid(row=i+1, column=0, padx=20, pady=5, sticky="e")
    entry = tk.Entry(second_frame, width=25, font=("Arial", 10))
    entry.grid(row=i+1, column=1, padx=20, pady=5)
    entry_fields.append(entry)

# زر التوقع
btn_predict = tk.Button(second_frame, text="إجراء الفحص والتوقع", command=predict_bug, 
                        bg="#27ae60", fg="white", font=("Arial", 11, "bold"), width=20, height=2)
btn_predict.grid(row=len(df_columns)+1, column=0, columnspan=2, pady=30)

root.mainloop()