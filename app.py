import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# 1. تحميل النموذج المحفوظ
model = joblib.load('defect_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_bug():
    try:
        # قراءة الأرقام من الخانات (مثال: أول ميزتين فقط للتبسيط)
        val1 = float(entry1.get())
        val2 = float(entry2.get())
        # أضيفي بقية الميزات حسب عدد الأعمدة في ملفك
        
        # التجهيز للتوقع (يجب أن يكون عدد المدخلات نفس عدد أعمدة ملف CSV)
        # هنا سنضع مصفوفة وهمية بنفس طول ميزاتك الأصلية
        features = np.zeros(model.n_features_in_) 
        features[0] = val1
        features[1] = val2
        
        # عمل Scaling
        final_features = scaler.transform([features])
        
        # التوقع
        prediction = model.predict(final_features)
        
        if prediction[0] == 1:
            messagebox.showwarning("النتيجة", "⚠️ تحذير: هذا الكود يحتوي على أخطاء برمجية!")
        else:
            messagebox.showinfo("النتيجة", "✅ هذا الكود سليم وجاهز للاستخدام.")
            
    except Exception as e:
        messagebox.showerror("خطأ", "الرجاء إدخال أرقام صحيحة في الخانات.")

# 2. بناء النافذة
root = tk.Tk()
root.title("نظام التوقع الذكي للأخطاء")
root.geometry("400x300")

tk.Label(root, text="أدخل مقاييس الكود (Metrics):", font=("Arial", 12, "bold")).pack(pady=10)

tk.Label(root, text="عدد أسطر الكود (LOC):").pack()
entry1 = tk.Entry(root)
entry1.pack()

tk.Label(root, text="درجة التعقيد (Complexity):").pack()
entry2 = tk.Entry(root)
entry2.pack()

tk.Button(root, text="ابدأ فحص الكود", command=predict_bug, bg="blue", fg="white").pack(pady=20)

root.mainloop()