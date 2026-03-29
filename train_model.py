import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. تحميل البيانات
dataset_path = 'pc1.csv' 
df = pd.read_csv(dataset_path)

# اطبعي الأسماء لنتأكد منها (ستظهر لك في الـ CMD)
print("الأعمدة الموجودة في ملفك هي:")
print(df.columns.tolist())

# 2. اختيار أول 7 أعمدة (المقاييس) والعمود الأخير (الهدف)
# هذه الطريقة تتخطى مشكلة اختلاف الأسماء (KeyError)
X = df.iloc[:, :7]  # يأخذ أول 7 أعمدة في الملف
y = df.iloc[:, -1]   # يأخذ آخر عمود في الملف (الذي يحتوي على 0 أو 1)

print(f"✅ تم اختيار {X.shape[1]} مقاييس للتدريب.")

# 3. توحيد البيانات (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. بناء وتدريب الشبكة العصبية
print("🔄 جاري التدريب...")
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
mlp.fit(X_scaled, y)

# 5. حفظ الموديل الجديد
joblib.dump(mlp, 'defect_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✨ مبروك! تم إنتاج الموديل الجديد بنجاح.")
print("الآن ارفعي ملف 'defect_model.pkl' و 'scaler.pkl' إلى GitHub.")