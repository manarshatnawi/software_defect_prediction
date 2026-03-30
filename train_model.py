import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. تحميل الملفين
try:
    df1 = pd.read_csv('big_metrics.csv')
    df2 = pd.read_csv('kc2.csv')
    print("✅ تم تحميل الملفين بنجاح.")
except Exception as e:
    print(f"❌ خطأ في تحميل الملفات: {e}")
    exit()

# 2. دمج الملفين (Concatenation)
# سنأخذ الأعمدة المشتركة فقط بين الملفين لضمان التوافق
common_columns = list(set(df1.columns) & set(df2.columns))
df_combined = pd.concat([df1[common_columns], df2[common_columns]], ignore_index=True)

print(f"📊 إجمالي عدد الأسطر بعد الدمج: {len(df_combined)}")

# 3. تنظيف البيانات (إبقاء الأرقام فقط)
df_numeric = df_combined.select_dtypes(include=[np.number])

# نفترض أن العمود الأخير هو النتيجة (Target) والباقي هي المقاييس (Features)
X = df_numeric.iloc[:, :-1]  # كل الأعمدة ما عدا الأخير
y = df_numeric.iloc[:, -1]   # العمود الأخير فقط

# 4. توحيد البيانات والتدريب
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🔄 جاري تدريب الموديل على البيانات المدمجة...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_scaled, y)

# 5. حفظ الموديل والسكيلر وقائمة الأعمدة (مهم جداً للموقع)
joblib.dump(mlp, 'defect_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features_list.pkl') # سنحفظ أسماء الأعمدة لنعرفها في الموقع

print("✨ تم دمج البيانات والتدريب بنجاح!")