import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. تحميل الملفات
try:
    df1 = pd.read_csv('big_metrics.csv')
    df2 = pd.read_csv('kc2.csv')
    print("✅ تم تحميل الملفين بنجاح.")
except Exception as e:
    print(f"❌ خطأ في تحميل الملفات: {e}")
    exit()

# 2. العثور على الأعمدة المشتركة ودمج الملفين
common_columns = list(set(df1.columns) & set(df2.columns))
df_combined = pd.concat([df1[common_columns], df2[common_columns]], ignore_index=True)

# تنظيف الفراغات
df_combined = df_combined.dropna() 

# 3. إبقاء الأرقام فقط وتجهيز المدخلات والمخرجات
df_numeric = df_combined.select_dtypes(include=[np.number])

X = df_numeric.iloc[:, :-1]  # الميزات
y_raw = df_numeric.iloc[:, -1] # النتيجة الخام (التي سببت المشكلة)

# --- الحل: تحويل النتيجة إلى 0 و 1 ---
# أي قيمة أكبر من 0 تصبح 1 (Defective)، وغير ذلك 0 (Clean)
y = (y_raw > 0).astype(int)
# -----------------------------------

print(f"📊 حجم البيانات: {len(df_combined)} سطر.")
print(f"🎯 توزيع النتائج: {np.bincount(y)} (0: سليم، 1: معيوب)")

# 4. توحيد البيانات والتدريب
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🔄 جاري تدريب الموديل (الشبكة العصبية)...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
mlp.fit(X_scaled, y)

# 5. حفظ الموديل والسكيلر وقائمة الأعمدة
joblib.dump(mlp, 'defect_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features_list.pkl')

print("✨ تم التدريب وحل مشكلة التصنيفات بنجاح!")