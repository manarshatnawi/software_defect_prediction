import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. تحميل البيانات الضخمة
try:
    # قراءة الملف مع اعتبار أن القيم المفقودة قد تكون علامات استفهام
    df = pd.read_csv('big_metrics.csv')
    print(f"✅ تم تحميل البيانات. العدد الأولي للأسطر: {len(df)}")
except Exception as e:
    print(f"❌ خطأ في تحميل الملف: {e}")
    exit()

# 2. معالجة القيم المفقودة والرموز الغريبة (Data Cleaning)
# استبدال علامات الاستفهام '?' بقيمة فارغة (NaN)
df = df.replace('?', np.nan)

# حذف أي سطر يحتوي على قيمة فارغة (NaN) لضمان نظافة البيانات
df = df.dropna()

# التأكد من تحويل كل البيانات إلى أرقام (float) لأن الشبكات العصبية لا تقبل النصوص
df = df.astype(float)

print(f"✅ تم تنظيف البيانات. عدد الأسطر النهائي الجاهز للتدريب: {len(df)}")

# 3. تجهيز الميزات (X) والهدف (y)
X = df.iloc[:, :-1] # جميع الأعمدة ما عدا الأخير
y = df.iloc[:, -1]  # العمود الأخير (الذي يحدد وجود خطأ أم لا)

# تحويل التارجت ليكون (1) في حال وجود أخطاء و (0) في حال كان الكود سليماً
y = y.map(lambda x: 1 if x > 0 else 0)

# 4. تقسيم البيانات (80% تدريب و 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. توحيد المقاييس (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. بناء نموذج الشبكة العصبية العميق (Deep MLP)
# استخدمنا 3 طبقات خفية (100, 50, 25 نيورون) لتناسب حجم البيانات الضخم
print("⏳ بدأ تدريب النموذج العميق على آلاف العينات، قد يستغرق لحظات...")
model = MLPClassifier(hidden_layer_sizes=(100, 50, 25), 
                      max_iter=500, 
                      solver='adam', 
                      random_state=1,
                      verbose=True) # verbose=True عشان تشوفي تقدم التدريب في الشاشة السوداء

model.fit(X_train, y_train)

# 7. التقييم والحفظ
y_pred = model.predict(X_test)
print("\n" + "="*30)
print(f"🎯 الدقة النهائية على البيانات الضخمة: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("="*30)

# حفظ "عقل" النموذج والـ Scaler
joblib.dump(model, 'defect_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 تم تحديث الملفات بنجاح: defect_model.pkl & scaler.pkl")