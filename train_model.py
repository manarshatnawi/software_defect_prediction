import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. تحميل البيانات من الملف
try:
    # r تعني Raw string للتعامل مع المسارات في ويندوز بسهولة
    df = pd.read_csv('kc2.csv') 
    print("✅ تم تحميل البيانات بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل الملف: {e}")
    exit()

# 2. تجهيز البيانات (الميزات والهدف)
# نفترض أن العمود الأخير هو نتيجة الفحص (Defect: True/False)
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]

# تحويل التارجت لأرقام (1 للخطأ، 0 للسليم)
y = y.map(lambda x: 1 if x in [True, 'true', 't', 1, 'yes'] else 0)

# 3. تقسيم البيانات (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. توحيد المقاييس (Scaling) - خطوة جوهرية للـ Deep Learning
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. بناء وتدريب نموذج الشبكة العصبية (MLP)
print("⏳ بدأ تدريب النموذج الذكي (Deep Learning MLP)...")
# طبقتين خفيتين (64 نيورون ثم 32 نيورون)
model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                      max_iter=1000, 
                      random_state=1,
                      activation='relu',
                      solver='adam')

model.fit(X_train, y_train)

# 6. التقييم واستخراج النتائج للمناقشة
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"🎯 دقة النموذج النهائية: {acc*100:.2f}%")
print("="*30)

print("\n📋 تقرير الأداء التفصيلي (للكتاب المطبوع):")
print(classification_report(y_test, y_pred))

# 7. حفظ النموذج والـ Scaler كملفات خارجية
joblib.dump(model, 'defect_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n💾 تم حفظ 'عقل النموذج' بنجاح في الملفات التالية:")
print("- defect_model.pkl")
print("- scaler.pkl")
print("\nالآن يمكنك استخدام هذه الملفات لبناء واجهة مستخدم لاحقاً!")