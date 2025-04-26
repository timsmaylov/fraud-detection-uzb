import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# === 1. Загрузка данных ===
DATA_PATH = 'data.csv' 
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файл {DATA_PATH} не найден.")

# Попробуем разные кодировки для чтения файла
encodings_to_try = ['utf-8', 'cp1251', 'latin1']
for enc in encodings_to_try:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        print(f"Загружено строк: {len(df)} (encoding={enc})")
        break
    except UnicodeDecodeError:
        print(f"Не удалось прочитать файл с encoding={enc}")
else:
    raise UnicodeDecodeError(f"Не удалось прочитать файл {DATA_PATH} ни с одной из популярных кодировок: {encodings_to_try}. Сохраните файл в UTF-8 или cp1251.")

# === 2. Предобработка ===
# Удалим строки без целевой метки
if 'is_fraud' not in df.columns:
    raise ValueError("В датасете нет колонки 'is_fraud'.")
df = df[~df['is_fraud'].isnull()]

# Заполним пропуски в числовых признаках медианой
num_cols = ['amount_uzs', 'amount_usd']
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Категориальные признаки
cat_cols = ['merchant_category', 'transaction_type', 'device_type', 'location']
cat_encoders = {}
for col in cat_cols:
    if col in df.columns:
        original_values = df[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        le.fit(original_values)
        cat_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        df[col] = le.transform(original_values)

# Бинарные признаки 
bin_cols = ['is_online', 'is_foreign_transaction']
for col in bin_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Удаление неинформативных признаков !!
drop_cols = ['transaction_id', 'transaction_time', 'ip_address', 'fraud_reason']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

# === 3. Разделение на train/test ===
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Масштабирование числовых признаков
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# === 4. Обучение модели ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# === 5. Оценка качества ===
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# === 6. Сохранение модели ===
joblib.dump(clf, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# Сохраняем словари кодирования категориальных признаков
joblib.dump(cat_encoders, 'cat_encoders.pkl')
print("\nСохранено")
