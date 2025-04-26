import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Fraud Detection Uzbekistan",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("💸 Fraud Detection Uzbekistan")
st.markdown("""
<div style='background: linear-gradient(90deg, #fc466b 0%, #3f5efb 100%); padding: 10px 20px; border-radius: 10px; color: white; font-size: 20px;'>
  <b>Проверьте транзакцию на мошенничество за секунды!</b> <span style='float:right;'>🇺🇿</span>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tabs = st.tabs(["Проверка транзакции", "Статистика о мошенничестве"])

with tabs[0]:
    # Загрузка модели и scaler
    def load_model():
        clf = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return clf, scaler
    st.header("Введите параметры транзакции")

with tabs[1]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    st.header("📊 Статистика о мошенничестве")
    try:
        df = pd.read_csv("data.csv", encoding="utf-8")
    except Exception:
        df = pd.read_csv("data.csv", encoding="cp1251")


    st.markdown(f"**Всего транзакций:** {len(df):,}")
    fraud_count = df['is_fraud'].sum()
    st.markdown(f"**Мошеннических транзакций:** {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    # Категории
    st.subheader("Где чаще всего встречается мошенничество?")
    fig1, ax1 = plt.subplots()
    fraud_by_cat = df[df['is_fraud']==1]['merchant_category'].value_counts().head(10)
    sns.barplot(x=fraud_by_cat.values, y=fraud_by_cat.index, ax=ax1, palette="Reds_r")
    ax1.set_xlabel('Количество мошеннических транзакций')
    ax1.set_ylabel('Категория магазина')
    st.pyplot(fig1)
    # Время суток
    st.subheader("Распределение мошенничества по времени суток")
    if 'transaction_time' in df.columns:
        df['hour'] = pd.to_datetime(df['transaction_time'], errors='coerce').dt.hour
        fig3, ax3 = plt.subplots()
        sns.histplot(df[df['is_fraud']==1]['hour'].dropna(), bins=24, ax=ax3, color='#fc466b')
        ax3.set_xlabel('Час суток')
        ax3.set_ylabel('Количество мошеннических транзакций')
        st.pyplot(fig3)
    # Feature importance
    st.subheader("Важнейшие признаки для модели")
    try:
        clf = joblib.load('fraud_model.pkl')
        feat_names = ['Сумма (UZS)', 'Сумма (USD)', 'Категория магазина', 'Тип транзакции', 'Тип устройства', 'Город', 'Онлайн-транзакция', 'Иностранная транзакция']
        importances = clf.feature_importances_
        feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=True)
        fig4, ax4 = plt.subplots()
        feat_imp.plot.barh(ax=ax4, color='#3f5efb')
        ax4.set_xlabel('Важность')
        st.pyplot(fig4)
    except Exception:
        st.info('Feature importance недоступна для этой модели.')
    st.markdown("---")
    st.markdown("<i>Данные и статистика только для ознакомления. Для принятия решений используйте комплексный анализ.</i>", unsafe_allow_html=True)

import requests
import json

def get_cbu_usd_rate():
    try:
        url = "https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        return float(data[-1]['Rate'])
    except Exception:
        return 12500

st.session_state.setdefault('exchange_rate', get_cbu_usd_rate())

col1, col2 = st.columns([2,1])
with col1:
    exchange_rate = st.number_input(
        "Курс доллара (UZS/USD)",
        min_value=1000.0,
        max_value=100_000.0,
        value=float(st.session_state['exchange_rate']),
        step=10.0,
        format="%.2f",
        key="usd_rate_input"
    )
with col2:
    if st.button("Обновить курс по ЦБУ 🇺🇿"):
        st.session_state['exchange_rate'] = get_cbu_usd_rate()
        st.success(f"Курс обновлён: {st.session_state['exchange_rate']:.2f} UZS/USD")
    exchange_rate = st.session_state['exchange_rate']

# --- Ввод суммы с автосвязью ---
if 'amount_uzs' not in st.session_state:
    st.session_state['amount_uzs'] = 100_000
if 'amount_usd' not in st.session_state:
    st.session_state['amount_usd'] = round(st.session_state['amount_uzs'] / exchange_rate, 2)

with col2:
    amount_mode = st.radio("Ввод суммы:", ["В сумах (UZS)", "В долларах (USD)"])

if amount_mode == "В сумах (UZS)":
    amount_uzs = st.number_input("Сумма (UZS)", min_value=0, max_value=10_000_000_000, value=st.session_state['amount_uzs'], step=1000)
    amount_usd = round(amount_uzs / exchange_rate, 2)
    st.session_state['amount_uzs'] = amount_uzs
    st.session_state['amount_usd'] = amount_usd
else:
    amount_usd = st.number_input("Сумма (USD)", min_value=0.0, max_value=1_000_000.0, value=st.session_state['amount_usd'], step=0.1)
    amount_uzs = round(amount_usd * exchange_rate)
    st.session_state['amount_usd'] = amount_usd
    st.session_state['amount_uzs'] = amount_uzs

st.markdown(f"<span style='color:gray'>Эквивалент: <b>{amount_uzs:,.0f} UZS</b> ≈ <b>{amount_usd:,.2f} USD</b> (курс {exchange_rate:,.0f})</span>", unsafe_allow_html=True)

merchant_categories = ["Супермаркет", "Электроника", "Рестораны", "Аптека", "АЗС", "Прочее"]
merchant_category = st.selectbox("Категория магазина", merchant_categories)
transaction_types = ["Покупка", "Снятие", "Перевод", "Пополнение"]
transaction_type = st.selectbox("Тип транзакции", transaction_types)
device_types = ["Android", "iOS", "Web", "Другое"]
device_type = st.selectbox("Тип устройства", device_types)
import pandas as pd
import joblib
# Загружаем кодировщики категориальных признаков
cat_encoders = joblib.load('cat_encoders.pkl')
# Используем только ключи (названия) для selectbox
merchant_categories = sorted(list(cat_encoders['merchant_category'].keys()))
transaction_types = sorted(list(cat_encoders['transaction_type'].keys()))
device_types = sorted(list(cat_encoders['device_type'].keys()))
locations = sorted(list(cat_encoders['location'].keys()))
merchant_category = st.selectbox("Категория магазина", merchant_categories)
transaction_type = st.selectbox("Тип транзакции", transaction_types)
device_type = st.selectbox("Тип устройства", device_types)
location = st.selectbox("Город", locations)
is_online = st.radio("Онлайн-транзакция?", ["Да", "Нет"])
is_foreign_transaction = st.radio("Иностранная транзакция?", ["Нет", "Да"])

# Кодирование признаков (как в train_model.py)
def encode_inputs():
    merchant_dict = cat_encoders['merchant_category']
    transaction_dict = cat_encoders['transaction_type']
    device_dict = cat_encoders['device_type']
    location_dict = cat_encoders['location']
    return [
        amount_uzs,
        amount_usd,
        merchant_dict[merchant_category],
        transaction_dict[transaction_type],
        device_dict[device_type],
        location_dict[location],
        1 if is_online == "Да" else 0,
        1 if is_foreign_transaction == "Да" else 0
    ]

# Кнопка предсказания
if st.button("Проверить на мошенничество 🚦"):
    X_input = np.array([encode_inputs()])
    clf, scaler = load_model()
    # Масштабируем числовые признаки
    X_input[:, :2] = scaler.transform(X_input[:, :2])
    proba = clf.predict_proba(X_input)[0,1]
    pred = clf.predict(X_input)[0]
    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ Внимание! Транзакция может быть мошеннической!\n\nВероятность: {proba:.2%}")
        st.markdown(":rotating_light: <span style='color:red'>Рекомендуется проверить детали транзакции!</span>", unsafe_allow_html=True)
    else:
        st.success(f"✅ Транзакция не выглядит мошеннической.\n\nВероятность мошенничества: {proba:.2%}")
    st.markdown("---")
    st.markdown("<small><i>Модель обучена на реальных транзакциях, но всегда проверяйте подозрительные случаи дополнительно.</i></small>", unsafe_allow_html=True)

# Красивый футер
st.markdown("""
<hr/>
<div style='text-align:center; color: #888;'>
  <b>Проект для диплома — 2025</b> | <span style='font-size:18px;'>🤖</span> <a href='https://streamlit.io/' target='_blank'>Streamlit</a>
</div>
""", unsafe_allow_html=True)
