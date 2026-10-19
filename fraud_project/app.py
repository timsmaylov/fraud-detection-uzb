from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

BASE = Path(__file__).parent
FEATURES = ['amount_uzs', 'amount_usd', 'merchant_category', 'transaction_type',
            'device_type', 'location', 'is_online', 'is_foreign_transaction']
FEATURE_LABELS = ['Сумма (UZS)', 'Сумма (USD)', 'Категория магазина', 'Тип транзакции',
                  'Тип устройства', 'Город', 'Онлайн-транзакция', 'Иностранная транзакция']
DEFAULT_RATE = 12500.0


@st.cache_resource
def load_artifacts():
    clf = joblib.load(BASE / 'fraud_model.pkl')
    scaler = joblib.load(BASE / 'scaler.pkl')
    encoders = joblib.load(BASE / 'cat_encoders.pkl')
    return clf, scaler, encoders


@st.cache_data
def load_data():
    path = BASE / 'data.csv'
    if not path.exists():
        return None
    for enc in ('utf-8-sig', 'cp1251'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return None


@st.cache_data(ttl=3600)
def get_cbu_usd_rate():
    """Текущий курс USD по данным Центрального банка Узбекистана."""
    try:
        resp = requests.get('https://cbu.uz/ru/arkhiv-kursov-valyut/json/USD/', timeout=5)
        return float(resp.json()[-1]['Rate'])
    except Exception:
        return DEFAULT_RATE


st.set_page_config(page_title='Fraud Detection Uzbekistan', layout='centered')
st.title('Fraud Detection Uzbekistan')
st.caption('Оценка вероятности мошенничества для карточной транзакции (RandomForest).')

clf, scaler, encoders = load_artifacts()
check_tab, stats_tab = st.tabs(['Проверка транзакции', 'Статистика'])

with check_tab:
    rate = st.number_input('Курс USD (UZS)', min_value=1000.0, max_value=100_000.0,
                           value=get_cbu_usd_rate(), step=10.0, format='%.2f')
    amount_uzs = st.number_input('Сумма (UZS)', min_value=0, max_value=10_000_000_000,
                                 value=100_000, step=1000)
    amount_usd = round(amount_uzs / rate, 2)
    st.caption(f'≈ {amount_usd:,.2f} USD')

    col1, col2 = st.columns(2)
    with col1:
        merchant = st.selectbox('Категория магазина', sorted(encoders['merchant_category']))
        tx_type = st.selectbox('Тип транзакции', sorted(encoders['transaction_type']))
        is_online = st.checkbox('Онлайн-транзакция')
    with col2:
        device = st.selectbox('Тип устройства', sorted(encoders['device_type']))
        location = st.selectbox('Город', sorted(encoders['location']))
        is_foreign = st.checkbox('Иностранная транзакция')

    if st.button('Проверить', type='primary'):
        row = pd.DataFrame([[
            amount_uzs, amount_usd,
            encoders['merchant_category'][merchant],
            encoders['transaction_type'][tx_type],
            encoders['device_type'][device],
            encoders['location'][location],
            int(is_online), int(is_foreign),
        ]], columns=FEATURES)
        row[['amount_uzs', 'amount_usd']] = scaler.transform(row[['amount_uzs', 'amount_usd']])
        proba = clf.predict_proba(row)[0, 1]
        if proba >= 0.5:
            st.error(f'Похоже на мошенничество. Вероятность: {proba:.1%}')
        else:
            st.success(f'Транзакция выглядит нормальной. Вероятность мошенничества: {proba:.1%}')

with stats_tab:
    df = load_data()
    if df is None:
        st.info('Файл data.csv не найден — статистика недоступна.')
    else:
        fraud = df[df['is_fraud'] == 1]
        st.metric('Транзакций в датасете', f'{len(df):,}')
        st.metric('Из них мошеннических', f'{len(fraud):,} ({len(fraud) / len(df):.2%})')

        st.subheader('Мошенничество по категориям')
        fig, ax = plt.subplots()
        fraud['merchant_category'].value_counts().head(10).sort_values().plot.barh(ax=ax, color='#c0392b')
        ax.set_xlabel('Количество')
        st.pyplot(fig)

        st.subheader('Мошенничество по часу суток')
        hours = pd.to_datetime(fraud['transaction_time'], errors='coerce').dt.hour.dropna()
        fig, ax = plt.subplots()
        ax.hist(hours, bins=np.arange(25) - 0.5, color='#c0392b')
        ax.set_xlabel('Час')
        ax.set_ylabel('Количество')
        st.pyplot(fig)

        st.subheader('Важность признаков')
        imp = pd.Series(clf.feature_importances_, index=FEATURE_LABELS).sort_values()
        fig, ax = plt.subplots()
        imp.plot.barh(ax=ax, color='#2c3e50')
        st.pyplot(fig)
