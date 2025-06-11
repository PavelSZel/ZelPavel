import streamlit as st
import pandas as pd
import catboost
from catboost import CatBoostClassifier, Pool
from datetime import datetime

@st.cache_resource
def load_model():
    model = catboost.CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

model = load_model()

st.title('Fraud detection interface')

with st.form(key="transaction_form"):

    creditLimit = st.number_input('Кредитный лимит', min_value=0, max_value=50000, value=250, step=50)
    availableMoney = st.number_input('Доступные деньги', min_value=0.0, max_value=50000.0, value=250.0, step=50.0)

    date = st.date_input("Дата транзакции", value=datetime.now().date(), max_value=datetime.now().date())
    time = st.time_input("Время транзакции", value=datetime.now().time(), step=60)
    transactionDateTime = datetime.combine(date, time).strftime("%Y-%m-%d %H:%M:%S")

    prev_date = st.date_input("Дата предыдущей транзакции", value=(datetime.now() - pd.Timedelta(minutes=30)).date(), max_value=datetime.now().date())
    prev_time = st.time_input("Время предыдущей транзакции", value=(datetime.now() - pd.Timedelta(minutes=30)).time(), step=60)
    previous_transaction_time = datetime.combine(prev_date, prev_time).strftime("%Y-%m-%d %H:%M:%S")

    transactionAmount = st.number_input('Сумма транзакции', min_value=1.0, max_value=50000.0, value=1.0)
    merchantName = st.text_input('Имя мерчанта', value='Unknown', max_chars=30, placeholder='Введите продавца')
    acqCountry = st.selectbox('Код эквайринга', ['Unknown', 'US', 'MEX', 'CAN', 'PR'], index=0)
    merchantCountryCode = st.selectbox('Код страны мерчанта', ['Unknown', 'US', 'MEX', 'CAN', 'PR'], index=0)
    posEntryMode = st.selectbox('Режим ввода данных', ['Unknown', '2.0', '5.0', '9.0', '80.0', '90.0'], index=0)
    posConditionCode = st.selectbox('Условия транзакции', ['Unknown', '1.0', '2.0', '5.0', '9.0', '80.0', '90.0'], index=0)

    merchantCategory = ['another', 'online_retail', 'fastfood', 'entertainment', 'food', 'online_gifts',
                        'rideshare','hotels', 'fuel', 'subscriptions', 'auto', 'health', 'personal care',
                        'airline', 'mobileapps', 'online_subscriptions', 'furniture', 'food_delivery',
                        'gym', 'cable/phone']
    merchantCategoryCode = st.selectbox('Категория мерчанта', merchantCategory, index=0)
    currentExpDate = st.text_input('Дата истечения карты', placeholder='Введите дату в формате MM/YYYY',
                                   max_chars=7)
    accountOpenDate = st.date_input("Дата открытия счета", value=datetime.now().date(), max_value=datetime.now().date())
    dateOfLastAddressChange = st.date_input("Дата последней замены адреса", value=datetime.now().date(), max_value=datetime.now().date())
    cardCVV = st.text_input('CVV карты', max_chars=3)
    enteredCVV = st.text_input('Введенный CVV', max_chars=3)
    cardLast4Digits = st.number_input('Последние 4 цифры карты', max_value=9999)
    transactionType = st.selectbox('Тип транзакции', ['Unknown', 'PURCHASE', 'REVERSAL', 'ADDRESS_VERIFICATION'], index=0)
    currentBalance = st.number_input('Текущий баланс', min_value=0.0, max_value=50000.0, value=50000.0)
    cardPresent = st.selectbox('Была ли предъявлена карта физически', [True, False])
    expirationDateKeyInMatch = st.selectbox('Совпал ли срок истечения карты с физическим', [True, False])
    avg_transaction_amount = st.number_input('Средняя сумма транзакции пользователя', min_value=1.0, max_value=50000.0, value=100.0)
    avg_transaction_hour = st.number_input('Типичное время транзакции', min_value=0.0, max_value=24.0)

    submit_button = st.form_submit_button("Обработать транзакцию")


def preprocess_data(df):
    data = df.copy()
    data['transactionDateTime'] = pd.to_datetime(data['transactionDateTime'])
    data['accountOpenDate'] = pd.to_datetime(data['accountOpenDate'])
    data['currentExpDate'] = pd.to_datetime(data['currentExpDate'])
    data['dateOfLastAddressChange'] = pd.to_datetime(data['dateOfLastAddressChange'])
    data['previous_transaction_time'] = pd.to_datetime(data['previous_transaction_time'])

    data['cardCVV'] = data['cardCVV'].astype(str)
    data['enteredCVV'] = data['enteredCVV'].astype(str)

    data['cardPresent'] = data['cardPresent'].astype(int)
    data['expirationDateKeyInMatch'] = data['expirationDateKeyInMatch'].astype(int)

    data['cvv_match'] = (data['cardCVV'] == data['enteredCVV']).astype(int)

    data['transaction_hour'] = data['transactionDateTime'].dt.hour
    data['transaction_day'] = data['transactionDateTime'].dt.day
    data['transaction_weekday'] = data['transactionDateTime'].dt.weekday
    data['night_time'] = data['transaction_hour'].apply(lambda x: 1 if 0 <= x < 6 else 0)
    data['is_weekend'] = (data['transaction_weekday'] >= 5).astype(int)

    data['second_addres_change'] = (data['dateOfLastAddressChange'] > data['accountOpenDate']).astype(int)
    data['days_since_address_change'] = (data['transactionDateTime'] - data['dateOfLastAddressChange']).dt.days

    data['diff_between_transactions'] = (data['transactionDateTime'] - data['previous_transaction_time']).dt.total_seconds() / 60
    data['amount_deviation'] = data['transactionAmount'] - data['avg_transaction_amount']
    data['hour_deviation'] = abs(data['transaction_hour'] - data['avg_transaction_hour'])

    datetime_cols = list(data.select_dtypes('datetime64[ns]').columns)
    data.drop(columns=datetime_cols, inplace=True)

    return data


category_features = ['merchantName', 'acqCountry', 'merchantCountryCode', 'posEntryMode',
                     'posConditionCode', 'merchantCategoryCode', 'transactionType']


if submit_button:
    if not currentExpDate.strip():
        st.error("Пожалуйста, введите дату истечения карты")
        st.stop()
    if transactionDateTime < previous_transaction_time:
        st.error("Время предыдущей транзакции должно быть раньше текущей")
        st.stop()

    input_data = {
        'creditLimit': creditLimit,
        'availableMoney': availableMoney,
        'transactionDateTime': transactionDateTime,
        'previous_transaction_time': previous_transaction_time,
        'transactionAmount': transactionAmount,
        'merchantName': merchantName,
        'acqCountry': acqCountry,
        'merchantCountryCode': merchantCountryCode,
        'posEntryMode': posEntryMode,
        'posConditionCode': posConditionCode,
        'merchantCategoryCode': merchantCategoryCode,
        'currentExpDate': currentExpDate,
        'accountOpenDate': accountOpenDate,
        'dateOfLastAddressChange': dateOfLastAddressChange,
        'cardCVV': cardCVV,
        'enteredCVV': enteredCVV,
        'cardLast4Digits': cardLast4Digits,
        'transactionType': transactionType,
        'currentBalance': currentBalance,
        'cardPresent': cardPresent,
        'expirationDateKeyInMatch': expirationDateKeyInMatch,
        'avg_transaction_amount': avg_transaction_amount,
        'avg_transaction_hour': avg_transaction_hour
    }

    data = pd.DataFrame([input_data])

    test = preprocess_data(data)
    pool = Pool(data=test, cat_features=category_features)

    predict = model.predict(pool)[0]

    if predict == 1:
        st.write('Мошенническая транзакция')
    else:
        st.write('Легитимная транзакция')


##### Для запуска интерфейса:   streamlit run interfaceStreamlit.py