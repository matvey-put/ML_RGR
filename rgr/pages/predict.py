import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import tensorflow as tf
import pickle

df = pd.read_csv('./data/data_weatherAUS_good.csv')
df = df.drop(['Unnamed: 0', 'Date', 'Day','Year', 'Month', 'RainTomorrow'], axis=1)

st.title("Будет ли завтра дождь?")
location = st.selectbox("Общее название местоположения метеостанции:", df['Location'].unique())

max_temp = st.number_input("Максимальная температура за день (в градусах цельсия):", min_value=-10, max_value=60, value=23)

min_temp = st.number_input("Минимальная температура за день (в градусах цельсия):", min_value=-10, max_value=60, value=13)

rain_fall = st.number_input("Количество осадков, выпавших за день, в мм:", min_value=0, max_value=400, value=5)

st.write('Испарение (мм) за 24 часа ')
evaporation = st.slider('Испарение (мм) за 24 часа ', min_value=0, max_value=145, 
                        value=6,)
sun_shine = st.number_input("Количество часов яркого солнечного света в течение дня:", min_value=0, max_value=24, value=8)
wind_gust_dir = st.selectbox("Направление самого сильного ветра за сутки ", df['WindGustDir'].unique())
    
max_wind = st.number_input("Максимальная скорость ветра в км/ч:", min_value=6, max_value=140, value=40)

wind_dir_9am = st.selectbox("Направление ветра в 9 утра:", df['WindDir9am'].unique())

wind_dir_3pm = st.selectbox("Направление ветра в 3 часа дня:", df['WindDir3pm'].unique())

wind_speed_9am = st.number_input('Скорость ветра (км/ч), в среднем составляющая более 10 метров в минут, до 9 утра:',
                                 min_value=0, max_value=130, value=40, step = 1)

wind_speed_3pm = st.number_input('Скорость ветра (км/ч), в среднем составляющая более 10 метров в минут, до 15:00:', 
                                 min_value=0, max_value=130, value=40, step = 1)

humidity_9am = st.slider("Влажность (в процентах) в 9 утра:", min_value=0.0, max_value=100.0, value=50.0, step = 0.5, format='%f')

humidity_3pm = st.slider("Влажность (в процентах) в 3 часа дня:", min_value=0.0, max_value=100.0, value=50.0, step = 0.5, format='%f')

st.write('`Pressure9am` — Атмосферное давление (гпа)  в 9 утра')  
st.write('`Pressure3pm` — Атмосферное давление (гпа) в 3 часа дня') 
pressure_9am = st.number_input('Атмосферное давление (гпа)  в 9 утра',
                                 min_value=979, max_value=1041, value=1020, step = 1)
pressure_3pm = st.number_input('Атмосферное давление (гпа) в 3 часа дня',
                                 min_value=979, max_value=1040, value=1015, step = 1)

st.write('Доля неба, закрытого облаками в 9 утра. Это измеряется в `октах`')
cloud_9am = st.slider('Значение 0 указывает на полностью ясное небо, в то время как значение 8 указывает на то, что оно полностью затянуто облаками',
                       min_value=0, max_value=8, value=4)
cloud_3pm = st.slider('Часть неба, закрытая облаками (в "октасе": восьмые доли) в 15:00. Описание значений смотрите выше',
                       min_value=0, max_value=8, value=4)
 
temp_9am = st.number_input('Температура (градусы по Цельсию) в 9 утра ',
                                 min_value=-8, max_value=40, value=40, step = 1)
temp_3pm =st.number_input('Температура (градусы по Цельсию) в 15:00',
                                 min_value=-6, max_value=46, value=40, step = 1)
rain_today = st.checkbox('Был ли дождь сегодня?',value=False)

if rain_today:
    rain_today = 1
else: rain_today = 0
st.write(df.columns)
custom_df = pd.DataFrame([[location, min_temp, max_temp, rain_fall, evaporation, 
                    sun_shine, wind_gust_dir, max_wind, wind_dir_9am, wind_dir_3pm, 
                    wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, 
                    pressure_3pm, cloud_9am, cloud_3pm, temp_9am, temp_3pm, rain_today
]], columns =df.columns) 

st.header('Пользовательские данные')

df = pd.concat([custom_df, df], axis=0, ignore_index=True)
st.write(custom_df)
categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
categorical_encoder = OneHotEncoder(sparse_output=False)

ct = ColumnTransformer(transformers=[
     ('cat', categorical_encoder, categorical_features)
     ])
ct.set_output(transform='pandas')
encoded_features = ct.fit_transform(df)
scaler = StandardScaler()
 
df_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int32']).to_numpy())
df_scaled = pd.DataFrame(df_scaled , columns=df.select_dtypes(include=['float64', 'int32']).columns)

if rain_today== 1: rain_df = pd.DataFrame([[0,1]], columns =['cat__RainToday_0', "cat__RainToday_1"]) 
else: rain_df = pd.DataFrame([[1,0]], columns =['cat__RainToday_0', "cat__RainToday_1"]) 

data_concat = pd.concat([df.select_dtypes(include=['float64', 'int32']), 
                         encoded_features, rain_df], axis=1)

custom_row= data_concat.loc[:0]

st.write(custom_row)
btn = st.button("Предсказать")
if btn:

    st.write('Если `1`, то дождь будет. Если же `0`, то дождя не будет')

    with open("./models/KNN.pkl", "rb") as f:
            KNN = pickle.load(f)
            pred = KNN.predict(custom_row)
            st.write("KNN: " + str(int(pred[0])))

    with open("./models/Logistic Regression.pkl", "rb") as f:
            LR = pickle.load(f)
            pred = LR.predict(custom_row)
            st.write("Logistic Regression: " + str(int(pred[0])))

    with open("./models/SVM.pkl", "rb") as f:
            SVM = pickle.load(f)
            pred = SVM.predict(custom_row)
            st.write("SVM: " + str(int(pred[0])))

    with open("./models/Bagging.pkl", "rb") as f:
            bg = pickle.load(f)
            pred = bg.predict(custom_row)
            st.write("Bagging: " + str(int(pred[0])))

    with open("./models/Stacking.pkl", "rb") as f:
            stack = pickle.load(f)
            pred = stack.predict(custom_row)
            st.write("Stacking: " + str(int(pred[0])))

    with open("./models/Gradient Boosting.pkl", "rb") as f:
            gb = pickle.load(f)
            pred = gb.predict(custom_row)
            st.write("Gradient Boosting: " + str(int(pred[0])))

    with open("./models/KMeans.pkl", "rb") as f:
            KMeanS = pickle.load(f)
            pred = KMeanS.predict(custom_row)
            st.write("KMeans: " + str(int(pred[0])))
    nn = tf.keras.models.load_model('./models/NN_Classification')
    pred = nn.predict(custom_row)
    st.write("nn: " + str(int(pred[0])))



    