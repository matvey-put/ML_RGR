import streamlit as st
from PIL import Image

st.title("Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")


st.header("Автор")
st.write("ФИО: Путинцев Матвей Александрович")
st.write("Группа: МО-221")
c = Image.open('.\ML\фото.jpg')
st.image(c)
st.write("2023 год")
