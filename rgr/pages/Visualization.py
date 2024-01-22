import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

uploaded_file = st.file_uploader("Выберите файл датасета с именем `weatherAUS.csv`", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # df = df.drop(['Unnamed: 0'], axis=1)

    st.title("Датасет weatherAus")
    
    st.write("Загруженный датасет:", df)


    st.header("Тепловая карта с корреляцией между основными признаками")

    plt.figure(figsize=(10, 10))
    sns.heatmap(df.select_dtypes(include=["float"]).corr(), annot = True, fmt='.1g')
    plt.title('Тепловая карта')
    st.pyplot(plt)

    st.header("Гистограммы для основных признаков")
    
    columns = df.select_dtypes(include=["float64"]).columns

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], bins=100, kde=True)
        plt.title(f'Гистограмма для {col}')
        st.pyplot(plt)
    
    st.header("Ящик с усами для основных признаков")
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(df[col])
        plt.title(f'{col}')
        plt.xlabel('Значение')
        st.pyplot(plt)
    
    columns = df.drop(['Date'], axis=1).select_dtypes(include=["object", 'bool']).columns
    st.header("Круговая диаграмма основных категориальных признаков")
    for col in columns:
        plt.figure(figsize=(8, 8))
        df[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'{col}')
        plt.ylabel('')
        st.pyplot(plt)