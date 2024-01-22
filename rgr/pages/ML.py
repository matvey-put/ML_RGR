import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import silhouette_score, calinski_harabasz_score,davies_bouldin_score,adjusted_rand_score,completeness_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

st.title('Машинное обучение')
def Metric(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # threshold = 0.5
    # y_pred = (y_pred > threshold).astype(int)

    st.write('F1-score: {:.3f}'.format(f1_score(y_test, y_pred)))
    st.write('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
    st.write('Precision: {:.3f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall: {:.3f}'.format(recall_score(y_test, y_pred)))
    st.write('ROC-AUC: {:.3f}'.format(roc_auc_score(y_test, y_pred)))
    
def MetricClaster(model, X, y):
    y_pred = model.predict(X)
    st.write("Silhouette Score:", silhouette_score(X, y_pred))
    st.write("Calinski-Harabasz Score:", calinski_harabasz_score(X, y_pred))
    st.write("Davies-Bouldin Score:", davies_bouldin_score(X, y_pred))
    st.write("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
    st.write("Completeness Score:", completeness_score(y, y_pred))

def MetricNN(model, X_test, y_test):
    y_pred = np.around(model.predict(X_test, verbose=None))
    st.write('F1-score: {:.3f}'.format(f1_score(y_test, y_pred)))
    st.write('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
    st.write('Precision: {:.3f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall: {:.3f}'.format(recall_score(y_test, y_pred)))
    st.write('ROC-AUC: {:.3f}'.format(roc_auc_score(y_test, y_pred)))

uploaded_file = st.file_uploader("Выберите файл датасета с названием `data_weatherAUS_goood.csv`", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.drop(['Unnamed: 0'], axis=1)
    st.write(df.head())
    name_model = st.selectbox("Выбирите модель", ['KNN', "Logistic Regression", "SVM", 'Bagging',
                                            'Stacking', "Gradient Boosting", "KMeans", 'Нейронная сеть'])
    
    btn = st.button("Разбить данные на выборки")
    
    if btn:
        X = df.drop(['RainTomorrow'], axis=1)
        y = df['RainTomorrow']
        scaler = StandardScaler()
 
        X_scaler= scaler.fit_transform(X)
        nm = NearMiss()
        X_balance, y_balance = nm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_balance, y_balance, test_size=0.2, random_state=0)

        if (name_model is ["KMeans",'Нейронная сеть']) :
            with open(f"./models/{name_model}.pkl", "rb") as f:
                model = pickle.load(f)
                Metric(model, X_test, y_test)
        elif name_model == 'KMeans':
            with open(f"./models/{name_model}.pkl", "rb") as f:
                model = pickle.load(f)
                X_balance, y_balance = nm.fit_resample(X_scaler, y)
                X_train, X_test, y_train, y_test = train_test_split(X_balance, y_balance, test_size=0.2, random_state=0)
                MetricClaster(model, X_balance, y_balance)
        else :
            nn = tf.keras.models.load_model('./models/NN_Classification')
            MetricNN(nn, X_test, y_test)
            # pred = nn.predict(custom_row)

    # with open("./models/KMeans.pkl", "rb") as f:
    #         KMeanS = pickle.load(f)
    #         pred = KMeanS.predict(custom_row)
    #         st.write("KMeans: " + str(int(pred[0])))
    # nn = tf.keras.models.load_model('./models/NN_Classification')
    # pred = nn.predict(custom_row)
    # st.write("nn: " + str(int(pred[0])))

        # iterImputer = IterativeImputer()
        # Copy_date = df.copy(deep = True)
        # iterImputer.fit(Copy_date.select_dtypes(include = ['float64']))

        # data_iterImputer = iterImputer.transform(Copy_date.select_dtypes(include = ['float64']))

        # data_iterImputer = pd.DataFrame(data_iterImputer, columns = Copy_date.select_dtypes(include = ['float64']).columns)

        # data_iterImputer.columns

        # df[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        #     'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
        #     'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
        #     'Temp9am', 'Temp3pm']]= data_iterImputer
        
        # df.dropna(inplace = True)

        # df["Date"] = pd.to_datetime(df["Date"], format = '%Y-%m-%d')
        # df['Year'] = df['Date'].dt.year
        # df['Month'] = df['Date'].dt.month
        # df['Day'] = df['Date'].dt.day
        # df = df.drop(['Date'], axis=1)




