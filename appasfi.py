import streamlit as st
import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import pickle

st.title("Aplikasi Web Datamining")
st.write("By: Dwi Asfi Fajrin (200411100121)")
st.write("======================================================================================")
st.write("""
## Pengklasifikasian Buah Mangga Harum Manis
Pembuatan aplikasi berbasis web ini digunakan untuk membantu user dalam mengklasifikasikan sebuah dataset mangga harum manis,
yang dimana nantinya user akan menginput data untuk setiap fiturnya yang ada dalam dataset buah mangga harum manis, sehingga
nanti akan diketahui termasuk dalam jenis klasifikasi apa. selain itu ada beberapa algoritma yang disediakan dalam aplikasi web ini
untuk melihat akurasi yang terbaik dari model algoritma yang telah tersedia, sehingga algoritma yang terbaik ini bisa dijadikan untuk
pengklasifikasian.
### Menu yang disediakan :
""")

# inisialisasi data 
data = pd.read_csv("data.csv")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description Data", "Preprocessing Data", "Modelling", "Implementation", "Profil"])

with tab1:

    st.subheader("Deskripsi Dataset")
    st.write("""
    Dataset di bawah ini dibuat karena belum ada dataset fisik Harumanis Mango yang tersedia secara bebas seperti berat, 
    panjang dan keliling. Dataset ini berisi 67 data pengukuran fisik tabular Mangga Harumanis (nomor klon MA 128) yang 
    dikumpulkan dari Fruit Collection Center, FAMA Perlis, Malaysia.
    """)

    st.write("""
    ### Sumber Data
    - Dataset [kaggel.com](https://www.kaggle.com/datasets/mohdnazuan/harumanis-mango-physical-measurement)
    - Github Account [github.com](https://github.com/DwiAsfi/interfaceUAS)
    """)

    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
    st.write("""
    ### Data Understanding
    Disini dijelaskan data-data yang ada dalam dataset tersebut seperti penjelasan dari setiap fitur yang
    ada dalam dataset tersebut :
    1. No : ID unik yang dimiliki oleh setiap buah mangga harum manis
    2. Weight : Berat buah mangga dalam satuan gram (gr)
    3. length : Panjang buah mangga dalam satuan centimeter (cm)
    4. Circumference : Keliling mangga dalam satuan sentimeter (cm)
    5. grade : Kelas mangga yang diklasifikasikan berdasarkan kelas A atau B
    """)

with tab2:
    st.subheader("Data Preprocessing")
    st.subheader("Data Asli")
    data = pd.read_csv("data.csv")
    st.write(data)

    proc = st.checkbox("Normalisasi")
    if proc:

        # Min_Max Normalisasi
        from sklearn.preprocessing import MinMaxScaler
        df_for_minmax_scaler=pd.DataFrame(data, columns = ['No',	'Weight',	'Length',	'Circumference'])
        df_for_minmax_scaler.to_numpy()
        scaler = MinMaxScaler()
        df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

        st.subheader("Hasil Normalisasi Min_Max")
        st.write("""Metode normalisasi min-max mengubah sebuah kumpulan data menjadi skala mulai dari 0 (min) hingga 1 (max). Data asli mengalami modifikasi linear dalam prosedur normalisasi data ini. Nilai minimum dan maksimum dari data diambil, """)
        df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ['No',	'Weight',	'Length',	'Circumference'])
        st.write(df_hasil_minmax_scaler)

        st.subheader("tampil data grade")
        df_Grade= pd.DataFrame(data, columns = ['Grade'])
        st.write(df_Grade.head())

        st.subheader("Gabung Data")
        df_new = pd.concat([df_hasil_minmax_scaler,df_Grade], axis=1)
        st.write(df_new)

        st.subheader("Drop fitur Grade")
        df_drop_site = df_new.drop(['Grade'], axis=1)
        st.write(df_drop_site)

        st.subheader("Hasil Preprocessing")
        df_new = pd.concat([df_hasil_minmax_scaler,df_Grade], axis=1)
        st.write(df_new)

with tab3:

    X=data.iloc[:,0:4].values
    y=data.iloc[:,4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    st.subheader("Pilih Model")
    model1 = st.checkbox("KNN")
    model2 = st.checkbox("Naive Bayes")
    model3 = st.checkbox("Random Forest")

    if model1:
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "KNeighborsClassifier.pkl"
        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma KNN : ",score)
    if model2:
        model = GaussianNB()
        filename = "GaussianNB.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    if model3:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForestClassifier.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Random Forest : ",score)

with tab4:
    # Min_Max Normalisasi
    from sklearn.preprocessing import MinMaxScaler
    df_for_minmax_scaler=pd.DataFrame(data, columns = ['No',	'Weight',	'Length',	'Circumference'])
    df_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

    df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ['No',	'Weight',	'Length',	'Circumference'])

    df_Grade = pd.DataFrame(data, columns = ['Grade'])

    df_new = pd.concat([df_hasil_minmax_scaler,df_Grade], axis=1)

    df_drop_site = df_new.drop(['Grade'], axis=1)

    df_new = pd.concat([df_hasil_minmax_scaler,df_Grade], axis=1)

    st.subheader("Parameter Inputan")
    No = st.number_input("Masukkan Nomor ID :")
    Weigth = st.number_input("Masukkan Berat Buah Mangga :")
    Length = st.number_input("Masukkan Lebar Buah Mangga :")
    Circumference = st.number_input("Masukkan Keliling Buah Mangga :")
    hasil = st.button("cek klasifikasi")

    # Memakai yang sudah di preprocessing
    X=df_new.iloc[:,0:4].values
    y=df_new.iloc[:,4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    if hasil:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForestClassifier.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        
        dataArray = [No, Weigth, Length, Circumference]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        st.write(f"Algoritma yang digunakan adalah = Random Forest Algorithm")
        st.success(f"Hasil Akurasi : {score}")

with tab5:

    st.subheader("Profil Mahasiswa")
    st.write("""
    - Nama : Dwi Asfi Fajrin
    - NIM : 200411100121
    - Kelas : Penambangan Data A
    - Email : dwiasfi2@gmail.com
    - Github : https://github.com/DwiAsfi/interfaceUAS
    - Instagram : @dwiasfi
    """)
