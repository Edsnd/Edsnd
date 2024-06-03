import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set konfigurasi halaman
st.set_page_config(
    page_title="Stock Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
    
# Fungsi untuk memuat data dari Yahoo Finance
def load_data(stock_name, start_date, end_date):
    df = yf.download(stock_name, start=start_date, end=end_date)
    # Menangani nilai kosong
    df = df.fillna(method='ffill')  # Mengisi nilai kosong dengan nilai sebelumnya
    # atau
    df = df.dropna()  # Menghapus baris dengan nilai kosong
    return df

# Memuat dan mempersiapkan data
with st.sidebar :
    stock_name = st.text_input('Kode Saham', 'BBCA.JK')
    start_date = st.date_input('Tanggal Mulai', value=pd.to_datetime('2015-01-01').date())
    end_date = st.date_input('Tanggal Akhir', value=pd.to_datetime('2024-01-01').date())
    
# Fungsi untuk mempersiapkan data
def prepare_data(df, window_size):
    # Debugging: Periksa apakah kolom 'Close' ada dalam DataFrame
    if 'Close' not in df.columns:
        raise ValueError("DataFrame tidak memiliki kolom 'Close'")
    data = df['Close'].values.reshape(-1, 1)
    # Debugging: Periksa apakah data kosong
    if len(data) == 0:
        raise ValueError("Kolom 'Close' tidak memiliki data, periksa internet anda")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    x = []
    y = []
    for i in range(window_size, len(data)):
        x.append(data[i-window_size:i, 0]) 
        y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

# Streamlit interface
st.title(f'Prediksi Harga Saham {stock_name}')
st.write('Model dilatih menggunakan data dari Yahoo Finance')
pricing_data, tuning = st.tabs(["Data dan Grafik", "Tuning Model"])

#window_size adalah rentang data yang ditampilkan dalam grafik
window_size = 90

df = load_data(stock_name, start_date, end_date)

# Opsi tuning model
with tuning:
    st.subheader('Tuning Model')

    expander = st.expander("Tuning Terbaik BBCA.JK")
    expander.write(
        "1. Epoch = 10 , Batch Size = 24/32 , Layer 1 = 114, layer 2 = 50")
    expander.write(
        "2. Epoch = 10 , Batch Size = 16 , Layer 1 = 82, layer 2 = 18")
    expander = st.expander("Tuning Terbaik GOTO.JK")
    expander.write(
        "1. Epoch = 10 , Batch Size = 24/32 , Layer 1 = 114, layer 2 = 50")
    
    epochs = st.number_input('Jumlah Epoch', min_value=10, max_value=100, value=10, step=10)
    batch_size = st.number_input('Batch Size', min_value=8, max_value=64, value=32, step=8)
    units_1 = st.number_input('Unit LSTM Lapisan 1', min_value=32, max_value=128, value=50, step=32)
    units_2 = st.number_input('Unit LSTM Lapisan 2', min_value=16, max_value=64, value=50, step=16)
    split_ratio = st.number_input('Rasio Data Latih:Uji', min_value=0.5, max_value=0.9, value=0.8, step=0.1)
    
x,y ,scaler = prepare_data(df,window_size)

# Bagi data menjadi latih dan uji
split = int(split_ratio * len(df))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

# Buat model
model = Sequential()
model.add(Bidirectional(LSTM(units=units_1, return_sequences=True, input_shape=(x_train.shape[1], 1))))
model.add(Bidirectional(LSTM(units=units_2)))

model.add(Dense(units=1))

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error',metrics = ['accuracy'])

with tuning:
    if st.button('Latih Model'):
        model.fit(x_train, y_train, epochs=int(epochs), batch_size=int(batch_size))
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Accuracy : {:.2f}".format(accuracy * 100))
        model.save('stock_prediction_model.h5')
        st.success('Model berhasil disimpan!')

#mengfilter data dalam periode
def filter_data(df, period):
    if period == '5 hari':
        return df[-5:]
    elif period == '30 hari':
        return df[-30:]
    elif period == '3 bulan':
        return df[-90:]
    elif period == '1 tahun':
        return df[-365:]
    elif period == '5 tahun':
        return df[-1825:]
    else:
        return df

with pricing_data:
    
    #Membuat radiobutton untuk melihat grafik yang ditentukan
    col1, col2, = st.columns([3,2])

    #menampilkan grafik penutupan
    data2 = df
    data2['% Change'] = df['Adj Close'] / df['Adj Close'].shift(1) - 1
    data2.dropna(inplace = True)
    with col1:
        st.write(data2)
    
    with col2:
        period = st.selectbox("Pilih jangka waktu:", ('5 hari', '30 hari', '3 bulan', '1 tahun','5 tahun'))
        filtered_data = filter_data(df, period)

    #mengload kembali model
    model = tf.keras.models.load_model('stock_prediction_model.h5')
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Menampilkan grafik harga penutupan saham
    ma100 = filtered_data.Close.rolling(100).mean()
    ma200 = filtered_data.Close.rolling(200).mean()

    # Membuat subplot dengan dua baris
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=('Harga Penutupan Saham', 'Volume Perdagangan'))
    
    # Menambahkan plot harga penutupan
    fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], mode='lines', name='Harga Penutupan'), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered_data.index, y=ma100, mode='lines', name='MA100', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered_data.index, y=ma200, mode='lines', name='MA200', line=dict(color='green')), row=1, col=1)
     
    # Menambahkan plot volume perdagangan
    fig.add_trace(go.Bar(x=filtered_data.index, y=filtered_data['Volume'], name='Volume'), row=2, col=1)
    fig.update_layout(
        title='Grafik Harga Penutupan Saham dan Volume Perdagangan',
        yaxis_title='Harga Penutupan (USD)',
        hovermode='x unified'
    )
    #menampilkan grafik
    #fig.update_yaxes(title_text='Volume', row=2, col=1)
    #with col1:
    st.plotly_chart(fig)

with pricing_data:
    # Prediksi dan evaluasi
    model = tf.keras.models.load_model('stock_prediction_model.h5')
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
        
    #membuat Evaluasi Model 
    RMSE = float(format(np.sqrt(mean_squared_error(y_test, predictions)),'.3f'))
    MAE = mean_absolute_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)
    # Menghitung akurasi prediksi
    actual_close = df['Close'][split+window_size:]
    predictions_flat = predictions.flatten()
    accuracy = []

    # Menampilkan grafik perbandingan harga penutupan asli dan prediksi menggunakan plotly
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=df.index[split+window_size:], y=df['Close'][split+window_size:], mode='lines', name='Harga Penutupan Asli', line=dict(color='blue')))
    pred_fig.add_trace(go.Scatter(x=df.index[split+window_size:], y=predictions.flatten(), mode='lines', name='Harga Prediksi', line=dict(color='red')))
    
    # Menghitung akurasi prediksi
    actual_close = df['Close'][split+window_size:]
    last_actual = actual_close.iloc[-1]
    last_prediction = predictions[-1][0]
    #accuracy = accuracy_score(last_actual,last_prediction)
    accuracy = ((last_actual - last_prediction) / last_actual) * 100
        
    # Menambahkan anotasi akurasi pada grafik
    pred_fig.add_annotation(
        x=df.index[-1],
        y=last_prediction,
        text=f"Akurasi Prediksi: {accuracy:.2f}%",
        showarrow=True,
        arrowhead=1
    )
    pred_fig.update_layout(
        title = 'Perbandingan Harga Penutupan Asli dan Prediksi',
        xaxis_title='Tanggal',
        yaxis_title='Harga Saham',
        hovermode='x unified'
    )
        
    st.plotly_chart(pred_fig)
    col1, col2,col3 = st.columns(3)
    with col1:
        st.write(f'RMSE: {RMSE:.2f}')
        st.write(f'MAE: {MAE:.2f}')
        st.write(f'MSE: {MSE:.2f}')
    with col2:
        st.write(f'Harga Penutupan Asli (Terakhir): {last_actual:.2f}')
        st.write(f'Harga Prediksi (Terakhir): {last_prediction:.2f}')
        st.write(f'Akurasi Prediksi: {accuracy:.2f}%')
                
