import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import seaborn as sns


def display_ui():
    st.markdown('''
        <style>
            .main-header {
                background-color: #000099; 
                color: white; 
                padding: 15px; 
                border-radius: 0px;
                text-align: center;
            }
            .section-header {
                background-color: #00FF99; 
                color: white; 
                padding: 10px; 
                border-radius: 25px;
                text-align: center;
                margin-top: 10px;
            }
            .upload-section {
                margin-top: 10px;
                padding: 10px;
            }
            .section-content {
                margin-bottom: 20px;
                padding: 15px;
                background-color: #99FFFF;
                border-radius: 8px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }
        </style>
    ''', unsafe_allow_html=True)

    # แสดงหัวเรื่องด้วย HTML และกำหนดสีข้อความเป็นสีขาว
    st.markdown('<div class="main-header"><h1 style="color: white;">Solar Power Prediction</h1></div>', unsafe_allow_html=True)
    # แสดงภาพจาก URL
    image_url = 'https://s01.sgp1.cdn.digitaloceanspaces.com/article/105770-cmwsvkfons-1542345130.jpg'
    st.image(image_url, use_column_width=True)
    
    # Upload Section Header
    st.markdown('<div class="section-header"><h3>Data Upload</h3></div>', unsafe_allow_html=True)
    
    # Columns for Data Upload
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="upload-section"><strong>Generate Data:</strong></div>', unsafe_allow_html=True)
        dataset1 = st.file_uploader("Upload Generation Dataset (CSV)", type="csv", key="dataset1")

    with col2:
        st.markdown('<div class="upload-section"><strong>Weather Data:</strong></div>', unsafe_allow_html=True)
        dataset2 = st.file_uploader("Upload Weather Dataset (CSV)", type="csv", key="dataset2")

    # Model Selection Header
    st.markdown('<div class="section-header"><h3>Model Selection</h3></div>', unsafe_allow_html=True)
    model_choice = st.selectbox('Select Model', ('XGBoost', 'RandomForest', 'LinearRegression', 'LSTM'))

    # Button to Run Model
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    if st.button('Run Model'):
        return dataset1, dataset2, model_choice
    return dataset1, dataset2, None

def process_datasets(dataset1, dataset2, model_choice):
    if dataset1 is not None and dataset2 is not None:
        st.markdown('<div class="section-header"><h3>Data Preview</h3></div>', unsafe_allow_html=True)

        # อ่านและรวม datasets
        df_Generation_Data = pd.read_csv(io.StringIO(dataset1.getvalue().decode('utf-8')))
        df_Weather_Data = pd.read_csv(io.StringIO(dataset2.getvalue().decode('utf-8')))
        
        df_Generation_Data['DATE_TIME'] = pd.to_datetime(df_Generation_Data['DATE_TIME'], errors='coerce')
        df_Weather_Data['DATE_TIME'] = pd.to_datetime(df_Weather_Data['DATE_TIME'], errors='coerce')
        
        merged_df = pd.merge(df_Generation_Data, df_Weather_Data, on='DATE_TIME', how='inner')
        st.dataframe(merged_df)

        plt.figure(figsize=(14, 8))

        # Plot DC_POWER
        plt.subplot(3, 1, 1)  # Subplot 1: DC_POWER
        plt.plot(merged_df['DATE_TIME'], merged_df['DC_POWER'], color='blue')
        plt.title('DC Power Over Time')
        plt.xlabel('Date')
        plt.ylabel('DC Power')

        # Plot AC_POWER
        plt.subplot(3, 1, 2)  # Subplot 2: AC_POWER
        plt.plot(merged_df['DATE_TIME'], merged_df['AC_POWER'], color='green')
        plt.title('AC Power Over Time')
        plt.xlabel('Date')
        plt.ylabel('AC Power')

        # Adjust layout
        plt.tight_layout()

        st.pyplot(plt)

        # ตรวจสอบว่าคอลัมน์ใดใน DataFrame เป็นตัวเลข
        numeric_df = merged_df.select_dtypes(include=[np.number])

        # คำนวณค่า correlation matrix เฉพาะคอลัมน์ที่เป็นตัวเลข
        correlation_matrix = numeric_df.corr()

        # สร้าง heatmap ของ correlation matrix
        plt.figure(figsize=(10, 8))  # ปรับขนาดของภาพตามที่ต้องการ
        sns.heatmap(correlation_matrix, annot=True, cmap='magma', center=0, vmin=-1, vmax=1)
        plt.title('Correlation Matrix Heatmap')


        # Adjust layout
        plt.tight_layout()

        st.pyplot(plt)

        merged_df['DATE_TIME'] = pd.to_datetime(merged_df['DATE_TIME'])
        merged_df['HOUR'] = merged_df['DATE_TIME'].dt.hour
        merged_df['DAY'] = merged_df['DATE_TIME'].dt.day
        merged_df['MONTH'] = merged_df['DATE_TIME'].dt.month

        # เก็บ 'DATE_TIME' สำหรับการสร้างกราฟ
        date_time_column = merged_df['DATE_TIME']

        # ลบคอลัมน์ที่ไม่จำเป็น
        merged_df = merged_df.drop(['PLANT_ID_x', 'SOURCE_KEY_x', 'PLANT_ID_y', 'SOURCE_KEY_y', 'DATE_TIME'], axis=1)

        # กำหนดเป้าหมาย (target) และคุณลักษณะ (features)
        y = merged_df[['DC_POWER', 'AC_POWER']]
        X = merged_df.drop(['DC_POWER', 'AC_POWER'], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=42)

        # เลือกและฝึกโมเดลตามที่เลือก
        if model_choice == 'XGBoost':
            pipeline = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor(n_estimators=100, random_state=42))])
            pipeline.fit(X_train, y_train)

        elif model_choice == 'RandomForest':
            pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(n_estimators=100, random_state=42))])
            pipeline.fit(X_train, y_train)

        elif model_choice == 'LinearRegression':
            pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
            pipeline.fit(X_train, y_train)

        elif model_choice == 'LSTM':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.08, random_state=42)
            
            pipeline = Sequential()
            pipeline.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            pipeline.add(LSTM(50))
            pipeline.add(Dense(2))
            pipeline.compile(optimizer='adam', loss='mean_squared_error')
            pipeline.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)

        # ทำการทำนายผล
        y_pred = pipeline.predict(X_test)
        y_pred = pd.DataFrame(y_pred, columns=['Predicted_DC_POWER', 'Predicted_AC_POWER'])
        y_pred_total = y_pred['Predicted_DC_POWER'] * y_pred['Predicted_AC_POWER']

        # แสดงผลการทำนาย
        st.markdown('<div class="section-header"><h3>Prediction Results</h3></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Predicted DC and AC Power")
            st.dataframe(y_pred)

        with col2:
            st.write("Predicted Total Power")
            st.dataframe(y_pred_total)

        mse = mean_squared_error(y_test, y_pred)
        st.write(f'MSE: {mse}') 

        st.markdown('<div class="section-header"><h3>Graphical Analysis</h3></div>', unsafe_allow_html=True)

        # สร้างกราฟผลลัพธ์
        plt.figure(figsize=(20, 15))

        # ใช้ข้อมูลล่าสุดที่มีขนาดตรงกัน (เช่น ขนาด 2592 แถว)
        trimmed_date_time = date_time_column[-len(y_test):]  # ทำให้ `DATE_TIME` มีขนาดเท่ากับ y_test

        plt.subplot(3, 1, 1)
        plt.plot(trimmed_date_time, y_test['DC_POWER'], color='blue', label='Actual DC Power')
        plt.plot(trimmed_date_time, y_pred['Predicted_DC_POWER'], color='orange', linestyle='dashed', label='Predicted DC Power')
        plt.title('DC Power: Actual vs Predicted')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(trimmed_date_time, y_test['AC_POWER'], color='green', label='Actual AC Power')
        plt.plot(trimmed_date_time, y_pred['Predicted_AC_POWER'], color='orange', linestyle='dashed', label='Predicted AC Power')
        plt.title('AC Power: Actual vs Predicted')
        plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        plt.plot(trimmed_date_time, y_pred_total, color='orange', linestyle='dashed', label='Predicted Total Power')
        plt.title('Predicted Total Power')
        plt.legend()
        st.pyplot(plt)


# Display UI
dataset1, dataset2, model_choice = display_ui()

# Process datasets and run the model if both datasets are uploaded
if dataset1 and dataset2 and model_choice:
    process_datasets(dataset1, dataset2, model_choice)
