import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="股票预测WebAPP", layout="wide")
st.title("股票时间序列预测 | LSTM模型")

def create_time_series(data, window=20):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

upload_file = st.file_uploader("上传股票CSV文件", type="csv")
if upload_file:
    df = pd.read_csv(upload_file)
    st.subheader("原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    df.columns = [col.lower() for col in df.columns]

    st.subheader("列信息")
    for c in df.columns:
        st.write(f"{c}: {df[c].dtype} | 示例: {df[c].iloc[0]}")

    date_col = None
    close_col = None

    for c in df.columns:
        if 'date' in c.lower() or '时间' in c.lower():
            test_vals = pd.to_datetime(df[c], errors='coerce')
            if test_vals.notna().sum() > len(df) * 0.5:
                date_col = c
                break

    for c in df.columns:
        if 'close' in c.lower() or '收盘' in c.lower():
            test_vals = pd.to_numeric(df[c], errors='coerce')
            if test_vals.notna().sum() > len(df) * 0.5:
                close_col = c
                break

    if not date_col or not close_col:
        st.error("无法自动识别日期列或收盘价列！请确保CSV包含日期和收盘价数据")
        st.stop()

    st.success(f"已自动识别 - 日期列: {date_col}, 收盘价列: {close_col}")

    df["date"] = pd.to_datetime(df[date_col])
    df["close"] = pd.to_numeric(df[close_col], errors='coerce')
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    st.write(f"收盘价范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    total_num = len(df)
    split_point = total_num // 2
    train_all = df.iloc[:split_point]
    test_all = df.iloc[split_point:]

    st.write(f"数据总长度：{total_num} 条 | 训练集：{split_point} 条 | 预测集：{len(test_all)} 条")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = train_all[["close"]].values
    train_scaled = scaler.fit_transform(train_close)

    window_size = 20
    X_train, y_train = create_time_series(train_scaled, window_size)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    st.info("开始训练LSTM模型，请稍等...")
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    st.success("模型训练完成！")

    full_close = df[["close"]].values
    full_scaled = scaler.transform(full_close)
    X_full, _ = create_time_series(full_scaled, window_size)
    X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)

    pred_scaled = model.predict(X_full, verbose=0)
    pred_real = scaler.inverse_transform(pred_scaled)

    result_df = df.copy()
    result_df["预测收盘价"] = np.nan
    result_df.loc[window_size:, "预测收盘价"] = pred_real.flatten()

    compare_df = result_df.iloc[split_point:].dropna()
    mse = mean_squared_error(compare_df["close"], compare_df["预测收盘价"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(compare_df["close"], compare_df["预测收盘价"])

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE平均绝对误差", round(mae, 2))
    col2.metric("MSE均方误差", round(mse, 2))
    col3.metric("RMSE均方根误差", round(rmse, 2))

    st.subheader("后半段每日【真实价格-预测价格】对比表")
    show_df = compare_df[["date", "close", "预测收盘价"]].copy()
    show_df.columns = ["日期", "真实收盘价", "预测收盘价"]
    st.dataframe(show_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train_all["date"], train_all["close"], label="前半段训练真实价格", color="#1f77b4")
    ax.plot(compare_df["date"], compare_df["close"], label="后半段真实价格", color="#2ca02c")
    ax.plot(compare_df["date"], compare_df["预测收盘价"], label="模型预测价格", color="#ff7f0e", linestyle="--")
    ax.legend(fontsize=12)
    ax.set_title("股票价格 训练数据 + 后半段预测对比", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价")
    st.pyplot(fig)

    csv_data = show_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="下载后半段预测结果CSV",
        data=csv_data,
        file_name="股票后半段预测数据.csv",
        mime="text/csv"
    )