import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

st.title("股票股价预测WebAPP | Prophet模型")
st.sidebar.header("上传CSV股价数据")

uploaded_file = st.sidebar.file_uploader("选择股价CSV文件", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("原始数据预览")
    st.dataframe(df.head())

    df.columns = [col.lower() for col in df.columns]

    st.subheader("列信息")
    for c in df.columns:
        st.write(f"{c}: {df[c].dtype} | 示例: {df[c].iloc[0]}")

    potential_close = None
    for c in df.columns:
        if 'close' in c.lower() or '收盘' in c:
            test_vals = pd.to_numeric(df[c], errors='coerce')
            if test_vals.notna().sum() > len(df) * 0.5:
                potential_close = c
                df[c] = test_vals
                break

    if potential_close:
        st.success(f"已自动识别收盘价列: {potential_close}")
    else:
        st.warning("未自动识别到收盘价列，请手动选择")

    potential_date = None
    for c in df.columns:
        if 'date' in c.lower() or '时间' in c:
            test_vals = pd.to_datetime(df[c], errors='coerce')
            if test_vals.notna().sum() > len(df) * 0.5:
                potential_date = c
                df[c] = test_vals
                break

    st.subheader("选择列")
    col1, col2 = st.columns(2)
    date_col = col1.selectbox("日期列", df.columns.tolist(), index=0)
    close_col = col2.selectbox("收盘价列", df.columns.tolist(), index=df.columns.tolist().index(potential_close) if potential_close else 1)

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

    df = df.dropna(subset=[date_col, close_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df[[date_col, close_col]].copy()
    df.columns = ["ds", "y"]

    st.success(f"有效数据: {len(df)} 行")
    st.write(f"收盘价范围: {df['y'].min():.2f} ~ {df['y'].max():.2f}")

    total_len = len(df)
    split_idx = total_len // 2
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    st.info(f"训练集: {split_idx}天 | 预测集: {len(test_df)}天")

    with st.spinner('训练模型中...'):
        model = Prophet(seasonality_mode="additive")
        model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = model.predict(future)

    pred_df = forecast.iloc[split_idx:][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    pred_df.columns = ["日期", "预测收盘价", "预测下限", "预测上限"]
    test_compare = test_df.copy()
    test_compare.columns = ["日期", "真实收盘价"]
    compare_result = pd.merge(test_compare, pred_df, on="日期", how="left")

    mae = mean_absolute_error(compare_result["真实收盘价"], compare_result["预测收盘价"])
    rmse = np.sqrt(mean_squared_error(compare_result["真实收盘价"], compare_result["预测收盘价"]))
    st.metric("MAE", round(mae, 2))
    st.metric("RMSE", round(rmse, 2))

    st.subheader("预测对比表")
    st.dataframe(compare_result)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df["ds"], train_df["y"], label="训练数据", color="#2E86AB")
    ax.plot(test_compare["日期"], test_compare["真实收盘价"], label="真实值", color="#A23B72")
    ax.plot(compare_result["日期"], compare_result["预测收盘价"], label="预测值", color="#F18F01", linestyle="--")
    ax.legend()
    ax.set_title("股价预测")
    st.pyplot(fig)

    csv_download = compare_result.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下载结果CSV", csv_download, "stock_predict_result.csv", "text/csv")