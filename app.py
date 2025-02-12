import streamlit as st
import io
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.title("符号条件付き回帰分析")

# **ファイルアップロード**
uploaded_file = st.file_uploader("Excel ファイル (.xlsx) をアップロードしてください", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("アップロードしたデータの先頭5行:")
    st.write(df.head())

    # **目的変数 & 説明変数の選択**
    target_col = st.selectbox("【目的変数】", df.columns.tolist())
    feature_cols = st.multiselect("【説明変数】", df.columns.tolist())
    intercept_flag = st.radio("【切片の設定】", ["切片あり", "切片なし"])

    # **回帰分析を実行**
    if st.button("回帰を実行"):
        if target_col and feature_cols:
            X = df[feature_cols].values
            y = df[target_col].values
            n_samples, n_features = X.shape

            # **符号制約付き回帰**
            w = cp.Variable(n_features)
            b = cp.Variable() if intercept_flag == "切片あり" else 0
            residuals = y - (X @ w + b)
            objective = cp.Minimize(cp.sum_squares(residuals))
            problem = cp.Problem(objective, [])
            result = problem.solve(solver=cp.OSQP)

            coef_vals = w.value
            intercept_val = b.value if intercept_flag == "切片あり" else 0.0
            y_pred = X @ coef_vals + intercept_val

            # **評価指標**
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # **結果を表示**
            st.write("【回帰結果】")
            st.write(f"目的関数値(SSR): {problem.value:.4f}")
            st.write(f"切片: {intercept_val:.4f}")
            for col, cval in zip(feature_cols, coef_vals):
                st.write(f"{col}: {cval:.4f}")

            st.write("【評価指標】")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"R^2: {r2:.4f}")

            # **結果の Excel ダウンロード**
            output = io.BytesIO()
            df_results = pd.DataFrame({"Feature": feature_cols, "Coefficient": coef_vals})
            df_metrics = pd.DataFrame({"Metric": ["MSE", "RMSE", "MAE", "R^2"], "Value": [mse, rmse, mae, r2]})
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_results.to_excel(writer, sheet_name='Coefficients', index=False)
                df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                writer.close()
            output.seek(0)

            st.download_button(
                label="回帰結果をダウンロード",
                data=output,
                file_name="regression_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
