import streamlit as st
import io
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# 必要なライブラリをインストール（暫定対応）
os.system("pip install xlsxwriter")

# **ツールタイトル**
st.title("符号条件付き回帰分析")

# **製作者名をタイトル下に表示**
st.markdown(
    """
    <div style="text-align: right; font-size: 12px; color: gray;">
        <b>土居拓務（DOI, Takumu）</b>
    </div>
    """,
    unsafe_allow_html=True
)

# **ツールの引用表記を右下に小さく表示**
st.markdown(
    """
    <div style="text-align: right; font-size: 12px; color: gray; margin-top: 20px;">
        もし本ツール使用による成果物を公表する際は、以下の例のように引用していただけると嬉しいです。<br>
        <i>If you use this tool, we would appreciate it if you could cite it as follows:</i><br><br>
        <b>DOI, Takumu (2025). SignReg [Computer software]. Usage date: YYYY-MM-DD.</b>
    </div>
    """,
    unsafe_allow_html=True
)

# **スペース（余白）を追加**
st.markdown("<br><br>", unsafe_allow_html=True)
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

    # **各説明変数の符号制約を設定**
    sign_constraints = {}
    for col in feature_cols:
        sign_constraints[col] = st.selectbox(f"{col} の符号制約", ["free", "positive", "negative"])

    # **t値の計算関数**
    def calc_t_values(X, y, w_fit, intercept_flag):
        """
        OLSの公式を用いた近似的なt値を計算する。
        - X, y: データ (numpy.ndarray)
        - w_fit: フィットされた係数ベクトル
                 （切片がある場合は w_fit=[b, w1, w2, ...] の形を想定）
        - intercept_flag: True → 切片あり, False → 切片なし
        """
        n_samples, n_features_raw = X.shape
        
        # 切片ありの場合は X を拡張 (先頭列=1)
        if intercept_flag:
            X_aug = np.hstack([np.ones((n_samples, 1)), X])
            k = n_features_raw + 1  # パラメータ数
        else:
            X_aug = X
            k = n_features_raw

        # 残差を計算
        y_pred = X_aug @ w_fit
        residuals = y - y_pred
        sse = np.sum(residuals**2)
        
        # 自由度調整 (暫定的に n_samples - k で割る)
        mse = sse / (n_samples - k)

        # (X^T X)^{-1} を計算
        xtx_inv = np.linalg.inv(X_aug.T @ X_aug)
        
        # 各パラメータの標準誤差 = mse * 対角成分
        var_params = mse * np.diag(xtx_inv)
        se_params = np.sqrt(var_params)
        
        t_values = w_fit / se_params
        return t_values

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
            constraints = []

            # **符号制約の適用**
            for i, col in enumerate(feature_cols):
                if sign_constraints[col] == "positive":
                    constraints.append(w[i] >= 0)
                elif sign_constraints[col] == "negative":
                    constraints.append(w[i] <= 0)

            objective = cp.Minimize(cp.sum_squares(residuals))
            problem = cp.Problem(objective, constraints)
            result = problem.solve(solver=cp.OSQP)

            coef_vals = w.value
            intercept_val = b.value if intercept_flag == "切片あり" else 0.0
            y_pred = X @ coef_vals + intercept_val

            # **t値の計算**
            w_fit_ols_like = np.concatenate([[intercept_val], coef_vals]) if intercept_flag == "切片あり" else coef_vals
            t_values = calc_t_values(X, y, w_fit_ols_like, intercept_flag)

            # **評価指標**
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # **結果を表示**
            st.write("【回帰結果】")
            st.write(f"目的関数値(SSR): {problem.value:.4f}")
            st.write(f"切片: {intercept_val:.4f} (t値: {t_values[0]:.4f})" if intercept_flag == "切片あり" else "切片なし")
            for col, cval, tval in zip(feature_cols, coef_vals, t_values[1:] if intercept_flag == "切片あり" else t_values):
                st.write(f"{col}: {cval:.4f} (t値: {tval:.4f}) [{sign_constraints[col]}]")

            st.write("【評価指標】")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"R^2: {r2:.4f}")

            # **結果の Excel ダウンロード**
            output = io.BytesIO()
            df_results = pd.DataFrame({
                "Feature": ["Intercept"] + feature_cols if intercept_flag == "切片あり" else feature_cols,
                "Coefficient": [intercept_val] + list(coef_vals) if intercept_flag == "切片あり" else list(coef_vals),
                "t-value": list(t_values),
                "Sign Constraint": ["(None)"] + [sign_constraints[col] for col in feature_cols] if intercept_flag == "切片あり" else [sign_constraints[col] for col in feature_cols]
            })
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
