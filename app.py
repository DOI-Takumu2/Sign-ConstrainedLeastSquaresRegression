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

# **製作者名をタイトル下に表示（右寄せ）**
st.markdown(
    """
    <div style="text-align: right; font-size: 12px; color: gray;">
        <b>土居拓務（DOI, Takumu）</b>
    </div>
    """,
    unsafe_allow_html=True
)

# **スペース（余白）を追加**
st.markdown("<br><br>", unsafe_allow_html=True)

# **ファイルアップロード**
uploaded_file = st.file_uploader("Excel ファイル (.xlsx) をアップロードしてください", type=["xlsx"])

# **ツールの引用表記を右寄せに変更**
st.markdown(
    """
    <div style="text-align: right; font-size: 10px; color: gray; margin-bottom: 5px;">
        もし本ツール使用による成果物を公表する際は、以下の例のように引用していただけると嬉しいです。<br>
        <i>If you use this tool, we would appreciate it if you could cite it as follows:</i>
    </div>
    <div style="text-align: right; font-size: 12px; font-weight: bold;">
        DOI, Takumu (2025). SignReg [Computer software]. Usage date: YYYY-MM-DD.
    </div>
    """,
    unsafe_allow_html=True
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("アップロードしたデータの先頭5行:")
    st.write(df.head())

    # **目的変数 & 説明変数の選択**
    target_col = st.selectbox("【目的変数】", df.columns.tolist())
    feature_cols = st.multiselect("【説明変数】", df.columns.tolist())
    intercept_flag = st.radio("【切片の設定】", ["切片あり", "切片なし"])

    # **各説明変数の符号制約を設定**
    sign_constraints = {col: st.selectbox(f"{col} の符号制約", ["free", "positive", "negative"]) for col in feature_cols}

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

            try:
                result = problem.solve(solver=cp.OSQP)
            except:
                st.warning("OSQP ソルバーで解が求まらなかったため、ECOS ソルバーに切り替えます。")
                result = problem.solve(solver=cp.ECOS)

            coef_vals = w.value
            intercept_val = b.value if intercept_flag == "切片あり" else 0.0

            # **エラーハンドリング**
            if coef_vals is None:
                st.error("エラー: 最適化が収束せず、回帰係数が計算されませんでした。データのスケールを確認してください。")
                st.stop()

            if intercept_flag == "切片あり" and intercept_val is None:
                st.error("エラー: 最適化が収束せず、切片が計算されませんでした。データのスケールを確認してください。")
                st.stop()

            # **形状を統一**
            if coef_vals.ndim == 1:
                coef_vals = coef_vals.reshape(-1, 1)

            # **予測値の計算**
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
            for col, cval in zip(feature_cols, coef_vals.flatten()):
                st.write(f"{col}: {cval:.4f} [{sign_constraints[col]}]")

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
