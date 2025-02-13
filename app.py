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

            # ===== 符号制約付き回帰を定義 =====
            # 変数定義
            w = cp.Variable(n_features)
            b = cp.Variable() if intercept_flag == "切片あり" else 0
            residuals = y - (X @ w + b)

            # 追加の制約リスト
            constraints = []
            # 符号制約の適用
            for i, col in enumerate(feature_cols):
                if sign_constraints[col] == "positive":
                    constraints.append(w[i] >= 0)
                elif sign_constraints[col] == "negative":
                    constraints.append(w[i] <= 0)

            # 最小化する目的関数 (SSR = Σ(residual^2))
            objective = cp.Minimize(cp.sum_squares(residuals))
            problem = cp.Problem(objective, constraints)

            # ソルバー実行
            try:
                result = problem.solve(solver=cp.OSQP)
            except:
                st.warning("OSQP ソルバーで解が求まらなかったため、ECOS ソルバーに切り替えます。")
                result = problem.solve(solver=cp.ECOS)

            coef_vals = w.value
            intercept_val = b.value if intercept_flag == "切片あり" else 0.0

            # **エラーハンドリング**（解が得られなかった場合）
            if coef_vals is None:
                st.error("エラー: 最適化が収束せず、回帰係数が計算されませんでした。データのスケールを確認してください。")
                st.stop()

            if intercept_flag == "切片あり" and intercept_val is None:
                st.error("エラー: 最適化が収束せず、切片が計算されませんでした。データのスケールを確認してください。")
                st.stop()

            # **次元を統一（縦ベクトル化）**
            if coef_vals.ndim == 1:
                coef_vals = coef_vals.reshape(-1, 1)

            # ===== 予測・評価 =====
            # 予測値
            y_pred = X @ coef_vals + intercept_val

            # 評価指標
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # ===== OLS に基づく近似的な t 値の計算 =====
            """
            符号制約付き回帰の場合、理論的に厳密な標準誤差・t値を計算するには
            より高度な手法（ブートストラップ等）が必要となる場合がある。
            ここでは便宜的にOLSベースの近似計算を行い、あくまで参考値として出力する。
            """
            t_values = []
            try:
                # デザイン行列の作成（切片ありなら列を追加）
                if intercept_flag == "切片あり":
                    X_design = np.hstack([X, np.ones((n_samples, 1))])  # 最後の列が切片用
                    # 回帰係数 + 切片をまとめたベクトル
                    w_full = np.vstack([coef_vals, intercept_val])
                else:
                    X_design = X
                    w_full = coef_vals

                # 予測値
                y_pred_design = X_design @ w_full

                # 残差
                residual = y - y_pred_design.flatten()
                df_error = n_samples - X_design.shape[1]

                # SSE (residual sum of squares)
                SSE = np.sum(residual**2)
                # 残差分散
                s2 = SSE / df_error if df_error > 0 else np.nan

                # (X^T X) の逆行列
                xtx_inv = np.linalg.inv(X_design.T @ X_design)

                # 分散共分散行列の対角成分を取り出して標準誤差を計算
                var_beta = s2 * np.diag(xtx_inv)
                se_beta = np.sqrt(var_beta)

                # t 値 = 係数 / 標準誤差
                # w_full の順番は [coef1, coef2, ..., coefN, intercept(オプション)]
                w_full_flat = w_full.flatten()
                for i in range(len(w_full_flat)):
                    if se_beta[i] == 0:
                        t_values.append(np.nan)
                    else:
                        t_values.append(w_full_flat[i] / se_beta[i])

            except np.linalg.LinAlgError:
                st.warning("警告: (X^T X) の逆行列が計算できませんでした。t値は計算されません。")
                t_values = [np.nan]*(n_features + (1 if intercept_flag == "切片あり" else 0))

            # ===== 結果の表示 =====
            st.write("【回帰結果】")
            st.write(f"目的関数値(SSR): {problem.value:.4f}")
            if intercept_flag == "切片あり":
                st.write(f"切片: {intercept_val:.4f}  (t={t_values[-1]:.4f} if not NaN)")
            for i, col in enumerate(feature_cols):
                # 切片ありの場合、t_valuesの最後が切片なので説明変数は先頭～末から-1まで
                t_val_current = t_values[i] if intercept_flag == "切片なし" else t_values[i]
                st.write(f"{col}: {coef_vals[i, 0]:.4f} [{sign_constraints[col]}] (t={t_val_current:.4f} if not NaN)")

            st.write("【評価指標】")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"R^2: {r2:.4f}")

            # ===== ダウンロード用 Excel を作成 =====
            output = io.BytesIO()

            # ダウンロード用の結果DataFrameを作成
            # t値の順番は [coef_1, coef_2, ..., coef_n, intercept(任意)]
            if intercept_flag == "切片あり":
                # インターセプト + 各係数
                feature_list = ["Intercept"] + feature_cols
                coef_list = [intercept_val] + list(coef_vals.flatten())
                # t 値も同様の順番
                t_value_list = [t_values[-1]] + t_values[0:-1]
                sign_list = ["(None)"] + [sign_constraints[col] for col in feature_cols]
            else:
                feature_list = feature_cols
                coef_list = list(coef_vals.flatten())
                t_value_list = t_values
                sign_list = [sign_constraints[col] for col in feature_cols]

            df_results = pd.DataFrame({
                "Feature": feature_list,
                "Coefficient": coef_list,
                "t-value": t_value_list,
                "Sign Constraint": sign_list
            })

            df_metrics = pd.DataFrame({
                "Metric": ["MSE", "RMSE", "MAE", "R^2"],
                "Value": [mse, rmse, mae, r2]
            })

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

            # ===== 主要なポイントまとめ =====
            """
            1. 符号制約付き回帰を cvxpy で解き、SSR を最小化。
            2. 切片の有無を選択可能。
            3. t値は OLS に基づく簡易的な計算のため、符号制約の影響を正確には反映できない可能性がある。
            4. 回帰係数や評価指標はエクセルでダウンロード可能。
            """
