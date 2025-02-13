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

# t値のフォーマット用関数
def format_t_value(t):
    """t値がNaNでなければ (t=XXX) の文字列を返し, NaNなら空文字を返す."""
    if pd.isna(t):
        return ""
    else:
        return f"(t={t:.4f})"

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("アップロードしたデータの先頭5行:")
    st.write(df.head())

    # **目的変数 & 説明変数の選択**
    target_col = st.selectbox("【目的変数】", df.columns.tolist())
    feature_cols = st.multiselect("【説明変数】", df.columns.tolist())
    intercept_flag = st.radio("【切片の設定】", ["切片あり", "切片なし"])

    # **各説明変数の符号制約を設定**
    sign_constraints = {
        col: st.selectbox(f"{col} の符号制約", ["free", "positive", "negative"])
        for col in feature_cols
    }

    # **回帰分析を実行（ボタン）**
    if st.button("回帰を実行"):
        try:
            # --- 1. 入力チェック ---
            if not target_col or not feature_cols:
                st.error("エラー: 目的変数と説明変数を正しく選択してください。")
                st.stop()

            # --- 2. 分析に使用する列のみ抽出 ---
            analysis_cols = [target_col] + feature_cols

            # --- 3. 数値型への変換 + 欠損の集計 + 平均値補完 ---
            total_filled = 0
            for col in analysis_cols:
                # 数値変換（変換できない値は NaN になる）
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 各列ごとに欠損値を平均値で埋める
            for col in analysis_cols:
                n_missing_before = df[col].isnull().sum()
                if n_missing_before > 0:
                    col_mean = df[col].mean()
                    # col_mean が NaN の場合、全ての行が NaN か計算不能 -> 0.0 で埋めるなど適宜対応
                    if pd.isna(col_mean):
                        col_mean = 0.0
                    df[col].fillna(col_mean, inplace=True)
                    n_filled = n_missing_before - df[col].isnull().sum()
                    total_filled += n_filled

            if total_filled > 0:
                st.write(f"{total_filled} 個のデータに不備（欠損・数値以外のデータ）があったため平均値で補完しました。")

            # --- 4. データセットの作成 ---
            X = df[feature_cols].values
            y = df[target_col].values
            n_samples, n_features = X.shape

            # --- 5. 符号制約付き回帰問題を定義 ---
            w = cp.Variable(n_features)
            b = cp.Variable() if intercept_flag == "切片あり" else 0
            residuals = y - (X @ w + b)

            constraints = []
            for i, col in enumerate(feature_cols):
                if sign_constraints[col] == "positive":
                    constraints.append(w[i] >= 0)
                elif sign_constraints[col] == "negative":
                    constraints.append(w[i] <= 0)

            objective = cp.Minimize(cp.sum_squares(residuals))
            problem = cp.Problem(objective, constraints)

            # --- 6. ソルバーを実行 (OSQP -> fallback: ECOS) ---
            try:
                result = problem.solve(solver=cp.OSQP)
            except:
                st.warning("OSQP ソルバーで解が求まらなかったため、ECOS ソルバーに切り替えます。")
                result = problem.solve(solver=cp.ECOS)

            coef_vals = w.value
            intercept_val = b.value if intercept_flag == "切片あり" else 0.0

            # --- エラーチェック ---
            if coef_vals is None:
                st.error("エラー: 最適化が収束しなかったため、回帰係数が計算できませんでした。データやスケールを確認してください。")
                st.stop()

            if intercept_flag == "切片あり" and intercept_val is None:
                st.error("エラー: 最適化が収束しなかったため、切片が計算できませんでした。データやスケールを確認してください。")
                st.stop()

            if coef_vals.ndim == 1:
                coef_vals = coef_vals.reshape(-1, 1)

            # --- 7. モデル評価 ---
            y_pred = X @ coef_vals + intercept_val
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # --- 8. 近似的な t 値の計算 ---
            t_values = []
            try:
                # デザイン行列（切片ありの場合は最後に1列足す）
                if intercept_flag == "切片あり":
                    X_design = np.hstack([X, np.ones((n_samples, 1))])
                    w_full = np.vstack([coef_vals, intercept_val])
                else:
                    X_design = X
                    w_full = coef_vals

                residual = y - (X_design @ w_full).flatten()
                df_error = n_samples - X_design.shape[1]
                SSE = np.sum(residual**2)
                s2 = SSE / df_error if df_error > 0 else np.nan
                xtx_inv = np.linalg.inv(X_design.T @ X_design)
                var_beta = s2 * np.diag(xtx_inv)
                se_beta = np.sqrt(var_beta)

                w_full_flat = w_full.flatten()
                for i in range(len(w_full_flat)):
                    if se_beta[i] == 0:
                        t_values.append(np.nan)
                    else:
                        t_values.append(w_full_flat[i] / se_beta[i])
            except np.linalg.LinAlgError:
                st.warning("警告: (X^T X) の逆行列が計算できませんでした。t値は計算されません。")
                t_values = [np.nan]*(n_features + (1 if intercept_flag == "切片あり" else 0))

            # --- 9. 結果表示 ---
            st.write("【回帰結果】")
            st.write(f"目的関数値(SSR): {problem.value:.4f}")

            # 切片（intercept_flagがある場合）
            if intercept_flag == "切片あり":
                t_str_intercept = format_t_value(t_values[-1])
                st.write(f"切片: {intercept_val:.4f} {t_str_intercept}")

            # 各特徴量
            for i, col in enumerate(feature_cols):
                t_val_current = t_values[i]
                t_str_current = format_t_value(t_val_current)
                st.write(f"{col}: {coef_vals[i, 0]:.4f} [{sign_constraints[col]}] {t_str_current}")

            st.write("【評価指標】")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"R^2: {r2:.4f}")

            # --- 10. ダウンロード用ファイル生成 ---
            output = io.BytesIO()

            if intercept_flag == "切片あり":
                feature_list = ["Intercept"] + feature_cols
                coef_list = [intercept_val] + list(coef_vals.flatten())
                # t値の順番：最後が切片
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

            # --- 主要なポイントまとめ ---
            """
            1. 欠損値や型違いの値は自動的に NaN に変換し、平均値で補完するためエラーが起こりにくい。
            2. ソルバーが収束しない場合は別のソルバーに切り替えます。
            3. t値は OLSに基づく近似値であり、符号制約の影響を厳密には反映しない場合があります。
            4. 結果はExcelファイルに出力可能です。
            """

        except Exception as e:
            # 万が一ここまでで想定外のエラーが出たときも、Streamlitを落とさないために捕捉する
            st.error(f"エラーが発生しました: {e}")
else:
    st.write("上記に Excel ファイルをアップロードしてください。")
