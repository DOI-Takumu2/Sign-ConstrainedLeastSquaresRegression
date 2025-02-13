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

def format_t_and_sign(t_val, sign_val):
    """
    t値と符号制約を substack でまとめた上付き文字列を返す。
    - t値がNaNの場合は (t=...) を表示しない。
    - sign_val は 'free', 'positive', 'negative' または '(None)' を想定。
    """
    # t値の整形
    t_str = ""
    if not pd.isna(t_val):
        t_str = f"(t={t_val:.4f})"

    # signの整形
    # (None) は切片の場合などに使う
    if sign_val is None:
        sign_str = "(sign=none)"
    else:
        sign_str = f"(sign={sign_val})"

    # t値も sign もないとき
    if (t_str == "") and (sign_str == "(sign=none)" or sign_str == ""):
        return ""  # 何も表示しない
    
    # substack で改行させる
    # 例:  \substack{(t=1.2345) \\ (sign=positive)}
    # t_val がNaNの場合は sign だけ表示
    if t_str != "":
        return rf"^\substack{{{t_str}\\{sign_str}}}"
    else:
        # tなし signだけ
        return rf"^\substack{{{sign_str}}}"

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
                df[col] = pd.to_numeric(df[col], errors='coerce')  # 数値変換

            for col in analysis_cols:
                n_missing_before = df[col].isnull().sum()
                if n_missing_before > 0:
                    col_mean = df[col].mean()
                    if pd.isna(col_mean):  # 全行NaNの場合など
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
            st.subheader("【回帰式（数式表示）】")

            # 目的変数名
            # 例: y = ...
            latex_equation = rf"{target_col} = "

            # 切片 (if intercept_flag == 切片あり)
            if intercept_flag == "切片あり":
                t_int = t_values[-1]  # 切片のt値
                # (t=..., sign=none) を substack表示
                intercept_exponent = format_t_and_sign(t_int, None)  # sign=None として扱う
                intercept_str = rf"\beta_0{intercept_exponent}"
                latex_equation += intercept_str
            else:
                # 切片なしの場合は何も足さない
                intercept_str = ""

            # それぞれの係数×変数を追加
            # β_i^(...) x_i
            # ただし、最初の項以降は " + " を付ける
            for i, col in enumerate(feature_cols):
                # t値
                t_val_current = t_values[i] if intercept_flag == "切片あり" else t_values[i]
                sign_val = sign_constraints[col]
                exponent_str = format_t_and_sign(t_val_current, sign_val)

                # 例: \beta_1^(t=..., sign=positive) x_1
                term_str = rf"\beta_{{{i+1}}}{exponent_str} x_{{\text{{{col}}}}}"

                if i == 0 and intercept_flag == "切片あり":
                    # 切片がある場合、これは「 + 」で繋ぐ
                    latex_equation += rf" + {term_str}"
                elif i == 0 and intercept_flag == "切片なし":
                    # 切片なしの場合、最初の項なので「 = β_1 x_1」 の形
                    latex_equation += term_str
                else:
                    # 2番目以降
                    latex_equation += rf" + {term_str}"

            # 数式を $$...$$ で囲んで表示
            latex_display = f"$$ {latex_equation} $$"
            st.markdown(latex_display)

            # 目的関数値(SSR)
            ssr_val = problem.value

            # 表形式で指標を表示
            st.subheader("【評価指標】")
            metrics_data = [
                ["SSR (Sum of Squared Residuals)", f"{ssr_val:.4f}"],
                ["MSE (Mean Squared Error)", f"{mse:.4f}"],
                ["RMSE (Root Mean Squared Error)", f"{rmse:.4f}"],
                ["MAE (Mean Absolute Error)", f"{mae:.4f}"],
                ["R^2 (Coefficient of Determination)", f"{r2:.4f}"],
            ]
            metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
            st.table(metrics_df)

            # --- 10. ダウンロード用ファイル生成 ---
            output = io.BytesIO()

            # Excelに出力するデータフレーム（Coefficients, Metrics）
            if intercept_flag == "切片あり":
                feature_list = ["Intercept"] + feature_cols
                coef_list = [intercept_val] + list(coef_vals.flatten())
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
                "Metric": [
                    "SSR",
                    "MSE",
                    "RMSE",
                    "MAE",
                    "R^2"
                ],
                "Value": [
                    ssr_val,
                    mse,
                    rmse,
                    mae,
                    r2
                ]
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
            1. 目的変数名 = 切片(上付きに t値・sign) + Σ (係数(上付きに t値・sign) × 説明変数)
            2. t値が NaN の場合は (t=..) 表示を省略。切片は sign=none として扱う。
            3. 目的関数値や各種評価指標は表形式で表示。
            4. 欠損値や型違いデータは平均値で補完し、OSQP → ECOS の順でソルバーを試行。
            """

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
else:
    st.write("上記に Excel ファイルをアップロードしてください。")
