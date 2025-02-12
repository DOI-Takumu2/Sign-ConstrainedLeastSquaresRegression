# ============================================
#  1) 必要なライブラリのインストール（Colab）
# ============================================

# ============================================
#  2) ライブラリの読み込み
# ============================================
import io
import numpy as np
import pandas as pd
from google.colab import files
import cvxpy as cp
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ipywidgets 関連
import ipywidgets as widgets
from IPython.display import display

# ============================================
#  3) ファイルのアップロード
# ============================================
print("★解析対象のExcelファイルをアップロードしてください。")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

df = pd.read_excel(io.BytesIO(uploaded[filename]))

print("\nアップロードしたデータを読み込みました。")
print("データの先頭5行を表示します。")
display(df.head())

print("以下の列が利用可能です:")
all_cols = df.columns.tolist()
print(all_cols)

# ============================================
#  4) ウィジェットでパラメータ選択
# ============================================

# --- 目的変数を選ぶドロップダウン ---
target_dropdown = widgets.Dropdown(
    options=all_cols,
    description='【目的変数】',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='50%')
)

# --- 説明変数を選ぶ複数選択ウィジェット ---
feature_select = widgets.SelectMultiple(
    options=all_cols,
    description='【説明変数】\n（Ctrl/Cmd で複数選択）',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='50%', height='150px')
)

# --- 切片を0にするか否か ---
intercept_radio = widgets.RadioButtons(
    options=['切片あり', '切片なし'],
    value='切片あり',
    description='【切片の設定】',
    style={'description_width': 'initial'}
)

# --- 各説明変数の符号制約を選択するためのウィジェットを動的に作る ---
# 説明変数が確定した後に制約を設定したいので、一旦は空のコンテナにしておく
sign_widgets_box = widgets.VBox()

# ボタン：制約の設定を反映 → 回帰実行
run_button = widgets.Button(
    description='回帰を実行',
    button_style='primary',
    tooltip='指定した設定で回帰を実行します'
)

# 結果出力用エリア
output_area = widgets.Output()

# ============================================
#  5) 近似的なt値計算のための関数
# ============================================
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
    # ※実際の制約付き回帰の自由度は異なる可能性があります
    mse = sse / (n_samples - k)

    # (X^T X)^{-1} を計算
    xtx_inv = np.linalg.inv(X_aug.T @ X_aug)
    
    # 各パラメータの分散 = mse * 対角成分
    var_params = mse * np.diag(xtx_inv)
    se_params = np.sqrt(var_params)
    
    t_values = w_fit / se_params
    return t_values

# ============================================
#  6) 回帰を実行する関数
# ============================================
def run_regression(_):
    output_area.clear_output()
    
    # 選択された目的変数
    target_col = target_dropdown.value
    
    # 選択された説明変数
    feature_cols = list(feature_select.value)
    
    # 切片を入れるかどうか
    intercept_flag = (intercept_radio.value == '切片あり')
    
    # 符号制約をウィジェットから集める
    constraint_dict = {}
    for w_box in sign_widgets_box.children:
        # w_box.children = (Label, Dropdown) の想定
        if len(w_box.children) == 2:
            col_label = w_box.children[0].value  # 説明変数名を保持
            sign_val = w_box.children[1].value  # 'positive'/'negative'/'free'
            constraint_dict[col_label] = sign_val
    
    with output_area:
        if target_col not in df.columns:
            print("エラー: 目的変数がデータに存在しません。")
            return
        # 説明変数のうちデータに存在するものだけ抽出（念のため）
        feature_cols = [c for c in feature_cols if c in df.columns]
        if len(feature_cols) == 0:
            print("エラー: 有効な説明変数が選択されていません。")
            return
        
        print("【設定内容】")
        print(f"   目的変数: {target_col}")
        print(f"   説明変数: {feature_cols}")
        print(f"   切片: {'あり' if intercept_flag else 'なし'}")
        print("   符号制約:", constraint_dict)
        print("---------------------------------------------------")
        
        # 欠損を含む行を削除
        df_clean = df.dropna(subset=[target_col] + feature_cols)
        
        # NumPy配列に変換
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        n_samples, n_features = X.shape
        
        # ========================
        # cvxpy による符号制約付き回帰
        # ========================
        if intercept_flag:
            # 切片用に b を変数とし、y ≈ Xw + b
            w = cp.Variable(n_features)
            b = cp.Variable()
            
            # 残差 = y - (Xw + b)
            residuals = y - (X @ w + b)
            
            # 制約設定
            constraints = []
            for i, col in enumerate(feature_cols):
                sign_type = constraint_dict.get(col, 'free')
                if sign_type == 'positive':
                    constraints.append(w[i] >= 0)
                elif sign_type == 'negative':
                    constraints.append(w[i] <= 0)
            
            # 目的関数: 残差平方和の最小化
            objective = cp.Minimize(cp.sum_squares(residuals))
            
            problem = cp.Problem(objective, constraints)
            result = problem.solve(solver=cp.OSQP)
            
            # 結果取得
            coef_vals = w.value
            intercept_val = b.value
            
            # 予測値
            y_pred = X @ coef_vals + intercept_val
            
            # t値の近似計算用に w_fit を作る ([b, w1, w2, ...] の形)
            w_fit_ols_like = np.concatenate([[intercept_val], coef_vals])
            
        else:
            # 切片 = 0 として w のみを推定
            w = cp.Variable(n_features)
            
            # 残差 = y - Xw
            residuals = y - (X @ w)
            
            # 制約設定
            constraints = []
            for i, col in enumerate(feature_cols):
                sign_type = constraint_dict.get(col, 'free')
                if sign_type == 'positive':
                    constraints.append(w[i] >= 0)
                elif sign_type == 'negative':
                    constraints.append(w[i] <= 0)
            
            # 目的関数: 残差平方和の最小化
            objective = cp.Minimize(cp.sum_squares(residuals))
            
            problem = cp.Problem(objective, constraints)
            result = problem.solve(solver=cp.OSQP)
            
            # 結果取得
            coef_vals = w.value
            intercept_val = 0.0
            
            # 予測値
            y_pred = X @ coef_vals
            
            # t値の近似計算用に w_fit を作る ([w1, w2, ...] の形)
            w_fit_ols_like = coef_vals
        
        # ========================
        # 評価指標
        # ========================
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # ========================
        # t値（近似）を計算
        # ========================
        t_values = calc_t_values(X, y, w_fit_ols_like, intercept_flag)
        
        # t値の出力の都合で、先頭が切片の場合もある
        if intercept_flag:
            # 先頭が切片
            t_intercept = t_values[0]
            t_coefs = t_values[1:]
        else:
            t_intercept = None
            t_coefs = t_values
        
        # ========================
        # 結果表示
        # ========================
        print("【回帰結果】")
        print(f"   目的関数値(SSR) : {problem.value:.4f}")
        if intercept_flag:
            print(f"   切片 (Intercept) : {intercept_val:.4f}  (t={t_intercept:.4f})")
        else:
            print(f"   切片 (Intercept) : 0 (固定)")
        print("   係数一覧:")
        for col, cval, tval in zip(feature_cols, coef_vals, t_coefs):
            print(f"      {col:>10} : {cval: .4f}  (t={tval:.4f})")
        
        print("\n【評価指標】")
        print(f"   MSE  : {mse:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   MAE  : {mae:.4f}")
        print(f"   R^2  : {r2:.4f}")
        
        # ========================
        # Excel 出力
        # ========================
        result_dict = {
            'Feature': ['Intercept'] + feature_cols if intercept_flag else feature_cols,
            'Coefficient': ([intercept_val] + list(coef_vals)) if intercept_flag else list(coef_vals),
            't-value': ([t_intercept] + list(t_coefs)) if intercept_flag else list(t_coefs),
            'SignConstraint': (['(None)'] + [constraint_dict.get(c, 'free') for c in feature_cols])
                              if intercept_flag else [constraint_dict.get(c, 'free') for c in feature_cols]
        }
        df_results = pd.DataFrame(result_dict)
        
        df_metrics = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R^2'],
            'Value': [mse, rmse, mae, r2]
        })
        
        with pd.ExcelWriter('regression_results.xlsx') as writer:
            df_results.to_excel(writer, sheet_name='Coefficients', index=False)
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
        
        files.download('regression_results.xlsx')
        print("\n★回帰結果をExcelファイル (regression_results.xlsx) に出力しました。ダウンロードしてください。")

# ============================================
#  7) 「説明変数の符号制約」ウィジェットを生成する関数
# ============================================
def create_sign_widgets(change):
    """
    feature_select が変更されたときに呼び出し、
    選択された各説明変数に対して「符号制約」用のドロップダウンを作成する。
    """
    sign_widgets_box.children = []  # いったんクリア
    
    selected_features = list(feature_select.value)
    # 選択されている変数分だけウィジェット作成
    new_children = []
    for fcol in selected_features:
        # ラベル的に保持するため、value として列名を渡す
        label_w = widgets.Label(value=fcol)
        sign_w = widgets.Dropdown(options=['positive','negative','free'],
                                  value='free',
                                  layout=widgets.Layout(width='150px'))
        hbox = widgets.HBox([label_w, sign_w])
        new_children.append(hbox)
    
    sign_widgets_box.children = new_children

# 説明変数選択ウィジェットにコールバックを設定
feature_select.observe(create_sign_widgets, names='value')

# ============================================
#  8) ボタン押下で回帰を実行
# ============================================
run_button.on_click(run_regression)

# ============================================
#  9) すべてのウィジェットを表示
# ============================================
print("\n▼ 以下のウィジェットでパラメータを選択してください。")
display(target_dropdown)
display(feature_select)
display(intercept_radio)
print("▼ 各説明変数の符号制約を指定してください。")
display(sign_widgets_box)
display(run_button)

print("\n▼ 「回帰を実行」ボタンを押すと計算が始まります。")
display(output_area)


