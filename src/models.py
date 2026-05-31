import pandas as pd
import numpy as np

# 检测当前是否在 Jupyter Notebook 环境中运行
import sys
from pathlib import Path
def _import_config():
    """
    根据运行环境动态导入 config 模块
    Notebook 环境 -> 返回 src.config
    普通 py 环境 -> 返回 config
    """
    is_notebook = False
    ipy = None
    
    # 1. 安全检测环境（不依赖额外的 import IPython）
    try:
        ipy = get_ipython()
        if ipy.__class__.__name__ == 'ZMQInteractiveShell':
            is_notebook = True
    except NameError:
        pass

    # 2. 根据环境进行不同的导入逻辑
    if is_notebook:
        # === Jupyter Notebook 环境 ===
        # 将项目根目录加入环境变量 (假设 notebook 在子文件夹中)
        project_root = str(Path.cwd().parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 【关键修复】使用代码的方式调用魔法命令，避免在 .py 中报 SyntaxError
        ipy.run_line_magic('load_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')

        # 导入并返回模块
        from src import config
        print('✅ [动态加载] Jupyter Notebook 环境：已加载 src.config')
        return config

    else:
        # === 普通 Python 脚本环境 ===
        import config
        print('✅ [动态加载] 普通脚本环境：已加载当前目录的 config')
        return config
config = _import_config()

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline

def train_and_calibrate_models(tourney_data, features, xgb_params, num_rounds=700, spline_t=25):
    """
    第一段：使用 Leave-one-season-out 训练 XGBoost 预测分差。
    第二段：使用 UnivariateSpline 将预测分差校准为 0~1 的胜率概率。
    
    返回:
    - models (dict): 每个赛季对应的训练好的 XGBoost 模型
    - spline_model (UnivariateSpline): 训练好的概率转换函数
    """
    print("🚀 开始训练 XGBoost 并进行概率校准...")
    models = {}
    oof_preds = []
    oof_targets = []
    oof_ss = []

    seasons = sorted(tourney_data['Season'].unique())

    # 1. 训练阶段 (预测 PointDiff)
    for oof_season in seasons:
        # 保护机制：确保有足够的前置历史数据（至少有3年的数据再开始验证）
        if oof_season < min(seasons) + 3:
            continue

        x_train = tourney_data.loc[tourney_data["Season"] < oof_season, features].values
        y_train = tourney_data.loc[tourney_data["Season"] < oof_season, "PointDiff"].values
        x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
        y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
        s_val = tourney_data.loc[tourney_data["Season"] == oof_season, "Season"].values

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        
        # 训练该赛季的模型
        models[oof_season] = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            verbose_eval=False
        )
        
        # 记录验证集预测结果
        preds = models[oof_season].predict(dval)
        mae = mean_absolute_error(y_val, preds)
        print(f"✅ Season {oof_season} - PointDiff MAE: {mae:.4f}")
        
        oof_preds.extend(preds)
        oof_targets.extend(y_val)
        oof_ss.extend(s_val)

    # 2. 概率校准阶段 (样条插值)
    # 将模型输出的分差排序，并与真实的胜负结果(>0为胜)对齐
    dat = list(zip(oof_preds, np.array(oof_targets) > 0))
    dat = sorted(dat, key=lambda x: x[0])
    sorted_preds, sorted_labels = list(zip(*dat))

    # 训练 UnivariateSpline，拟合 S 型曲线
    spline_model = UnivariateSpline(np.clip(sorted_preds, -spline_t, spline_t), sorted_labels, k=5)
    
    # 在 OOF 数据上计算最终的 Brier Score 评估概率质量
    spline_fit_oof = np.clip(spline_model(np.clip(oof_preds, -spline_t, spline_t)), 0.01, 0.99)
    final_brier = brier_score_loss(np.array(oof_targets) > 0, spline_fit_oof)
    
    print(f"\n🎯 训练完成! 整体交叉验证 Brier Score: {final_brier:.5f} (越接近0越好)")
    
    return models, spline_model


def predict_submission(submission_df, models, spline_model, features, spline_t=25):
    """
    利用训练好的多赛季模型和样条函数，对最终提交集进行概率预测。
    """
    print("\n🔮 开始为 Submission 集合生成预测概率...")
    
    # 确保 submission_df 包含所需的特征列
    missing_cols = set(features) - set(submission_df.columns)
    if missing_cols:
        raise ValueError(f"submission_df 缺少必要的特征列: {missing_cols}")

    final_probs = []
    
    # 遍历要预测的每一行 (每场比赛)
    for _, row in submission_df.iterrows():
        season = row['Season']
        
        # 提取特征转为 DMatrix
        x_test = row[features].values.reshape(1, -1)
        dtest = xgb.DMatrix(x_test)
        
        # 智能选择模型：
        # 如果是回测历史赛季，用那个赛季对应的模型；
        # 如果是预测当年未发生的比赛(如2024)，用我们手上最新的那个模型。
        if season in models:
            pred_diff = models[season].predict(dtest)[0]
        else:
            latest_season = max(models.keys())
            pred_diff = models[latest_season].predict(dtest)[0]
            
        # 将预测出的分差，喂给样条模型转化为胜率
        pred_prob = spline_model(np.clip(pred_diff, -spline_t, spline_t))
        
        # 截断极值：防止模型给出 0 或 1 这种会因爆冷导致 LogLoss 爆炸的自信概率
        pred_prob = np.clip(pred_prob, 0.01, 0.99)
        
        # 由于我们预测的永远是 T1 赢的概率，如果是 T2 实际上是主视角，这里也可以灵活调整
        final_probs.append(float(pred_prob))

    # 生成 Kaggle 标准的提交格式 DataFrame
    result_df = pd.DataFrame({
        'ID': submission_df['ID'],
        'Pred': final_probs
    })
    
    print("✨ 预测完成！")
    return result_df