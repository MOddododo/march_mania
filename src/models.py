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
    
    try:
        ipy = get_ipython()
        if ipy.__class__.__name__ == 'ZMQInteractiveShell':
            is_notebook = True
    except NameError:
        pass

    if is_notebook:
        # === Jupyter Notebook 环境 ===
        project_root = str(Path.cwd().parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        ipy.run_line_magic('load_ext', 'autoreload')
        ipy.run_line_magic('autoreload', '2')

        # 导入并返回模块
        from src import config
        print('Jupyter Notebook 环境')
        return config

    else:
        # === 普通 Python 脚本环境 ===
        try:
            from src import config
            print('普通脚本环境')
            return config
        except ImportError:
            import config
            print('普通脚本环境')
            return config

config = _import_config()

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline

def train_and_calibrate_models(tourney_data, features, xgb_params, num_rounds=700, spline_t=25):
    models = {}
    oof_preds = []
    oof_targets = []
    oof_ss = []
    
    seasons = sorted(tourney_data['Season'].unique())

    # 1. 训练
    for oof_season in seasons:
        if oof_season < min(seasons) + 3:
            continue

        x_train = tourney_data.loc[tourney_data["Season"] < oof_season, features].values
        y_train = tourney_data.loc[tourney_data["Season"] < oof_season, "PointDiff"].values
        x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
        y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
        s_val = tourney_data.loc[tourney_data["Season"] == oof_season, "Season"].values

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        
        models[oof_season] = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            verbose_eval=False
        )
        
        preds = models[oof_season].predict(dval)
        mae = mean_absolute_error(y_val, preds)
        print(f"Season {oof_season} - PointDiff MAE: {mae:.4f}")
        
        oof_preds.extend(preds)
        oof_targets.extend(y_val)
        oof_ss.extend(s_val)

    # 2. 样条插值
    # 将模型输出的分差排序，并与真实的胜负结果(>0为胜)对齐
    dat = list(zip(oof_preds, np.array(oof_targets) > 0))
    dat = sorted(dat, key=lambda x: x[0])
    sorted_preds, sorted_labels = list(zip(*dat))

    # 训练 UnivariateSpline，拟合 S 型曲线
    spline_model = UnivariateSpline(np.clip(sorted_preds, -spline_t, spline_t), sorted_labels, k=5)
    
    spline_fit_oof = np.clip(spline_model(np.clip(oof_preds, -spline_t, spline_t)), 0.01, 0.99)
    final_brier = brier_score_loss(np.array(oof_targets) > 0, spline_fit_oof)
    
    print(f"\n Brier Score: {final_brier:.5f} ")
    
    return models, spline_model


def predict_submission_ensemble(submission_df, models, spline_model, features, spline_t=25, aggression=1.0):
    
    dtest = xgb.DMatrix(submission_df[features].values)
    preds = []
    season = 2025

    margin_preds = models[season].predict(dtest) * aggression
    probs = np.clip(spline_model(np.clip(margin_preds, -spline_t, spline_t)), 0.01, 0.99)
    preds.append(probs)

    result_df = pd.DataFrame({
        'ID': submission_df['ID'],
        'Pred': np.array(preds).mean(axis=0) 
    })
    result_df.drop_duplicates(inplace=True)
    
    return result_df

def main_models(tourney_results, submission_df):
    models, spline_model = train_and_calibrate_models(tourney_results, config.XGB_features, config.param, config.num_rounds, config.spline_t)
    result_df = predict_submission_ensemble(submission_df, models, spline_model, config.XGB_features, config.spline_t, config.aggression)
    return result_df