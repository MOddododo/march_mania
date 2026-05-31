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
import pandas as pd
import numpy as np
def get_main_datasets(full_data=False):
    if config.is_kaggle():
        input_dir = Path("/kaggle/input/competitions/march-machine-learning-mania-2026")
        W_seeds = pd.read_csv(input_dir / 'WNCAATourneySeeds.csv')
        M_seeds = pd.read_csv(input_dir / 'MNCAATourneySeeds.csv')

        M_regular_results = pd.read_csv(input_dir / 'MRegularSeasonDetailedResults.csv')
        W_regular_results = pd.read_csv(input_dir / 'WRegularSeasonDetailedResults.csv')

        M_tourney_results = pd.read_csv(input_dir / 'MNCAATourneyDetailedResults.csv')
        W_tourney_results = pd.read_csv(input_dir / 'WNCAATourneyDetailedResults.csv')
        regular_results = pd.concat([M_regular_results, W_regular_results])
        tourney_results = pd.concat([M_tourney_results, W_tourney_results])
        seeds = pd.concat([M_seeds, W_seeds])
        return regular_results, tourney_results, seeds
    else:
        project_note = Path(__file__).resolve().parents[1]
        if full_data:
            input_dir=project_note / 'data' / 'raw'
        else:
            input_dir = project_note / 'data' / 'local_small'
        W_seeds = pd.read_csv(input_dir / 'WNCAATourneySeeds.csv')
        M_seeds = pd.read_csv(input_dir / 'MNCAATourneySeeds.csv')

        M_regular_results = pd.read_csv(input_dir / 'MRegularSeasonDetailedResults.csv')
        W_regular_results = pd.read_csv(input_dir / 'WRegularSeasonDetailedResults.csv')

        M_tourney_results = pd.read_csv(input_dir / 'MNCAATourneyDetailedResults.csv')
        W_tourney_results = pd.read_csv(input_dir / 'WNCAATourneyDetailedResults.csv')
        regular_results = pd.concat([M_regular_results, W_regular_results])
        tourney_results = pd.concat([M_tourney_results, W_tourney_results])
        seeds = pd.concat([M_seeds, W_seeds])
        return regular_results, tourney_results, seeds
    


if __name__ == '__main__':
    regular_results, tourney_results, seeds=get_main_datasets()
    print(regular_results.head())
