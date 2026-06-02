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

def main_download_submission(submission_df):
    if config.is_kaggle():
        submission_df[['ID','Pred']].to_csv('predictions.csv',index=None)
    else:
        project_note = Path(__file__).resolve().parents[1]
        output_dir = project_note / 'submissions' / 'predictions.csv'
        submission_df[['ID','Pred']].to_csv(output_dir,index=None)
