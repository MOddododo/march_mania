import os
# 本地数据下载需要配置
SEASON_COL = "Season"
FRAC = 0.1  # 每个赛季抽取 10%
CHUNKSIZE = 20000
RANDOM_STATE = 42

PROXIES = {
    "http": "http://127.0.0.1:7892",
    "https": "http://127.0.0.1:7892",
}

os.environ["HTTP_PROXY"] = PROXIES["http"]
os.environ["HTTPS_PROXY"] = PROXIES["https"]

LOCAL_PATH = './data/local_small'
# 判断是否是kaggle环境
def is_kaggle() -> bool:
    """判断当前是否在 Kaggle 云端运行"""
    return "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


# 全原始数据集仓库配置
def get_github_token() -> str:
    if is_kaggle():
        from kaggle_secrets import UserSecretsClient
        secret_label = "GH_TOKEN"
        secret_value = UserSecretsClient().get_secret(secret_label)
        return secret_value
    else:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        secret_value = os.environ.get("GH_TOKEN", "").strip()
        return secret_value
SECRET_TOKEN=get_github_token()   
OWNER="MOddododo"
REPO="march_mania_datasets"
BRANCH="main"

# 检测当前是否在 Jupyter Notebook 环境中运行
def is_running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False
    
if __name__ == "__main__":
    IS_NOTEBOOK = is_running_in_notebook()
    print("是否在 Jupyter Notebook 中运行：", IS_NOTEBOOK)
    print(f"当前环境：{'Jupyter Notebook' if IS_NOTEBOOK else '普通 Python 脚本'}")


# ELO rating system configurations
base_elo = 1000
elo_width = 400
k_factor = 100

# XGBoost configurations
# using features
XGB_features = [
    'Gender',
    'T1_seed',
    'T2_seed',
    'Seed_diff',
    'T1_avg_Score',
    'T1_avg_FGA',
    'T1_avg_Blk',
    'T1_avg_PF',
    'T1_avg_opponent_FGA',
    'T1_avg_opponent_Blk',
    'T1_avg_opponent_PF',
    'T1_avg_PointDiff',
    'T2_avg_Score',
    'T2_avg_FGA',
    'T2_avg_Blk',
    'T2_avg_PF',
    'T1_avg_Poss',
    'T1_avg_opponent_Poss',
    'T2_avg_opponent_FGA',
    'T2_avg_opponent_Blk',
    'T2_avg_opponent_PF',
    'T2_avg_PointDiff',
    'T2_avg_Poss',
    'T2_avg_opponent_Poss',
    'T1_elo',
    'T2_elo',
    'elo_diff',
    'T1_quality',
    'T2_quality',
    'T1_elo_tour',
    'T2_elo_tour',
    'elo_diff_tour'

]

param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.01
param["subsample"] = 0.6
param["colsample_bynode"] = 0.8
param["num_parallel_tree"] = 2
param["min_child_weight"] = 4
param["max_depth"] = 4
param["tree_method"] = "hist"
param['grow_policy'] = 'lossguide'
param["max_bin"] = 32

num_rounds = 700
spline_t = 25
aggression=1.0

