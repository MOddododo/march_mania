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
