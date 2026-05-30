import os

def is_kaggle() -> bool:
    """判断当前是否在 Kaggle 云端运行"""
    return "KAGGLE_URL_BASE" in os.environ or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

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
