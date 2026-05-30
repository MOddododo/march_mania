import os
import time
import zipfile
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv
from get_basic_info import *

SEASON_COL = "Season"
FRAC = 0.1  # 每个赛季抽取 10%
CHUNKSIZE = 20000
RANDOM_STATE = 42
def get_datasets(dataset_name,froms='gitee',frac=FRAC,timeout=15):
    if is_kaggle():
        cur_dir = Path('/kaggle/input/competitions/march-machine-learning-mania-2026')
        dataset_path = cur_dir / dataset_name
        if dataset_path.exists()==False:
            print(f"数据集 {dataset_name} 不存在")
            return None
        else:
            return pd.read_csv(dataset_path)
    else:
        local_dir = Path('local_data')
        dataset_path = local_dir / dataset_name
        if dataset_path.exists()==False:
            print(f"数据集 {dataset_name} 不存在，开始下载")
            base_url = "https://gitee.com/modaco/march_machine_datasets/raw/main/march-machine-learning-mania-2026"
            download_url = f"{base_url}/{dataset_name}"
            
            try:
                res=requests.head(download_url, allow_redirects=True, timeout=timeout)
                if res.status_code == 405:  # 部分 CDN 禁止 HEAD
                    res = requests.get(download_url, stream=True, timeout=timeout)
                    res.close()  # 立即关闭连接，不读 Body
            
                if res.status_code != 200:
                    print(f"URL 不可达 (HTTP {res.status_code})，请检查链接或权限")
                    return None
                print("URL 有效，开始流式采样...\n")
            except requests.RequestException as e:
                print(f"网络请求失败: {e}")
                return None

            sampled_chunks = []
            try:
                for i, chunk in enumerate(pd.read_csv(download_url, chunksize=CHUNKSIZE)):
                    if chunk.empty: break
                    # 当前块内按 Season 分组独立采样
                    sampled = chunk.groupby(SEASON_COL).sample(frac=frac, random_state=RANDOM_STATE)
                    sampled_chunks.append(sampled)
                    print(f"块 {i+1} 处理完成 | 累计采样: {sum(len(c) for c in sampled_chunks):,} 行")
            except Exception as e:
                print(f"读取/解析失败: {e}")
                return None

            if not sampled_chunks:
                print("未匹配到任何有效数据")
                return None
            df_final = pd.concat(sampled_chunks, ignore_index=True)
            print(f"\n采样完成: 共 {df_final.shape[0]:,} 行 | 覆盖 {df_final[SEASON_COL].nunique()} 个赛季")
            df_final.to_csv(dataset_path, index=False)
            return df_final
        else:
            return pd.read_csv(dataset_path)


