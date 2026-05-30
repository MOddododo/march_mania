import os
import time
import zipfile
import pandas as pd
import requests
from pathlib import Path
import io

SEASON_COL = "Season"
FRAC = 0.1  # 每个赛季抽取 10%
CHUNKSIZE = 20000
RANDOM_STATE = 42

# 🔑 1. 明确配置你的 7892 端口代理
PROXIES = {
    "http": "http://127.0.0.1:7892",
    "https": "http://127.0.0.1:7892",
}

# (可选) 告诉 pandas 等其他底层库也走代理
os.environ["HTTP_PROXY"] = PROXIES["http"]
os.environ["HTTPS_PROXY"] = PROXIES["https"]

def _build_headers() -> dict:
    """构造防拦截请求头"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/octet-stream,*/*;q=0.9",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://gitee.com/modaco/march_machine_datasets",
        "Origin": "https://gitee.com",
    }
    return headers

def _check_url_with_retry(url: str, max_retries: int = 3, timeout: int = 10) -> bool:
    """带重试的 URL 检测（应对临时 403/503）"""
    headers = _build_headers()
    
    for attempt in range(max_retries):
        try:
            # 🔑 2. 这里必须加上 proxies=PROXIES
            res = requests.head(url, headers=headers, proxies=PROXIES, allow_redirects=True, timeout=timeout)
            if res.status_code == 405:  
                res = requests.get(url, headers=headers, proxies=PROXIES, stream=True, timeout=timeout)
                res.close()
            
            if res.status_code == 200:
                return True
            elif res.status_code == 403 and attempt < max_retries - 1:
                print(f"⚠️ 尝试 {attempt+1}: 403 Forbidden，2 秒后重试...")
                time.sleep(2)
                continue
            else:
                print(f"❌ 检测失败 (HTTP {res.status_code})")
                return False
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"❌ 网络异常: {e}")
                return False
            time.sleep(1)
    return False

def download_local_datasets(dataset_name, froms='gitee', frac=FRAC, timeout=15):
    # 假设 is_kaggle 逻辑在外部，如果是本地环境直接走 else
    local_dir = Path('local_data')
    dataset_path = local_dir / dataset_name
    
    if not dataset_path.exists():
        print(f"数据集 {dataset_name} 不存在，开始下载")
        
        if froms == 'gitee':
            base_url = "https://gitee.com/modaco/march_machine_datasets/raw/main/march-machine-learning-mania-2026"
        elif froms == 'github':
            # 直接使用官方 GitHub 地址，因为现在我们有 VPN 代理了
            base_url = "https://raw.githubusercontent.com/MOddododo/march_mania_datasets/main/march-machine-learning-mania-2026"
        else:
            print("未知的 froms 参数")
            return None
            
        download_url = f"{base_url}/{dataset_name}"
        print(download_url)
        
        try:
            # 🔑 3. 检查 URL 时加上代理
            if not _check_url_with_retry(download_url, timeout=timeout):
                print("💡 提示：1. 检查仓库是否私有 2. 检查 VPN 7892 端口是否正常运行")
                return None
            print("✅ URL 有效，开始流式采样...\n")
        except requests.RequestException as e:
            print(f"网络请求失败: {e}")
            return None

        sampled_chunks = []
        try:
            print("📥 正在快速拉取数据到内存 (防连接中断)...")
            # 1. 创建一个内存缓冲区
            file_buffer = io.BytesIO()
            
            # 发起请求
            with requests.get(download_url, headers=_build_headers(), proxies=PROXIES, stream=True, timeout=timeout*3) as response:
                response.raise_for_status()
                
                # 🔑 核心修复：用大块（1MB）全速抽干网络流，不给 VPN 掐断的机会
                # (注: requests.iter_content 会自动处理 gzip 解压，所以存入 buffer 的已经是解压后的数据了)
                for net_chunk in response.iter_content(chunk_size=1024 * 1024):
                    if net_chunk:
                        file_buffer.write(net_chunk)
            
            print("✅ 网络下载完成，开始本地数据采样...")
            
            # 2. 将游标拨回缓冲区开头，转换为文本流交给 Pandas
            file_buffer.seek(0)
            text_stream = io.TextIOWrapper(file_buffer, encoding='utf-8')
            
            # 此时 Pandas 是在读取本地内存，绝对不会报网络连接断开了
            for i, chunk in enumerate(pd.read_csv(text_stream, chunksize=CHUNKSIZE, engine='python', on_bad_lines='skip')):
                if chunk.empty: break
                
                if SEASON_COL not in chunk.columns:
                    print(f"❌ 错误：下载的数据中找不到 '{SEASON_COL}' 列，可能代理返回了错误页面。")
                    return None
                    
                sampled = chunk.groupby(SEASON_COL).sample(frac=frac, random_state=RANDOM_STATE)
                sampled_chunks.append(sampled)
                print(f"块 {i+1} 处理完成 | 累计采样: {sum(len(c) for c in sampled_chunks):,} 行")
                
        except Exception as e:
            print(f"❌ 读取/解析失败: {type(e).__name__}: {e}")
            return None

        if not sampled_chunks:
            print("未匹配到任何有效数据")
            return None
            
        df_final = pd.concat(sampled_chunks, ignore_index=True)
        print(f"\n🎉 采样完成: 共 {df_final.shape[0]:,} 行 | 覆盖 {df_final[SEASON_COL].nunique()} 个赛季")
        
        local_dir.mkdir(exist_ok=True)
        df_final.to_csv(dataset_path, index=False)
        return df_final
    else:
        print(f"✅ 从本地缓存读取: {dataset_name}")
        return pd.read_csv(dataset_path)

# 测试运行
if __name__ == "__main__":
    massey = download_local_datasets('MMasseyOrdinals.csv', froms='github', frac=0.3)