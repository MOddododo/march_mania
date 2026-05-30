import os
import time
import zipfile
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv

class PrivateZipLoader:
    def __init__(self, owner, repo, branch, zip_filename, 
                 cache_dir="./data_cache", cache_hours=24):
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.zip_filename = zip_filename
        load_dotenv(override=True)
        self.token = os.environ.get("GH_TOKEN", "").strip()
        
        # GitHub API 端点
        self.api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{zip_filename}"
        
        self.cache_path = Path(cache_dir) / zip_filename
        self.cache_ttl = cache_hours * 3600

    def _is_cache_valid(self):
        """检查缓存是否有效"""
        if not self.cache_path.exists():
            return False
        return (time.time() - self.cache_path.stat().st_mtime) < self.cache_ttl

    def download(self, force=False):
        """下载 ZIP 文件"""
        if not force and self._is_cache_valid():
            rem_h = (self.cache_ttl - (time.time() - self.cache_path.stat().st_mtime)) / 3600
            print(f"命中缓存 (剩余 {rem_h:.1f}h)  {self.cache_path}")
            return self.cache_path
            
        print("正在通过 GitHub API 下载私有 ZIP...")
        
        #认证头
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3.raw",  # 获取原始二进制内容
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        try:
            resp = requests.get(self.api_url, headers=headers, stream=True, timeout=120)
            resp.raise_for_status()
            
            # 确保缓存目录存在
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 分块写入文件
            with open(self.cache_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✅ 下载完成: {self.cache_path} ({self.cache_path.stat().st_size / 1024 / 1024:.2f} MB)")
            return self.cache_path
            
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 401:
                print("❌ 401: Token 无效 / 未授权 / 环境变量未生效")
                print("💡 检查：\n   1. load_dotenv(override=True) 是否调用\n   2. Token 权限是否含 repo\n   3. 组织仓库是否完成 SSO 授权")
            elif resp.status_code == 404:
                print(f"❌ 404: 文件路径错误")
                print(f"💡 请确认仓库 {self.owner}/{self.repo} 的 {self.branch} 分支中存在: {self.zip_filename}")
            else:
                print(f"❌ HTTP {resp.status_code}: {resp.text[:200]}")
            raise
        except Exception as e:
            print(f"❌ 下载异常: {type(e).__name__} - {e}")
            raise

    