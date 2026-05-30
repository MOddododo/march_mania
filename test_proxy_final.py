# test_proxy_final.py
import requests
import os
import sys

def get_proxy_session(port=None, protocol="http"):
    """返回配置好的 Session（学校网络 + VPN 专用）"""
    # 1. 确定端口：参数 > 环境变量 > 默认值
    port = port or os.getenv("PROXY_PORT", "7890")
    
    # 2. 创建 Session
    session = requests.Session()
    
    # 3. 🔥 关键：禁用系统代理自动探测（避免学校网络干扰）
    session.trust_env = False
    
    # 4. 配置代理
    proxy = f"{protocol}://127.0.0.1:{port}"
    session.proxies = {
        "http": proxy,
        "https": proxy,
        "no_proxy": "localhost,127.0.0.1",
    }
    
    # 5. 防 403 请求头
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/csv,*/*;q=0.9",
    })
    
    return session, port

def test_url(session, url, desc=""):
    """测试单个 URL"""
    try:
        print(f"  🔗 {desc or url[:60]}...")
        r = session.get(url, timeout=15, stream=True)
        r.close()  # 只测试连接，不下载内容
        status = "✅" if r.status_code < 400 else "❌"
        print(f"  {status} HTTP {r.status_code}")
        return r.status_code == 200
    except requests.exceptions.ProxyError as e:
        print(f"  ❌ 代理连接失败: {str(e)[:80]}")
        return False
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {str(e)[:80]}")
        return False

def main():
    print("🚀 代理配置测试工具 (学校网络 + VPN 专用)\n")
    
    # 🔧 配置区域（按需修改）
    TEST_URLS = [
        ("https://raw.githubusercontent.com", "GitHub Raw 根域名"),
        ("https://raw.githubusercontent.com/MOddododo/march_mania_datasets/main/march-machine-learning-mania-2026/Cities.csv", "你的数据集文件"),
        ("https://httpbin.org/ip", "代理 IP 验证"),
    ]
    
    # 尝试的端口列表（自动轮询）
    PORTS = [
        os.getenv("PROXY_PORT"),  # 环境变量优先
        "7892",  # Clash 默认
        "7891",  # Clash SOCKS5
        "10809", # V2Ray HTTP
        "10808", # V2Ray SOCKS5
        "2080",  # Sing-Box
    ]
    PORTS = [p for p in PORTS if p]  # 过滤 None
    
    # 尝试的协议
    PROTOCOLS = ["http", "socks5"]
    
    # 🔁 轮询测试
    for protocol in PROTOCOLS:
        for port in PORTS:
            print(f"\n🔌 测试: {protocol}://127.0.0.1:{port}")
            print("-" * 50)
            
            try:
                session, used_port = get_proxy_session(port=port, protocol=protocol)
            except Exception as e:
                print(f"  ⚠️ Session 创建失败: {e}")
                continue
            
            # 测试所有 URL
            results = [test_url(session, url, desc) for url, desc in TEST_URLS]
            
            # 如果至少一个成功，说明代理可用
            if any(results):
                print(f"\n🎉 代理可用! 推荐配置:")
                print(f"   $env:PROXY_PORT=\"{used_port}\"")
                print(f"   $env:PROXY_PROTOCOL=\"{protocol}\"")
                print(f"\n💡 在代码中使用:")
                print(f"   session = get_proxy_session(port='{used_port}', protocol='{protocol}')")
                return 0
            else:
                print(f"  ⚠️ 该端口/协议组合不可用，尝试下一个...\n")
    
    # 全部失败
    print("\n❌ 所有代理配置均失败")
    print("\n💡 排查建议:")
    print("   1. 确认 VPN 已开启（Clash/V2Ray 状态为 'Running'）")
    print("   2. 在 VPN 设置中查看 '本地端口' / 'Local Port'")
    print("   3. 测试端口监听: netstat -ano | findstr '7890\\|10809'")
    print("   4. 如果学校拦截常见端口，尝试换非常用端口（如 8888）")
    print("   5. 终极保底: 手动下载文件到 local_data/，代码自动读本地")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())