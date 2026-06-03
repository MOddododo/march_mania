# March Mania: NCAA 比赛预测机器学习项目

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. 项目概述

**March Mania** 是一个基于机器学习的 NCAA 篮球锦标赛（March Madness）预测项目。本项目旨在解决锦标赛场景下高随机性、样本量小且存在显著时序演变的赛果预测问题。

### 核心功能定位
- **多维实力建模**：通过整合历史详尽统计数据，构建包含 Elo 评分、GLM 质量分及调整后场均表现的深度特征矩阵。
- **端到端流程**：自动化执行数据清洗、特征工程、模型训练、概率校准及预测结果生成的全过程。
- **跨平台适配**：完美兼容本地开发环境（Jupyter/Python）与 Kaggle 线上竞赛提交环境。

## 2. 环境依赖

项目运行需要满足以下环境要求：

### 核心运行时
- **Python**: 3.10 及以上版本
- **操作系统**: Windows / Linux / macOS (推荐 Windows 以匹配当前开发配置)

### 第三方库依赖
| 库名称 | 建议版本 | 说明 |
| :--- | :--- | :--- |
| `pandas` | ^2.0.0 | 数据处理与分析 |
| `numpy` | ^1.24.0 | 矩阵运算与数值计算 |
| `xgboost` | ^1.7.0 | 梯度提升回归模型 |
| `scikit-learn` | ^1.2.0 | 评估指标计算与预处理 |
| `scipy` | ^1.10.0 | 三次样条插值校准 |
| `statsmodels` | ^0.14.0 | 广义线性模型 (GLM) 建模 |
| `tqdm` | ^4.65.0 | 进度条展示 |
| `python-dotenv`| ^1.0.0 | 环境变量管理 |

## 3. 安装部署步骤

### 3.1 本地开发环境安装
1. **克隆仓库**
   ```bash
   git clone <repository_url>
   cd march_mania
   ```

2. **创建虚拟环境 (推荐)**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   *(注：若无 requirements.txt，可手动执行 `pip install pandas numpy xgboost scikit-learn scipy statsmodels tqdm python-dotenv`)*

### 3.2 数据准备
1. 在项目根目录下创建 `data/raw` 目录。
2. 将竞赛原始 CSV 文件（如 `MRegularSeasonDetailedResults.csv`, `MNCAATourneySeeds.csv` 等）放入该目录。

### 3.3 运行预测
直接运行入口主函数：
```bash
python main.py
```

## 4. 功能说明

### 4.1 数据加载 ([data_loader.py](file:///d:/codes/march_mania/march_mania/src/data_loader.py))
- **动态路径识别**：自动判断当前是 Kaggle 环境还是本地环境，切换数据读取路径。
- **数据集集成**：整合男篮与女篮的历史常规赛与锦标赛详尽数据。

### 4.2 特征工程 ([features.py](file:///d:/codes/march_mania/march_mania/src/features.py))
- **数据平滑**：针对加时赛进行统计修正，还原标准 40 分钟表现。
- **实力评估**：
    - **Elo 系统**：实现动态 K-Factor 的等级分计算。
    - **GLM Quality**：利用广义线性模型从分差中提取球队绝对实力。
    - **Adjusted Stats**：基于赛程强度（SOS）调整后的攻防效率指标。

### 4.3 模型训练与预测 ([models.py](file:///d:/codes/march_mania/march_mania/src/models.py))
- **分差回归**：使用 XGBoost 预测 T1 与 T2 的得分差。
- **概率校准**：通过 `UnivariateSpline` 将预测分差映射为 0-1 胜率，显著提升预测的 Brier Score。

## 5. 目录结构说明

```text
march_mania/
├── data/                   # 数据存放目录
│   ├── raw/                # 原始 CSV 数据 (需手动下载)
│   └── local_small/        # 样例/测试小规模数据
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── config.py           # 项目配置与超参数管理
│   ├── data_loader.py      # 数据加载模块
│   ├── features.py         # 特征工程与数据预处理
│   ├── models.py           # 模型训练、校准与预测逻辑
│   └── download_submission.py # 提交文件生成模块
├── submissions/            # 预测结果输出目录
├── main.py                 # 项目入口主函数
├── PROJECT_DOCUMENTATION.md# 详尽的技术文档 (简历展示版)
└── README.md               # 本文档
```

## 6. 贡献指南

1. **Fork 本项目** 并创建您的特性分支 (`git checkout -b feature/AmazingFeature`)。
2. **提交更改**：请确保代码符合 PEP 8 规范，并附带必要的注释。
3. **推送分支**：`git push origin feature/AmazingFeature`。
4. **发起 Pull Request**：请详细描述您的更改内容及测试情况。

## 7. 常见问题 (FAQ)

**Q: 为什么在计算 Elo 时 K-Factor 是动态的？**
A: 为了响应球队实力的时序演变，项目对较远赛季的 K-Factor 进行了指数衰减处理，使模型更关注近期表现。

**Q: 运行 main.py 时提示找不到 data 目录？**
A: 请确保您已按照 [3.2 数据准备](#32-数据准备) 的要求手动创建了目录并放置了数据文件。

**Q: 如何调整 XGBoost 的超参数？**
A: 请直接修改 [src/config.py](file:///d:/codes/march_mania/march_mania/src/config.py) 中的 `param` 字典。

---
*License: MIT | Created for March Madness Analytics*
