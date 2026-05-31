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

# 由于更加早前的数据不可信，所以我们使用来自2003年之后的数据
def process_after_year(regular_results, tourney_results, seeds,season=2003):
    regular_results = regular_results.loc[regular_results["Season"] >= season].reset_index(drop=True)
    tourney_results = tourney_results.loc[tourney_results["Season"] >= season].reset_index(drop=True)
    seeds = seeds.loc[seeds["Season"] >= season].reset_index(drop=True)
    return regular_results, tourney_results, seeds

# 平衡正负，避免模型过拟合
def prepare_dataset(df):
    # 提取需要的列并重命名
    df = df[["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT",
             "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
             "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",'WLoc']]
    
    # 调整加时赛数据
    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = ["LScore", "WScore",
               "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
               "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]
    for col in adjcols:
        df[col] = df[col] / adjot
        
    # 复制并交换 T1/T2，实现数据双倍化（让模型能预测 T1 输赢）
    df.rename(columns={'WLoc': 'WIs_Home'}, inplace=True)
    df['LIs_Home'] = df['WIs_Home']
    dfswap = df.copy()
    df.columns = [x.replace('W', 'T1_').replace('L', 'T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L', 'T1_').replace('W', 'T2_') for x in list(dfswap.columns)]
    result = pd.concat([df, dfswap]).reset_index(drop=True)
    #篮球进攻回合数（Possessions）估算
    result['T1_Poss'] = result['T1_FGA']-result['T1_OR']+result['T1_TO']+0.475*result['T1_FTA']
    result['T2_Poss'] = result['T2_FGA']-result['T2_OR']+result['T2_TO']+0.475*result['T2_FTA']
    
    
    result['PointDiff'] = result['T1_Score'] - result['T2_Score']
    result['Win'] = (result["PointDiff"] > 0) * 1
    result['Gender'] = (result['T1_TeamID'].apply(lambda t: str(t).startswith("1"))) * 1
    result['Win_Is_Home'] = result['T1_Is_Home']
    df.drop(columns=['T1_Is_Home', 'T2_Is_Home'], inplace=True)
    ## 0: women, 1: men
    return result

# 合并锦标赛数据和种子数据
# 由于种子数据只有锦标赛数据只合并锦标赛数据和种子数据
def merge_seeds_and_tourney(tourney_data ,seeds_):
    seeds = seeds_.copy()
    seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
    seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
    seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
    seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

    tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]
    return tourney_data


# 从常规赛得到的每个队伍基于攻防效率的“调整后场均数据”加权平均的特征，
# 并merge到锦标赛数据上面
def _get_mean_regular_data(regular_data):
    # 1. 提取基础统计列名
    score_columns = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff"
    ]
    base_stats = [c.replace("T1_", "") for c in score_columns if c.startswith("T1_") and c != "PointDiff"]
    # 2. 球队自身场均表现（进攻/效率）
    team_mean = regular_data.groupby(["Season", "T1_TeamID"])[[f"T1_{s}" for s in base_stats]].mean().reset_index()
    # 迭代的初始值
    adj_features = team_mean.copy()
    
    # 3. 该队允许对手打出的场均表现（即赛程强度/防守偏差源）
    allow_mean = regular_data.groupby(["Season", "T1_TeamID"])[[f"T2_{s}" for s in base_stats]].mean().reset_index()
    rename_dict = {col: f"Allow_{col.replace('T2_', '')}" for col in allow_mean.columns if col.startswith("T2_")}
    allow_mean.rename(columns=rename_dict, inplace=True)

    # 4. 联盟各赛季场均允许基准值（用于恢复原始量纲）
    league_allow_mean = regular_data.groupby("Season")[[f"T2_{s}" for s in base_stats]].mean().reset_index()
    rename_dict = {col: f"League_{col.replace('T2_', '')}" for col in league_allow_mean.columns if col.startswith("T2_")}
    league_allow_mean.rename(columns=rename_dict, inplace=True)

    # 5. 横向拼接
    merged = team_mean.merge(allow_mean, on=["Season", "T1_TeamID"], how="left").merge(league_allow_mean, on="Season", how="left")

    # 6. 核心公式计算：Adj = 自身均值 - 允许均值 + 联盟基准
    adj_features = merged[["Season", "T1_TeamID"]].copy()
    for s in base_stats:
        adj_features[f"T1_Adj_{s}"] = merged[f"T1_{s}"] - merged[f"Allow_{s}"] + merged[f"League_{s}"]


    # 🔑 PointDiff 单独处理：保留原始均值（相对指标不宜套用绝对值调整公式）
    pd_mean = regular_data.groupby(["Season", "T1_TeamID"])["PointDiff"].mean().reset_index()
    pd_mean = pd_mean.rename(columns={"PointDiff": "T1_Adj_PointDiff"})
    adj_features = adj_features.merge(pd_mean, on=["Season", "T1_TeamID"], how="left")
    
    # 7. 构造 T1 / T2 侧合并表（保持与你原代码一致的命名风格）
    reg_adj_T1 = adj_features.copy()
    reg_adj_T1.columns = reg_adj_T1.columns.str.replace("T1_Adj_", "T1_avg_", regex=False)
    reg_adj_T2 = adj_features.copy()
    
    reg_adj_T2.columns = reg_adj_T2.columns.str.replace("T1_Adj_", "T2_avg_", regex=False)
    reg_adj_T2.rename(columns={"T1_TeamID": "T2_TeamID"}, inplace=True)
    return reg_adj_T1, reg_adj_T2

def get_regular_mean(tourney_data ,regular_data):
    regular_data_mean_T1, regular_data_mean_T2 = _get_mean_regular_data(regular_data)
    tourney_data = pd.merge(tourney_data, regular_data_mean_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, regular_data_mean_T2, on=["Season", "T2_TeamID"], how="left")
    return tourney_data

# using ELO rating system
# 经典的 Elo 动态等级分系统
def _update_elo(winner_elo, loser_elo):
    expected_win = _expected_result(winner_elo, loser_elo)
    change_in_elo = config.k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def _expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / config.elo_width))

def get_elo_rating(tourney_data_, regular_data):
    tourney_data = tourney_data_.copy()
    base_elo = config.base_elo
    elo_width = config.elo_width
    k_factor = config.k_factor

    elos = []
    for season in tourney_data["Season"].unique():
        current_data = regular_data.loc[regular_data['Season']==season]
        current_data = current_data.loc[current_data['Win']==1].reset_index(drop=True)
        all_teams = set(current_data['T1_TeamID']) | set(current_data['T2_TeamID'])
        elo = dict(zip(all_teams, [base_elo] * len(all_teams)))
        for i in range(len(current_data)):
            w_team, l_team = current_data.loc[i, "T1_TeamID"], current_data.loc[i, "T2_TeamID"]
            w_elo, l_elo = elo[w_team], elo[l_team]
            w_elo_new, l_elo_new = _update_elo(w_elo, l_elo)
            elo[w_team] = w_elo_new
            elo[l_team] = l_elo_new
        elo = pd.DataFrame.from_dict(elo, orient="index").reset_index()
        elo = elo.rename({"index": "TeamID", 0: "elo"}, axis=1)
        elo["Season"] = season
        elos.append(elo)
    elos = pd.concat(elos)
    elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
    elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
    tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]
    return tourney_data


# 利用常规赛的“比分差（PointDiff）”，
# 通过广义线性模型（GLM）为每支球队计算出一个“绝对实力评分（Quality）”，
# 最后把交战双方的实力差（diff_quality）作为预测比赛胜负的核心特征。
import statsmodels.api as sm
import tqdm

def get_glm_quality_features(tourney_data, regular_data, seeds):
    # 1. 使用 copy 防止修改外部的原始 DataFrame
    reg_df = regular_data.copy()
    tourney_df = tourney_data.copy()
    
    # 2. 构造唯一的 赛季/球队 标识
    # 相比 apply lambda，直接用 astype 和 + 拼接在 Pandas 中速度快 10 倍以上
    reg_df['ST1'] = reg_df['Season'].astype(str) + '/' + reg_df['T1_TeamID'].astype(str)
    reg_df['ST2'] = reg_df['Season'].astype(str) + '/' + reg_df['T2_TeamID'].astype(str)
    
    # 获取所有进入锦标赛的球队集合 (直接从原始 seeds 表获取，不需要提前 split)
    st = set(seeds['Season'].astype(str) + '/' + seeds['TeamID'].astype(str))

    # 扩展集合：加入那些在常规赛击败过锦标赛球队的队伍
    st = st | set(reg_df.loc[(reg_df["T1_Score"] > reg_df["T2_Score"]) & 
                             (reg_df["ST2"].isin(st)), "ST1"])

    # 3. 构建用于建模的 dt 数据集 (降维处理)
    dt = reg_df.loc[reg_df["ST1"].isin(st) | reg_df["ST2"].isin(st)].copy()
    dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
    dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
    dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
    dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

    # 4. 定义内部建模函数 (利用闭包直接读取作用域内的 dt)
    def _team_quality(season, gender_flag):
        model_data = dt.loc[(dt["Season"] == season) & (dt["Gender"] == gender_flag), :]
        if model_data.empty:
            return pd.DataFrame() # 防止某些早期赛季没有女篮数据时报错

        formula = "PointDiff~-1+T1_TeamID+T2_TeamID"
        glm = sm.GLM.from_formula(
            formula=formula,
            data=model_data,
            family=sm.families.Gaussian(),
        ).fit()
        
        quality = pd.DataFrame(glm.params).reset_index()
        quality.columns = ["TeamID", "quality"]
        quality["Season"] = season
        
        # 过滤出 T1 的系数
        quality = quality.loc[quality['TeamID'].str.contains("T1_")].reset_index(drop=True)
        
        quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
        
        return quality

    # 5. 循环计算每个赛季的评分
    glm_quality_list = []
    seasons = sorted(set(seeds["Season"]))
    
    for s in tqdm.tqdm(seasons, desc="Calculating GLM Quality", unit="season"):
        if s >= 2010:  # min season for women
            glm_quality_list.append(_team_quality(s, 0))
        if s >= 2003:  # min season for men
            glm_quality_list.append(_team_quality(s, 1))

    # 6. 整理生成的评分表
    glm_quality = pd.concat(glm_quality_list, ignore_index=True)

    glm_quality_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
    glm_quality_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})

    # 7. 合并到锦标赛对阵表中
    tourney_df = pd.merge(tourney_df, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_df = pd.merge(tourney_df, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
    
    # 计算差值
    tourney_df["diff_quality"] = tourney_df["T1_quality"] - tourney_df["T2_quality"]

    return tourney_df
