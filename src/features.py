import pandas as pd
import numpy as np

# 检测当前是否在 Jupyter Notebook 环境中运行
import sys
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


# 从submssion中获得交手的队伍
def get_submission_teams(submission_df):
    submission_df['T1_TeamID']=submission_df['ID'].apply(lambda t: int(t.split('_')[1]))
    submission_df['T2_TeamID']=submission_df['ID'].apply(lambda t: int(t.split('_')[2]))
    return submission_df

# 由于更加早前的数据不可信，所以我们使用来自2003年之后的数据
def process_after_year(regular_results, tourney_results, seeds,season=2003):
    regular_results = regular_results.loc[regular_results["Season"] >= season].reset_index(drop=True)
    tourney_results = tourney_results.loc[tourney_results["Season"] >= season].reset_index(drop=True)
    seeds = seeds.loc[seeds["Season"] >= season].reset_index(drop=True)
    seeds['Season'] = seeds['Season']
    regular_results['Season'] = regular_results['Season']
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
    return tourney_data, seeds_T1, seeds_T2


# 从常规赛得到的每个队伍基于攻防效率的“调整后场均数据”加权平均的特征，
# 并merge到锦标赛数据上面

import pandas as pd

def add_regular_season_means(tourney_data, regular_data):
    """
    将常规赛的场均数据合并到锦标赛对阵数据中。
    
    参数:
    tourney_data: DataFrame, 锦标赛(Tourney)对阵数据。
    regular_data: DataFrame, 常规赛(Regular Season)数据。
    
    返回:
    DataFrame: 包含场均统计特征的新锦标赛数据集。
    """
    # 使用 copy() 避免修改传入的原始数据集
    tourney_df = tourney_data.copy()
    
    # 定义需要计算均值的特征列
    score_columns = [
        "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
        "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
        "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
        "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
        "PointDiff",'T1_Poss','T2_Poss'
    ]
    
    # 计算每个赛季每支球队的场均数据
    regular_data_mean = regular_data.groupby(["Season", "T1_TeamID"])[score_columns].mean().reset_index()
    
    # ---------------- 提取 T1 (球队1) 的特征 ----------------
    regular_data_mean_T1 = regular_data_mean.copy()
    # 重命名列名，添加 T1_avg 前缀，并将对方的特征标记为 opponent
    regular_data_mean_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") 
                                    for x in list(regular_data_mean_T1.columns)]
    # 恢复主键的原始名称，以便后续 merge
    regular_data_mean_T1 = regular_data_mean_T1.rename(
        columns={"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}
    )
    
    # ---------------- 提取 T2 (球队2) 的特征 ----------------
    regular_data_mean_T2 = regular_data_mean.copy()
    # 重命名列名，添加 T2_avg 前缀，并将对方的特征标记为 opponent
    regular_data_mean_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") 
                                    for x in list(regular_data_mean_T2.columns)]
    # 恢复主键的原始名称，注意这里将 T2_avg_TeamID 映射回 T2_TeamID
    regular_data_mean_T2 = regular_data_mean_T2.rename(
        columns={"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}
    )
    
    # ---------------- 合并回主表 ----------------
    tourney_df = pd.merge(tourney_df, regular_data_mean_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_df = pd.merge(tourney_df, regular_data_mean_T2, on=["Season", "T2_TeamID"], how="left")
    
    return tourney_df, regular_data_mean_T1, regular_data_mean_T2

def _get_mean_regular_data(regular_data):
    # 提取基础统计列名
    score_columns = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",'T1_Poss','T2_Poss'
    ]
    base_stats = [c.replace("T1_", "") for c in score_columns if c.startswith("T1_") and c != "PointDiff"]
    # 球队自身场均表现（进攻/效率）
    team_mean = regular_data.groupby(["Season", "T1_TeamID"])[[f"T1_{s}" for s in base_stats]].mean().reset_index()
    adj_features = team_mean.copy()
    
    # 该队允许对手打出的场均表现（即赛程强度/防守偏差源）
    allow_mean = regular_data.groupby(["Season", "T1_TeamID"])[[f"T2_{s}" for s in base_stats]].mean().reset_index()
    rename_dict = {col: f"Allow_{col.replace('T2_', '')}" for col in allow_mean.columns if col.startswith("T2_")}
    allow_mean.rename(columns=rename_dict, inplace=True)

    # 联盟各赛季场均允许基准值（用于恢复原始量纲）
    league_allow_mean = regular_data.groupby("Season")[[f"T2_{s}" for s in base_stats]].mean().reset_index()
    rename_dict = {col: f"League_{col.replace('T2_', '')}" for col in league_allow_mean.columns if col.startswith("T2_")}
    league_allow_mean.rename(columns=rename_dict, inplace=True)

    merged = team_mean.merge(allow_mean, on=["Season", "T1_TeamID"], how="left").merge(league_allow_mean, on="Season", how="left")

    # 核心公式计算：Adj = 自身均值 - 允许均值 + 联盟基准
    adj_features = merged[["Season", "T1_TeamID"]].copy()
    for s in base_stats:
        adj_features[f"T1_Adj_{s}"] = merged[f"T1_{s}"] - merged[f"Allow_{s}"] + merged[f"League_{s}"]


    #PointDiff 单独处理
    pd_mean = regular_data.groupby(["Season", "T1_TeamID"])["PointDiff"].mean().reset_index()
    pd_mean = pd_mean.rename(columns={"PointDiff": "T1_Adj_PointDiff"})
    adj_features = adj_features.merge(pd_mean, on=["Season", "T1_TeamID"], how="left")
    
    # 构造 T1 / T2 侧合并表
    reg_adj_T1 = weight_pass_mean_corrected(adj_features)
    reg_adj_T1.columns = reg_adj_T1.columns.str.replace("T1_Adj_", "T1_avg_", regex=False)
    reg_adj_T2 = adj_features.copy()
    
    reg_adj_T2.columns = reg_adj_T2.columns.str.replace("T1_Adj_", "T2_avg_", regex=False)
    reg_adj_T2.rename(columns={"T1_TeamID": "T2_TeamID"}, inplace=True)


    return reg_adj_T1, reg_adj_T2

def _get_mean_tourney_data(tourney_data):

    # 提取基础统计列名
    score_columns = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",'T1_Poss','T2_Poss'
    ]
    tourney_data = tourney_data.copy()
    base_stats = [c.replace("T1_", "") for c in score_columns if c.startswith("T1_") and c != "PointDiff"]
    # 球队自身场均表现（进攻/效率）
    team_mean = tourney_data.groupby(["Season", "T1_TeamID"])[[f"T1_{s}" for s in base_stats]].mean().reset_index()
    adj_features = team_mean.copy()
    
    # 该队允许对手打出的场均表现（即赛程强度/防守偏差源）
    allow_mean = tourney_data.groupby(["Season", "T1_TeamID"])[[f"T2_{s}" for s in base_stats]].mean().reset_index()
    rename_dict = {col: f"Allow_{col.replace('T2_', '')}" for col in allow_mean.columns if col.startswith("T2_")}
    allow_mean.rename(columns=rename_dict, inplace=True)

    # 联盟各赛季场均允许基准值（用于恢复原始量纲）
    league_allow_mean = tourney_data.groupby("Season")[[f"T2_{s}" for s in base_stats]].mean().reset_index()
    rename_dict = {col: f"League_{col.replace('T2_', '')}" for col in league_allow_mean.columns if col.startswith("T2_")}
    league_allow_mean.rename(columns=rename_dict, inplace=True)

    merged = team_mean.merge(allow_mean, on=["Season", "T1_TeamID"], how="left").merge(league_allow_mean, on="Season", how="left")

    # 核心公式计算：Adj = 自身均值 - 允许均值 + 联盟基准
    adj_features = merged[["Season", "T1_TeamID"]].copy()
    for s in base_stats:
        adj_features[f"T1_Adj_{s}"] = merged[f"T1_{s}"] - merged[f"Allow_{s}"] + merged[f"League_{s}"]


    #PointDiff 单独处理
    pd_mean = tourney_data.groupby(["Season", "T1_TeamID"])["PointDiff"].mean().reset_index()
    pd_mean = pd_mean.rename(columns={"PointDiff": "T1_Adj_PointDiff"})
    adj_features = adj_features.merge(pd_mean, on=["Season", "T1_TeamID"], how="left")
    
    # 构造 T1 / T2 侧合并表
    reg_adj_T1 = weight_pass_mean_corrected(adj_features)
    
    reg_adj_T1.columns = reg_adj_T1.columns.str.replace("T1_Adj_", "T1_tour_avg_", regex=False)
    reg_adj_T2 = adj_features.copy()
    
    reg_adj_T2.columns = reg_adj_T2.columns.str.replace("T1_Adj_", "T2_tour_avg_", regex=False)
    reg_adj_T2.rename(columns={"T1_TeamID": "T2_TeamID"}, inplace=True)
    return reg_adj_T1, reg_adj_T2


def weight_pass_mean_corrected(reg_adj):
    # 避免修改原始数据
    reg_weighted = reg_adj.copy()
    
    # 动态获取列名
    id_col_name = [c for c in reg_weighted.columns if 'TeamID' in c][0]
    cols = [c for c in reg_weighted.columns if '_avg_' in c]
    
    # 获取全局所有的赛季集合
    seasons = reg_weighted['Season'].unique()
    
    # 准备一个列表，用来存放需要新增的“填补行”
    new_rows = []
    
    # 按队伍分组，大幅降低条件筛选的计算开销
    for team_id, team_group in reg_weighted.groupby(id_col_name):
        cur_seasons = team_group['Season'].values
        
        for season in seasons:
            # 如果当前队伍已经有这个赛季的数据，直接跳过
            if season in cur_seasons:
                continue
                
            # 找到最近的历史赛季
            need_season = get_nearest_season(season, cur_seasons)
            
            # 【选项 A 逻辑实现】如果之前没有任何历史赛季，直接跳过，不强行填补
            if need_season is None:
                continue
            
            # 提取最近历史赛季的数据行
            nearest_row = team_group[team_group['Season'] == need_season].iloc[0].copy()
            
            # 将提取出的数据行的赛季标为当前缺失的赛季
            nearest_row['Season'] = season
            
            # 将新构建的行存入列表
            new_rows.append(nearest_row)
            
    # 如果列表非空，一次性合并回主表
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        reg_weighted = pd.concat([reg_weighted, new_df], ignore_index=True)
        
    return reg_weighted
        
def get_nearest_season(season, cur_seasons):
    valid_nums = [x for x in cur_seasons if x < season]
    return max(valid_nums, default=None)


def get_regular_mean(tourney_data ,regular_data):
    regular_data_mean_T1, regular_data_mean_T2 = _get_mean_regular_data(regular_data)
    tourney_data = pd.merge(tourney_data, regular_data_mean_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, regular_data_mean_T2, on=["Season", "T2_TeamID"], how="left")
    return tourney_data, regular_data_mean_T1, regular_data_mean_T2

def get_tourney_mean(tourney_data):
    tourney_data_mean_T1, tourney_data_mean_T2 = _get_mean_tourney_data(tourney_data)
    tourney_data_mean_T1_begin = tourney_data_mean_T1[tourney_data_mean_T1['Season']==2003].copy()
    tourney_data_mean_T2_begin = tourney_data_mean_T2[tourney_data_mean_T2['Season']==2003].copy()
    tourney_data_mean_T1['Season'] = tourney_data_mean_T1['Season']+1
    tourney_data_mean_T2['Season'] = tourney_data_mean_T2['Season']+1
    tourney_data_mean_T1 = pd.concat([tourney_data_mean_T1_begin, tourney_data_mean_T1], ignore_index=True)
    tourney_data_mean_T2 = pd.concat([tourney_data_mean_T2_begin, tourney_data_mean_T2], ignore_index=True)
    tourney_data = pd.merge(tourney_data, tourney_data_mean_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = pd.merge(tourney_data, tourney_data_mean_T2, on=["Season", "T2_TeamID"], how="left")
    return tourney_data, tourney_data_mean_T1, tourney_data_mean_T2

# using ELO rating system
# 经典的 Elo 动态等级分系统在常规赛中
def _update_elo(winner_elo, loser_elo, k_factor):
    expected_win = _expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def _expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / config.elo_width))

def get_elo_rating(tourney_data_, regular_data, ranges):
    tourney_data = tourney_data_.copy()
    base_elo = config.base_elo
    elo_width = config.elo_width
    k_factor = config.k_factor

    elos = []
    for season in tourney_data["Season"].unique():
        current_data = regular_data.loc[(regular_data['Season'] <= season) & (regular_data['Season'] >= season - ranges)]
        current_data = current_data.loc[current_data['Win']==1].reset_index(drop=True)
        all_teams = set(current_data['T1_TeamID']) | set(current_data['T2_TeamID'])
        elo = dict(zip(all_teams, [base_elo] * len(all_teams)))
        for i in range(len(current_data)):
            w_team, l_team = current_data.loc[i, "T1_TeamID"], current_data.loc[i, "T2_TeamID"]
            w_elo, l_elo = elo[w_team], elo[l_team]
            cur_season = current_data.loc[i,'Season']
            k_factor = config.k_factor * season_weight(cur_season, season, ranges+1)
            w_elo_new, l_elo_new = _update_elo(w_elo, l_elo, k_factor)
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
    elos_T1['Season'] = elos_T1['Season']+1
    elos_T2['Season'] = elos_T2['Season']+1
    return tourney_data, elos,elos_T1, elos_T2

def season_weight(curren_season, root_season, num_seasons):
    weight = np.exp(-(abs(curren_season-root_season)/num_seasons))
    return weight if weight>0.05 else 0.05

def modify_elo_rating_from_tourney(tourney_data_, elos_, ranges):
    tourney_data_ = tourney_data_.copy()
    # 在get_elo_rating之后
    elos_term = elos_.copy()
    tourney_data = tourney_data_.copy()
    base_elo = config.base_elo
    elo_width = config.elo_width
    k_factor = config.k_factor

    tourney_data_begin = tourney_data[tourney_data['Season']==2003].copy()
    tourney_data['Season']=tourney_data['Season']+1
    tourney_data = pd.concat([tourney_data_begin, tourney_data], ignore_index=True)

    elos_term_begin = elos_term[elos_term['Season']==2003].copy()
    elos_term['Season']=elos_term['Season']+1
    elos_term = pd.concat([elos_term_begin, elos_term], ignore_index=True)


    elos = []
    for season in tourney_data["Season"].unique():
        current_data = tourney_data.loc[(tourney_data['Season'] <= season) & (tourney_data['Season'] >= season - ranges)]
        current_data = current_data.loc[current_data['Win']==1].reset_index(drop=True)
        elo = elos_term.loc[elos_term['Season']==season].set_index("TeamID").to_dict()["elo"]
        for i in range(len(current_data)):
            w_team, l_team = current_data.loc[i, "T1_TeamID"], current_data.loc[i, "T2_TeamID"]
            w_elo, l_elo = elo[w_team], elo[l_team]
            current_season = current_data.loc[i,'Season']
            k_factor = config.k_factor * season_weight(current_season, season, ranges+1)
            w_elo_new, l_elo_new = _update_elo(w_elo, l_elo, k_factor)
            elo[w_team] = w_elo_new
            elo[l_team] = l_elo_new
        elo = pd.DataFrame.from_dict(elo, orient="index").reset_index()
        elo = elo.rename({"index": "TeamID", 0: "elo"}, axis=1)
        elo["Season"] = season
        elos.append(elo)
    elos = pd.concat(elos)
    elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo_tour"}, axis=1)
    elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo_tour"}, axis=1)
    tourney_data_ = pd.merge(tourney_data_, elos_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data_ = pd.merge(tourney_data_, elos_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data_["elo_diff_tour"] = tourney_data_["T1_elo_tour"] - tourney_data_["T2_elo_tour"]
    return tourney_data_, elos_T1, elos_T2


# 利用常规赛的“比分差（PointDiff）”，
# 通过广义线性模型（GLM）为每支球队计算出一个“绝对实力评分（Quality）”，
# 最后把交战双方的实力差（diff_quality）作为预测比赛胜负的核心特征。
import statsmodels.api as sm
import tqdm

def get_glm_quality_features(tourney_data, regular_data, seeds):
    reg_df = regular_data.copy()
    tourney_df = tourney_data.copy()
    
    #构造唯一的 赛季/球队 标识
    reg_df['ST1'] = reg_df['Season'].astype(str) + '/' + reg_df['T1_TeamID'].astype(str)
    reg_df['ST2'] = reg_df['Season'].astype(str) + '/' + reg_df['T2_TeamID'].astype(str)
    
    st = set(seeds['Season'].astype(str) + '/' + seeds['TeamID'].astype(str))

    # 加入那些在常规赛击败过锦标赛球队的队伍
    st = st | set(reg_df.loc[(reg_df["T1_Score"] > reg_df["T2_Score"]) & 
                             (reg_df["ST2"].isin(st)), "ST1"])

    dt = reg_df.loc[reg_df["ST1"].isin(st) | reg_df["ST2"].isin(st)].copy()
    dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
    dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
    dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
    dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

    # 定义建模函数
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
        
        quality = quality.loc[quality['TeamID'].str.contains("T1_")].reset_index(drop=True)
        
        quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
        
        return quality

    #循环每个赛季
    glm_quality_list = []
    seasons = sorted(set(seeds["Season"]))
    
    for s in tqdm.tqdm(seasons, desc="Calculating GLM Quality", unit="season"):
        if s >= 2010:  # min season for women
            glm_quality_list.append(_team_quality(s, 0))
        if s >= 2003:  # min season for men
            glm_quality_list.append(_team_quality(s, 1))

    glm_quality = pd.concat(glm_quality_list, ignore_index=True)

    glm_quality_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
    glm_quality_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})

    tourney_df = pd.merge(tourney_df, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_df = pd.merge(tourney_df, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
    
    # 计算差值
    tourney_df["diff_quality"] = tourney_df["T1_quality"] - tourney_df["T2_quality"]

    return tourney_df, glm_quality_T1, glm_quality_T2


def process_submission(submission, regular_data_mean_T1, regular_data_mean_T2,tourney_data_mean_T1, tourney_data_mean_T2, seeds_T1, seeds_T2, glm_quality_T1, glm_quality_T2, elos_T1, elos_T2, elos2_T1, elos2_T2):
    submission_df_1 = submission.copy()
    submission_df_1['Season'] = submission_df_1['ID'].apply(lambda t: int(t.split('_')[0]))
    submission_df_1['T1_TeamID'] = submission_df_1['ID'].apply(lambda t: int(t.split('_')[1]))
    submission_df_1['T2_TeamID'] = submission_df_1['ID'].apply(lambda t: int(t.split('_')[2]))
    submission_df_1['Gender'] = submission_df_1['T1_TeamID'].apply(lambda t: 0 if str(t)[0]=='1' else 1)    

    submission_df_1 = pd.merge(submission_df_1, regular_data_mean_T1, on=['Season', 'T1_TeamID'], how = 'left')
    submission_df_1 = pd.merge(submission_df_1, regular_data_mean_T2, on=['Season', 'T2_TeamID'], how = 'left')

    submission_df_1 = pd.merge(submission_df_1, tourney_data_mean_T1, on=['Season', 'T1_TeamID'], how = 'left')
    submission_df_1 = pd.merge(submission_df_1, tourney_data_mean_T2, on=['Season', 'T2_TeamID'], how = 'left')

    submission_df_1 = pd.merge(submission_df_1, seeds_T1, on=['Season', 'T1_TeamID'], how = 'left')
    submission_df_1 = pd.merge(submission_df_1, seeds_T2, on=['Season', 'T2_TeamID'], how = 'left')

    submission_df_1 = pd.merge(submission_df_1, glm_quality_T1, on=['Season', 'T1_TeamID'], how = 'left')
    submission_df_1 = pd.merge(submission_df_1, glm_quality_T2, on=['Season', 'T2_TeamID'], how = 'left')

    submission_df_1 = pd.merge(submission_df_1, elos_T1, on=['Season', 'T1_TeamID'], how = 'left')
    submission_df_1 = pd.merge(submission_df_1, elos_T2, on=['Season', 'T2_TeamID'], how = 'left')

    submission_df_1 = pd.merge(submission_df_1, elos2_T1, on=['Season', 'T1_TeamID'], how = 'left')
    submission_df_1 = pd.merge(submission_df_1, elos2_T2, on=['Season', 'T2_TeamID'], how = 'left')

    submission_df_1['Seed_diff'] = submission_df_1['T2_seed'] - submission_df_1['T1_seed']
    submission_df_1['elo_diff'] = submission_df_1['T1_elo'] - submission_df_1['T2_elo']
    submission_df_1["diff_quality"] = submission_df_1["T1_quality"] - submission_df_1["T2_quality"]
    submission_df_1['elo_diff_tour'] = submission_df_1['T1_elo_tour'] - submission_df_1['T2_elo_tour']
    return submission_df_1

# 得到的最终的训练集数据集
def main_processed_datasets(regular_results, tourney_results, seeds, submission_df):
    submission_df = get_submission_teams(submission_df)
    regular_results, tourney_results, seeds = process_after_year(regular_results, tourney_results, seeds)
    regular_data, tourney_data = prepare_dataset(regular_results), prepare_dataset(tourney_results)
    tourney_data,seeds_T1, seeds_T2=merge_seeds_and_tourney(tourney_data, seeds)

    tourney_data, regular_data_mean_T1, regular_data_mean_T2 = add_regular_season_means(tourney_data ,regular_data)
    tourney_data ,tourney_data_mean_T1, tourney_data_mean_T2= get_tourney_mean(tourney_data)
    
    tourney_data,  elos, elos_T1, elos_T2 = get_elo_rating(tourney_data, regular_data, 2)
    tourney_data,  elos2_T1, elos2_T2 = modify_elo_rating_from_tourney(tourney_data, elos, 2)

    tourney_data ,glm_quality_T1, glm_quality_T2 = get_glm_quality_features(tourney_data, regular_data, seeds)
    
    submission = process_submission(submission_df, regular_data_mean_T1, regular_data_mean_T2,tourney_data_mean_T1, tourney_data_mean_T2, seeds_T1, seeds_T2, glm_quality_T1, glm_quality_T2, elos_T1, elos_T2, elos2_T1, elos2_T2)
    return regular_data, tourney_data, seeds, submission