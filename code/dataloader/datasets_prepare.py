from get_basic_info import *
from local_datasets_downloader import *
from pathlib import Path
import pandas as pd
def get_main_datasets():
    if is_kaggle():
        input_dir = Path("/kaggle/input/competitions/march-machine-learning-mania-2026")
        W_seeds = pd.read_csv(input_dir / 'WNCAATourneySeeds.csv')
        M_seeds = pd.read_csv(input_dir / 'MNCAATourneySeeds.csv')

        M_regular_results = pd.read_csv(input_dir / 'MRegularSeasonDetailedResults.csv')
        W_regular_results = pd.read_csv(input_dir / 'WRegularSeasonDetailedResults.csv')

        M_tourney_results = pd.read_csv(input_dir / 'MNCAATourneyDetailedResults.csv')
        W_tourney_results = pd.read_csv(input_dir / 'WNCAATourneyDetailedResults.csv')
        regular_results = pd.concat([M_regular_results, W_regular_results])
        tourney_results = pd.concat([M_tourney_results, W_tourney_results])
        seeds = pd.concat([M_seeds, W_seeds])
        return regular_results, tourney_results, seeds
    else:
        input_dir = Path('./local_data')
        W_seeds = pd.read_csv(input_dir / 'WNCAATourneySeeds.csv')
        M_seeds = pd.read_csv(input_dir / 'MNCAATourneySeeds.csv')

        M_regular_results = pd.read_csv(input_dir / 'MRegularSeasonDetailedResults.csv')
        W_regular_results = pd.read_csv(input_dir / 'WRegularSeasonDetailedResults.csv')

        M_tourney_results = pd.read_csv(input_dir / 'MNCAATourneyDetailedResults.csv')
        W_tourney_results = pd.read_csv(input_dir / 'WNCAATourneyDetailedResults.csv')
        regular_results = pd.concat([M_regular_results, W_regular_results])
        tourney_results = pd.concat([M_tourney_results, W_tourney_results])
        seeds = pd.concat([M_seeds, W_seeds])
        return regular_results, tourney_results, seeds
    
# 由于更加早前的数据不可信，所以我们使用来自2003年之后的数据
def process_after_year( regular_results, tourney_results, seeds,season=2003):
    regular_results = regular_results.loc[regular_results["Season"] >= season].reset_index(drop=True)
    tourney_results = tourney_results.loc[tourney_results["Season"] >= season].reset_index(drop=True)
    seeds = seeds.loc[seeds["Season"] >= season].reset_index(drop=True)
    massey = massey.loc[massey['Season']>=season].reset_index(drop=True)
    return regular_results, tourney_results, seeds

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
    ## 0: women, 1: men
    result['Is_Home_Win'] = result['T1_T2_oc']
    result.drop(columns=['T1_T2_oc', 'T2_T1_oc'], inplace=True)
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

if __name__ == '__main__':
    regular_results, tourney_results, seeds=get_main_datasets()
    print(regular_results.head())
