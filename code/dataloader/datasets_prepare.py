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
    
if __name__ == '__main__':
    regular_results, tourney_results, seeds=get_main_datasets()
    print(regular_results.head())
