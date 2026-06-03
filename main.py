
"""
March Mania: NCAA Tournament Prediction Main Entry Point
This script executes the end-to-end pipeline:
1. Load datasets (Regular season, Tourney, Seeds)
2. Feature Engineering (Elo, GLM, Adjusted Stats)
3. Model Training & Probability Calibration
4. Generate & Save Submission
"""

from src import data_loader, features, models, config, download_submission

def run_pipeline():
    # 1. Load data
    print("Loading datasets...")
    regular_results, tourney_results, seeds = data_loader.get_main_datasets(full_data=True)
    submission = data_loader.get_submission_datasets()

    # 2. Feature engineering
    print("Processing features and engineering indicators...")
    regular_data, tourney_data, seeds, submission = features.main_processed_datasets(
        regular_results, tourney_results, seeds, submission
    )

    # 3. Model training and prediction
    print("Training models and generating predictions...")
    submission = models.main_models(tourney_data, submission)

    # 4. Download/Save results
    print("Saving submission to submissions/predictions.csv...")
    download_submission.main_download_submission(submission)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
