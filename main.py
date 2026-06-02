
from src import data_loader, features, models, config, download_submission

regular_results, tourney_results, seeds = data_loader.get_main_datasets(full_data=True)
submission = data_loader.get_submission_datasets()

regular_data, tourney_data, seeds, submission = features.main_processed_datasets(regular_results, tourney_results, seeds, submission)

submission = models.main_models(tourney_data, submission)

download_submission.main_download_submission(submission)
