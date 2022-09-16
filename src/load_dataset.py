# Set up Kaggle API credentials: https://github.com/Kaggle/kaggle-api#api-credentials
import os

from kaggle.api.kaggle_api_extended import KaggleApi

from src.utils import get_base_path

DATASET_NAME = "fedesoriano/heart-failure-prediction"
DATASET_FOLDER = os.path.join(get_base_path(), "data")


def load_dataset():
    """Download the dataset via Kaggle API."""
    api = KaggleApi()
    api.authenticate()
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    api.dataset_download_files(
        dataset=DATASET_NAME,
        path=DATASET_FOLDER,
        unzip=True,
    )
