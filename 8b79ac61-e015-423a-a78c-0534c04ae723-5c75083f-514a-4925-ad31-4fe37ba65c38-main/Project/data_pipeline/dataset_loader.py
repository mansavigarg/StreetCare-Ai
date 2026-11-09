import os

import pandas as pd

from config import config


def load_raw_dataset(file_path):
    frame = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    frame = frame.apply(lambda column: column.str.strip())
    return frame.mask(frame.apply(lambda column: column.str.lower().isin(config.NULL_TOKENS)))


def save_processed_dataset(dataframe, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    dataframe.to_csv(file_path, index=False)


def load_processed_dataset(file_path):
    return pd.read_csv(file_path, keep_default_na=False, na_values=[""])
