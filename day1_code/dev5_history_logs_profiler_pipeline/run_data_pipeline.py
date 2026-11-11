import os
import sys

from config import config
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset
from data_pipeline.history_cleaning import clean_history_dataset
from data_pipeline.quality_profiler import (profile_dataset, profile_history_details, profile_log_details,
                                            profile_sentence_details, write_quality_report)
from data_pipeline.repair_log_cleaning import clean_repair_log_dataset
from data_pipeline.sentence_cleaning import clean_sentence_dataset

DATASET_JOBS = [
    (config.RAW_REPORT_HISTORY_TRAIN_FILE, config.CLEAN_REPORT_HISTORY_TRAIN_FILE, clean_history_dataset,
     config.HISTORY_TARGET_COLUMN, profile_history_details),
    (config.RAW_REPORT_HISTORY_EVAL_FILE, config.CLEAN_REPORT_HISTORY_EVAL_FILE, clean_history_dataset,
     config.HISTORY_TARGET_COLUMN, profile_history_details),
    (config.RAW_REPAIR_LOGS_TRAIN_FILE, config.CLEAN_REPAIR_LOGS_TRAIN_FILE, clean_repair_log_dataset,
     config.LOG_TARGET_COLUMN, profile_log_details),
    (config.RAW_REPAIR_LOGS_EVAL_FILE, config.CLEAN_REPAIR_LOGS_EVAL_FILE, clean_repair_log_dataset,
     config.LOG_TARGET_COLUMN, profile_log_details),
    (config.RAW_STATEMENT_SENTENCES_FILE, config.CLEAN_STATEMENT_SENTENCES_FILE, clean_sentence_dataset,
     config.SENTENCE_TARGET_COLUMN, profile_sentence_details),
]


def main():
    cleaned, profiles = [], {}
    for raw_file, clean_file, clean, target_column, profile_details in DATASET_JOBS:
        raw = load_raw_dataset(raw_file)
        clean_frame = clean(raw)
        name = os.path.splitext(os.path.basename(raw_file))[0]
        profiles[name] = {**profile_dataset(raw, clean_frame, target_column), **profile_details(raw, clean_frame)}
        cleaned.append((clean_file, clean_frame))
    for clean_file, clean_frame in cleaned:
        save_processed_dataset(clean_frame, clean_file)
    write_quality_report(profiles)
    for name, profile in profiles.items():
        print(f"{name:<{config.DATASET_NAME_WIDTH}} {profile['rows_received']:>{config.ROW_COUNT_WIDTH}} received -> "
              f"{profile['rows_kept']:>{config.ROW_COUNT_WIDTH}} kept  ({profile['rows_removed']} removed)")
    print(f"quality report written to {config.DATA_QUALITY_REPORT_FILE}")


if __name__ == "__main__":
    sys.exit(main())
