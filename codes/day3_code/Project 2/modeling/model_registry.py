import json
import os

import joblib

from config import config


def _make_folder(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def save_model(model, file_path):
    _make_folder(file_path)
    joblib.dump(model, file_path, compress=config.MODEL_FILE_COMPRESSION)


def save_feature_schema(feature_names, file_path):
    _make_folder(file_path)
    with open(file_path, "w", encoding="utf-8") as schema_file:
        json.dump(list(feature_names), schema_file, indent=config.JSON_INDENT)


def load_model(file_path):
    return joblib.load(file_path)


def load_feature_schema(file_path):
    with open(file_path, encoding="utf-8") as schema_file:
        return json.load(schema_file)


def save_metrics(metrics):
    _make_folder(config.MODEL_METRICS_FILE)
    with open(config.MODEL_METRICS_FILE, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=config.JSON_INDENT, sort_keys=True)
