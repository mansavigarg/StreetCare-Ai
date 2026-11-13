import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import config
from tools.screening_tools import find_coordination_triggers, screen_report


def add_requirement_counts(dataframe, standards):
    result = dataframe.copy()
    screenings = [screen_report(report, standards) for report in dataframe.to_dict("records")]
    for column in config.REQUIREMENT_COUNT_RESULTS:
        result[column] = [screening[column] for screening in screenings]
    return result


def add_coordination_trigger_flags(dataframe):
    result = dataframe.copy()
    triggers = [find_coordination_triggers(report) for report in dataframe.to_dict("records")]
    for trigger, column in config.COORDINATION_TRIGGER_FLAG_COLUMNS.items():
        result[column] = [int(trigger in found) for found in triggers]
    return result


def build_feature_frame(dataframe, numeric_columns, categorical_columns):
    features = pd.DataFrame(index=dataframe.index)
    for column in numeric_columns:
        features[column] = pd.to_numeric(dataframe[column]).astype(float)
    for column in categorical_columns:
        values = dataframe[column].astype(object)
        features[column] = values.where(values.notna(), np.nan)
    return features


def build_model_pipeline(estimator, numeric_columns, categorical_columns):
    numbers = Pipeline([("fill", SimpleImputer(strategy=config.NUMBER_FILL_STRATEGY)), ("scale", StandardScaler())])
    categories = Pipeline([("fill", SimpleImputer(strategy=config.CATEGORY_FILL_STRATEGY)),
                           ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preparation = ColumnTransformer([("numbers", numbers, numeric_columns),
                                     ("categories", categories, categorical_columns)])
    return Pipeline([(config.PREPROCESSING_STEP_NAME, preparation), (config.MODEL_STEP_NAME, estimator)])
