from sklearn.ensemble import RandomForestRegressor

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_regressor


def train_crew_hours_model(train_features, train_target, held_out_features, held_out_target):
    estimator = RandomForestRegressor(n_estimators=config.CREW_HOURS_TREE_COUNT,
                                      min_samples_leaf=config.CREW_HOURS_MIN_SAMPLES_LEAF,
                                      max_features=config.CREW_HOURS_MAX_FEATURES, random_state=config.RANDOM_SEED,
                                      n_jobs=-1)
    model = build_model_pipeline(estimator, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES)
    model.fit(train_features, train_target)
    return model, evaluate_regressor(model, train_features, train_target, held_out_features, held_out_target)
