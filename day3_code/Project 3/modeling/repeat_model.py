from sklearn.linear_model import LogisticRegression

from config import config
from data_pipeline.feature_builder import build_model_pipeline
from modeling.model_evaluation import evaluate_classifier


def train_repeat_model(train_features, train_target, held_out_features, held_out_target):
    estimator = LogisticRegression(max_iter=config.LOGISTIC_REGRESSION_MAX_ITERATIONS, random_state=config.RANDOM_SEED)
    model = build_model_pipeline(estimator, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES)
    model.fit(train_features, train_target)
    return model, evaluate_classifier(model, train_features, train_target, held_out_features, held_out_target)
