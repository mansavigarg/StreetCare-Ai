from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import config
from modeling.model_evaluation import evaluate_text_classifier


def train_sentence_model(train_text, train_target, held_out_text, held_out_target):
    model = Pipeline([(config.PREPROCESSING_STEP_NAME, TfidfVectorizer()),
                      (config.MODEL_STEP_NAME, LogisticRegression(max_iter=config.LOGISTIC_REGRESSION_MAX_ITERATIONS,
                                                                  random_state=config.RANDOM_SEED))])
    model.fit(train_text, train_target)
    return model, evaluate_text_classifier(model, train_text, train_target, held_out_text, held_out_target)
