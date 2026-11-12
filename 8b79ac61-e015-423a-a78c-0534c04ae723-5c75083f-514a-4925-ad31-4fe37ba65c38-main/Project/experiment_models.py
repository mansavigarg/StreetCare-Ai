import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from config import config
from data_pipeline.dataset_loader import load_processed_dataset
from data_pipeline.feature_builder import (
    add_requirement_counts,
    add_coordination_trigger_flags,
    build_feature_frame,
    build_model_pipeline,
)
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_defect_standard
from modeling.model_evaluation import (
    evaluate_classifier,
    evaluate_regressor,
    evaluate_text_classifier,
)


def benchmark_repeat_models():
    docs = extract_text_from_directory(config.DOCS_DIRECTORY)
    standards = {
        dt: read_defect_standard(dt, docs)
        for dt in config.DEFECT_STANDARD_DOCUMENTS
    }

    train_df = load_processed_dataset(config.CLEAN_REPORT_HISTORY_TRAIN_FILE)
    eval_df = load_processed_dataset(config.CLEAN_REPORT_HISTORY_EVAL_FILE)

    train_df = add_coordination_trigger_flags(add_requirement_counts(train_df, standards))
    eval_df = add_coordination_trigger_flags(add_requirement_counts(eval_df, standards))

    X_train = build_feature_frame(
        train_df, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES
    )
    y_train = train_df[config.HISTORY_TARGET_COLUMN].map(config.REPEAT_TARGET_ENCODING)

    X_eval = build_feature_frame(
        eval_df, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES
    )
    y_eval = eval_df[config.HISTORY_TARGET_COLUMN].map(config.REPEAT_TARGET_ENCODING)

    candidates = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=config.LOGISTIC_REGRESSION_MAX_ITERATIONS,
                random_state=config.RANDOM_SEED,
            ),
            {
                f"{config.MODEL_STEP_NAME}__C": [0.05, 0.1, 1.0, 5.0, 10.0],
                f"{config.MODEL_STEP_NAME}__class_weight": [None, "balanced"],
            },
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=config.RANDOM_SEED),
            {
                f"{config.MODEL_STEP_NAME}__n_estimators": [200, 400],
                f"{config.MODEL_STEP_NAME}__max_depth": [6, 10, None],
                f"{config.MODEL_STEP_NAME}__min_samples_leaf": [2, 5, 10],
            },
        ),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier(random_state=config.RANDOM_SEED),
            {
                f"{config.MODEL_STEP_NAME}__learning_rate": [0.03, 0.05, 0.1],
                f"{config.MODEL_STEP_NAME}__max_depth": [3, 5, 7],
                f"{config.MODEL_STEP_NAME}__l2_regularization": [0.1, 1.0, 5.0],
            },
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}

    for name, (estimator, param_grid) in candidates.items():
        pipe = build_model_pipeline(
            estimator, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES
        )
        search = GridSearchCV(
            pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics = evaluate_classifier(search.best_estimator_, X_train, y_train, X_eval, y_eval)
        results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": round(float(search.best_score_), config.METRIC_DECIMALS),
            "metrics": metrics,
        }

    return results


def benchmark_crew_hours_models():
    train_df = add_coordination_trigger_flags(
        load_processed_dataset(config.CLEAN_REPAIR_LOGS_TRAIN_FILE)
    )
    eval_df = add_coordination_trigger_flags(
        load_processed_dataset(config.CLEAN_REPAIR_LOGS_EVAL_FILE)
    )

    X_train = build_feature_frame(
        train_df, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES
    )
    y_train = train_df[config.LOG_TARGET_COLUMN].astype(float)

    X_eval = build_feature_frame(
        eval_df, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES
    )
    y_eval = eval_df[config.LOG_TARGET_COLUMN].astype(float)

    candidates = {
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=config.RANDOM_SEED),
            {
                f"{config.MODEL_STEP_NAME}__n_estimators": [200, 400],
                f"{config.MODEL_STEP_NAME}__max_depth": [8, 12, None],
                f"{config.MODEL_STEP_NAME}__min_samples_leaf": [1, 3, 5],
                f"{config.MODEL_STEP_NAME}__max_features": ["sqrt", 0.5, 0.8],
            },
        ),
        "ExtraTreesRegressor": (
            ExtraTreesRegressor(random_state=config.RANDOM_SEED),
            {
                f"{config.MODEL_STEP_NAME}__n_estimators": [200, 400],
                f"{config.MODEL_STEP_NAME}__max_depth": [10, 15, None],
                f"{config.MODEL_STEP_NAME}__min_samples_leaf": [1, 3, 5],
            },
        ),
        "HistGradientBoostingRegressor": (
            HistGradientBoostingRegressor(random_state=config.RANDOM_SEED),
            {
                f"{config.MODEL_STEP_NAME}__learning_rate": [0.03, 0.05, 0.1],
                f"{config.MODEL_STEP_NAME}__max_iter": [200, 400],
                f"{config.MODEL_STEP_NAME}__max_depth": [4, 6, 8],
                f"{config.MODEL_STEP_NAME}__min_samples_leaf": [5, 10, 20],
            },
        ),
        "Ridge": (
            Ridge(random_state=config.RANDOM_SEED),
            {
                f"{config.MODEL_STEP_NAME}__alpha": [0.1, 1.0, 10.0, 50.0],
            },
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}

    for name, (estimator, param_grid) in candidates.items():
        pipe = build_model_pipeline(
            estimator, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES
        )
        search = GridSearchCV(
            pipe, param_grid=param_grid, scoring="r2", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics = evaluate_regressor(search.best_estimator_, X_train, y_train, X_eval, y_eval)
        results[name] = {
            "best_params": search.best_params_,
            "cv_r2": round(float(search.best_score_), config.METRIC_DECIMALS),
            "metrics": metrics,
        }

    return results


def benchmark_sentence_models():
    from sklearn.model_selection import train_test_split

    df = load_processed_dataset(config.CLEAN_STATEMENT_SENTENCES_FILE)
    X_train, X_eval, y_train, y_eval = train_test_split(
        df[config.SENTENCE_TEXT_COLUMN],
        df[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE,
        stratify=df[config.SENTENCE_TARGET_COLUMN],
        random_state=config.RANDOM_SEED,
    )

    candidates = {
        "Tfidf_LogisticRegression": (
            LogisticRegression(
                max_iter=config.LOGISTIC_REGRESSION_MAX_ITERATIONS,
                random_state=config.RANDOM_SEED,
            ),
            {
                f"{config.PREPROCESSING_STEP_NAME}__ngram_range": [(1, 1), (1, 2)],
                f"{config.PREPROCESSING_STEP_NAME}__sublinear_tf": [True, False],
                f"{config.MODEL_STEP_NAME}__C": [0.5, 1.0, 3.0, 5.0],
                f"{config.MODEL_STEP_NAME}__class_weight": [None, "balanced"],
            },
        ),
        "Tfidf_CalibratedLinearSVC": (
            CalibratedClassifierCV(LinearSVC(random_state=config.RANDOM_SEED)),
            {
                f"{config.PREPROCESSING_STEP_NAME}__ngram_range": [(1, 1), (1, 2)],
                f"{config.PREPROCESSING_STEP_NAME}__sublinear_tf": [True, False],
                f"{config.MODEL_STEP_NAME}__estimator__C": [0.25, 0.5, 1.0, 2.0],
            },
        ),
        "Tfidf_ComplementNB": (
            ComplementNB(),
            {
                f"{config.PREPROCESSING_STEP_NAME}__ngram_range": [(1, 1), (1, 2)],
                f"{config.MODEL_STEP_NAME}__alpha": [0.1, 0.5, 1.0],
            },
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}

    for name, (estimator, param_grid) in candidates.items():
        pipe = Pipeline(
            [
                (config.PREPROCESSING_STEP_NAME, TfidfVectorizer()),
                (config.MODEL_STEP_NAME, estimator),
            ]
        )
        search = GridSearchCV(
            pipe, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        metrics = evaluate_text_classifier(search.best_estimator_, X_train, y_train, X_eval, y_eval)
        results[name] = {
            "best_params": search.best_params_,
            "cv_macro_f1": round(float(search.best_score_), config.METRIC_DECIMALS),
            "metrics": metrics,
        }

    return results


if __name__ == "__main__":
    summary = {
        "repeat": benchmark_repeat_models(),
        "crew_hours": benchmark_crew_hours_models(),
        "sentence": benchmark_sentence_models(),
    }
    print(f"Starting")
    out_path = Path("output/model_benchmark_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=config.JSON_INDENT, sort_keys=True))
    print(f"Benchmark comparison saved to {out_path}")