"""SHAP explanations for the repeat-likelihood and crew-hours models.

Read-only: nothing here changes a prediction, a screening or a decision, and no language model is called.
The repeat model (logistic regression) is explained with shap.LinearExplainer against a sample of the
training history; the crew-hours model (random forest) with shap.TreeExplainer. Both models one-hot encode
their categorical answers, so the SHAP values of a category's encoded columns are added back together and
every contribution is reported against the form question it came from.
"""
import math
import threading

import numpy as np

from config import config
from data_pipeline.dataset_loader import load_processed_dataset
from data_pipeline.feature_builder import add_coordination_trigger_flags, add_requirement_counts, build_feature_frame
from tools.prediction_tools import _inputs

_lock = threading.Lock()
_explainers = {}


def _shap():
    import shap  # imported here so the rest of the service runs without the package
    return shap


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _column_groups(model):
    """The original column behind each encoded column, in the order the preprocessing writes them."""
    preparation = model.named_steps[config.PREPROCESSING_STEP_NAME]
    groups = []
    for name, transformer, columns in preparation.transformers_:
        if name == "numbers":
            groups += list(columns)
        elif name == "categories":
            encoder = transformer.named_steps["encode"]
            for column, categories in zip(columns, encoder.categories_):
                groups += [column] * len(categories)
    return np.array(groups)


def _group(values, groups, columns):
    """Adds the SHAP values of each column's encoded parts together: rows x encoded -> rows x columns."""
    values = np.atleast_2d(values)
    return np.column_stack([values[:, groups == column].sum(axis=1) for column in columns])


TRIGGER_FLAG_COLUMNS = set(config.COORDINATION_TRIGGER_FLAG_COLUMNS.values())


def _label(column):
    label = config.FORM_ANSWER_LABELS.get(column, column.replace("_", " "))
    label = label.split(", 1 when it applies")[0]
    return label[:1].upper() + label[1:]


def _shown(value, column=None):
    if column in TRIGGER_FLAG_COLUMNS and value in (0, 1, 0.0, 1.0):
        return config.BOOLEAN_YES if value else config.BOOLEAN_NO
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return config.NOT_GIVEN_TEXT
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def _history_frames(standards):
    """A fixed random sample of the cleaned training extracts, built into features as training built them."""
    history = load_processed_dataset(config.CLEAN_REPORT_HISTORY_TRAIN_FILE)
    history = history.sample(n=min(config.EXPLANATION_HISTORY_SAMPLE_ROWS, len(history)),
                             random_state=config.RANDOM_SEED)
    history = add_coordination_trigger_flags(add_requirement_counts(history, standards))
    repeat = build_feature_frame(history, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES)
    logs = load_processed_dataset(config.CLEAN_REPAIR_LOGS_TRAIN_FILE)
    logs = add_coordination_trigger_flags(logs.sample(n=min(config.EXPLANATION_HISTORY_SAMPLE_ROWS, len(logs)),
                                                      random_state=config.RANDOM_SEED))
    crew = build_feature_frame(logs, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES)
    return repeat, crew


def build_explainers(repeat_model, crew_hours_model, standards):
    """Builds both explainers once, on first use; later calls return the same ones."""
    with _lock:
        if _explainers:
            return _explainers
        shap = _shap()
        repeat_history, crew_history = _history_frames(standards)
        for key, model, history in (("repeat", repeat_model, repeat_history),
                                    ("crew_hours", crew_hours_model, crew_history)):
            preparation = model.named_steps[config.PREPROCESSING_STEP_NAME]
            estimator = model.named_steps[config.MODEL_STEP_NAME]
            encoded_history = _dense(preparation.transform(history))
            if key == "repeat":
                explainer = shap.LinearExplainer(
                    estimator, shap.maskers.Independent(encoded_history, max_samples=len(encoded_history)))
            else:
                explainer = shap.TreeExplainer(estimator)
            groups = _column_groups(model)
            columns = list(history.columns)
            history_values = _group(np.asarray(explainer.shap_values(encoded_history)), groups, columns)
            _explainers[key] = {"model": model, "explainer": explainer, "groups": groups, "columns": columns,
                                "base": float(np.ravel(explainer.expected_value)[0]),
                                "history_values": history_values}
        return _explainers


def _explain_frame(key, frame):
    built = _explainers[key]
    preparation = built["model"].named_steps[config.PREPROCESSING_STEP_NAME]
    encoded = _dense(preparation.transform(frame))
    return _group(np.asarray(built["explainer"].shap_values(encoded)), built["groups"], built["columns"])


def _contributions(key, frame, values):
    """One report's contributions, largest first. For the repeat model the log-odds SHAP values are also
    shared out over the change in likelihood, so each one reads in likelihood points."""
    built = _explainers[key]
    raw = values[0]
    if key == "repeat":
        base = _sigmoid(built["base"])
        prediction = float(built["model"].predict_proba(frame)[0, 1])
        total = raw.sum()
        effects = raw * ((prediction - base) / total) if abs(total) > 1e-12 else raw * 0.0
    else:
        base = built["base"]
        prediction = float(built["model"].predict(frame)[0])
        effects = raw
    row = frame.iloc[0]
    items = [{"feature": column, "label": _label(column), "value": _shown(row[column], column),
              "shap_value": round(float(raw[index]), config.EXPLANATION_DECIMALS),
              "effect": round(float(effects[index]), config.EXPLANATION_DECIMALS)}
             for index, column in enumerate(built["columns"])]
    items.sort(key=lambda item: abs(item["effect"]), reverse=True)
    return {"unit": config.EXPLANATION_REPEAT_UNIT if key == "repeat" else config.EXPLANATION_CREW_HOURS_UNIT,
            "base_value": round(base, config.EXPLANATION_DECIMALS),
            "prediction": round(prediction, config.EXPLANATION_DECIMALS),
            "contributions": items}


def _frames(report, screening):
    counts = {column: screening[column] for column in config.REQUIREMENT_COUNT_RESULTS}
    repeat = _inputs(report, screening, counts, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES,
                     _explainers["repeat"]["columns"])
    crew = _inputs(report, screening, {}, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES,
                   _explainers["crew_hours"]["columns"])
    return repeat, crew


def explain_report(report, screening):
    """Why the two models predicted what they did for one open report."""
    repeat, crew = _frames(report, screening)
    return {"repeat": _contributions("repeat", repeat, _explain_frame("repeat", repeat)),
            "crew_hours": _contributions("crew_hours", crew, _explain_frame("crew_hours", crew))}


def _drivers(key, values, top):
    """Mean absolute impact of each column over many reports, with how often it pushed the figure up or down."""
    columns = _explainers[key]["columns"]
    impact = np.abs(values).mean(axis=0)
    share = impact / impact.sum() if impact.sum() else impact
    order = np.argsort(-impact)[:top]
    return [{"feature": columns[index], "label": _label(columns[index]),
             "mean_impact": round(float(impact[index]), config.EXPLANATION_DECIMALS),
             "share": round(float(share[index]), config.EXPLANATION_DECIMALS),
             "pushes_up": int((values[:, index] > config.EXPLANATION_MINIMUM_EFFECT).sum()),
             "pulls_down": int((values[:, index] < -config.EXPLANATION_MINIMUM_EFFECT).sum())} for index in order]


def summarise_queue(reports):
    """reports: (reference, report profile, screening, queue row) for every open report."""
    rows, repeat_values, crew_values = [], [], []
    for reference, report, screening, queue_row in reports:
        repeat, crew = _frames(report, screening)
        repeat_raw, crew_raw = _explain_frame("repeat", repeat), _explain_frame("crew_hours", crew)
        repeat_values.append(repeat_raw[0])
        crew_values.append(crew_raw[0])
        explained = {"repeat": _contributions("repeat", repeat, repeat_raw),
                     "crew_hours": _contributions("crew_hours", crew, crew_raw)}
        reasons = {part: [item["label"] for item in explained[part]["contributions"]
                          if item["effect"] > config.EXPLANATION_MINIMUM_EFFECT][:config.EXPLANATION_REASON_COUNT]
                   for part in explained}
        rows.append({"report_reference": reference, "street_name": queue_row[config.STREET_NAME_COLUMN],
                     "district": queue_row["district"], "defect_type": queue_row[config.DEFECT_TYPE_COLUMN],
                     "repeat_likelihood": queue_row["repeat_likelihood"],
                     "likelihood_band": queue_row["likelihood_band"],
                     "expected_crew_hours": queue_row["expected_crew_hours"],
                     "repeat_reasons": reasons["repeat"], "crew_hours_reasons": reasons["crew_hours"]})
    top = config.EXPLANATION_TOP_FEATURES
    return {"repeat": {"base_value": round(_sigmoid(_explainers["repeat"]["base"]), config.EXPLANATION_DECIMALS),
                       "queue_drivers": _drivers("repeat", np.array(repeat_values), top),
                       "history_drivers": _drivers("repeat", _explainers["repeat"]["history_values"], top)},
            "crew_hours": {"base_value": round(_explainers["crew_hours"]["base"], config.EXPLANATION_DECIMALS),
                           "queue_drivers": _drivers("crew_hours", np.array(crew_values), top),
                           "history_drivers": _drivers("crew_hours", _explainers["crew_hours"]["history_values"],
                                                       top)},
            "history_rows": len(_explainers["repeat"]["history_values"]), "reports": rows}
