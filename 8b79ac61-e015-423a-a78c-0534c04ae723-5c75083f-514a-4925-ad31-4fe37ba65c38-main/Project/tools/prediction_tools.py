from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from config import config
from data_pipeline.feature_builder import build_feature_frame


def _same_report(report, screening):
    if screening["report_reference"] != report[config.REPORT_REFERENCE_COLUMN]:
        raise ValueError(config.SCREENING_OF_ANOTHER_REPORT_MESSAGE.format(
            screening_reference=screening["report_reference"], report_reference=report[config.REPORT_REFERENCE_COLUMN]))


def _inputs(report, screening, extra, numeric_columns, categorical_columns, feature_names):
    flags = {column: int(trigger in screening["coordination_triggers"])
             for trigger, column in config.COORDINATION_TRIGGER_FLAG_COLUMNS.items()}
    frame = build_feature_frame(pd.DataFrame([{**report, **extra, **flags}]), numeric_columns, categorical_columns)
    if list(frame.columns) != list(feature_names):
        raise ValueError(f"The inputs {list(frame.columns)} are not the saved schema {list(feature_names)}")
    return frame


def estimate_repeat_likelihood(repeat_model, feature_names, report, screening):
    _same_report(report, screening)
    counts = {column: screening[column] for column in config.REQUIREMENT_COUNT_RESULTS}
    frame = _inputs(report, screening, counts, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES,
                    feature_names)
    figure = Decimal(str(repeat_model.predict_proba(frame)[0, 1])).quantize(config.LIKELIHOOD_STEP, ROUND_HALF_UP)
    if figure >= config.HIGH_LIKELIHOOD_MINIMUM:
        band = config.HIGH_LIKELIHOOD_BAND
    elif figure >= config.MEDIUM_LIKELIHOOD_MINIMUM:
        band = config.MEDIUM_LIKELIHOOD_BAND
    else:
        band = config.LOW_LIKELIHOOD_BAND
    return {"repeat_likelihood": float(figure), "likelihood_band": band}


def estimate_expected_crew_hours(crew_hours_model, feature_names, report, screening):
    _same_report(report, screening)
    frame = _inputs(report, screening, {}, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES,
                    feature_names)
    expected = Decimal(str(crew_hours_model.predict(frame)[0])).quantize(config.TENTH_STEP, ROUND_HALF_UP)
    planned = report.get(config.PLANNED_CREW_HOURS_COLUMN)
    if planned is None or pd.isna(planned):
        return {"expected_crew_hours": float(expected), "planned_crew_hours": None, "overrun_expected": None}
    overrun = expected > Decimal(str(planned)) * config.OVERRUN_RATIO
    return {"expected_crew_hours": float(expected), "planned_crew_hours": float(planned), "overrun_expected": overrun}


def describe_expected_crew_hours(crew_hours):
    expected = f"{crew_hours['expected_crew_hours']:.{config.CREW_HOURS_DECIMALS}f}"
    if crew_hours["planned_crew_hours"] is None:
        return config.EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE.format(expected_crew_hours=expected)
    planned = f"{crew_hours['planned_crew_hours']:.{config.CREW_HOURS_DECIMALS}f}"
    template = config.OVERRUN_TEMPLATE if crew_hours["overrun_expected"] else config.NO_OVERRUN_TEMPLATE
    return template.format(expected_crew_hours=expected, planned_crew_hours=planned)
