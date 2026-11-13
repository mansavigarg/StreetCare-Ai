import sys

import sklearn
from sklearn.model_selection import train_test_split

from config import config
from data_pipeline.dataset_loader import load_processed_dataset
from data_pipeline.feature_builder import add_coordination_trigger_flags, add_requirement_counts, build_feature_frame
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.standard_reading import read_defect_standard
from modeling.crew_hours_model import train_crew_hours_model
from modeling.model_registry import save_feature_schema, save_metrics, save_model
from modeling.repeat_model import train_repeat_model
from modeling.sentence_model import train_sentence_model


def _repeat_inputs(file_path, standards):
    history = add_coordination_trigger_flags(add_requirement_counts(load_processed_dataset(file_path), standards))
    target = history[config.HISTORY_TARGET_COLUMN]
    unknown = sorted(set(target) - set(config.REPEAT_TARGET_ENCODING))
    if unknown:
        raise ValueError(f"{config.HISTORY_TARGET_COLUMN}: cannot encode {unknown}")
    features = build_feature_frame(history, config.REPEAT_NUMERIC_FEATURES, config.REPEAT_CATEGORICAL_FEATURES)
    return features, target.map(config.REPEAT_TARGET_ENCODING).astype(int)


def _crew_hours_inputs(file_path):
    logs = add_coordination_trigger_flags(load_processed_dataset(file_path))
    features = build_feature_frame(logs, config.CREW_HOURS_NUMERIC_FEATURES, config.CREW_HOURS_CATEGORICAL_FEATURES)
    return features, logs[config.LOG_TARGET_COLUMN].astype(float)


def _record(algorithm, train_rows, held_out_rows, scores, minimum):
    return {"algorithm": algorithm, "training_rows": train_rows, "held_out_rows": held_out_rows, "scores": scores,
            "minimum_required": minimum, "scikit_learn_version": sklearn.__version__}


def main():
    print("training three models")
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    standards = {defect_type: read_defect_standard(defect_type, documents)
                 for defect_type in config.DEFECT_STANDARD_DOCUMENTS}
    repeat_train, repeat_train_target = _repeat_inputs(config.CLEAN_REPORT_HISTORY_TRAIN_FILE, standards)
    repeat_held, repeat_held_target = _repeat_inputs(config.CLEAN_REPORT_HISTORY_EVAL_FILE, standards)
    repeat, repeat_scores = train_repeat_model(repeat_train, repeat_train_target, repeat_held, repeat_held_target)
    hours_train, hours_train_target = _crew_hours_inputs(config.CLEAN_REPAIR_LOGS_TRAIN_FILE)
    hours_held, hours_held_target = _crew_hours_inputs(config.CLEAN_REPAIR_LOGS_EVAL_FILE)
    crew_hours, hours_scores = train_crew_hours_model(hours_train, hours_train_target, hours_held, hours_held_target)
    sentences = load_processed_dataset(config.CLEAN_STATEMENT_SENTENCES_FILE)
    text_train, text_held, type_train, type_held = train_test_split(
        sentences[config.SENTENCE_TEXT_COLUMN], sentences[config.SENTENCE_TARGET_COLUMN],
        test_size=config.SENTENCE_HELD_OUT_SHARE, stratify=sentences[config.SENTENCE_TARGET_COLUMN],
        random_state=config.RANDOM_SEED)
    sentence, sentence_scores = train_sentence_model(text_train, type_train, text_held, type_held)
    shortfalls = [f"{name} {score} is below {minimum}" for name, score, minimum in [
        ("repeat held-out ROC AUC", repeat_scores["held_out_roc_auc"], config.MINIMUM_REPEAT_ROC_AUC),
        ("crew hours held-out R2", hours_scores["held_out_r2"], config.MINIMUM_CREW_HOURS_R2),
        ("sentence held-out macro F1", sentence_scores["held_out_macro_f1"], config.MINIMUM_SENTENCE_MACRO_F1)]
        if score < minimum]
    if shortfalls:
        raise SystemExit("No model was saved: " + "; ".join(shortfalls))
    for model, file_path in [(repeat, config.REPEAT_MODEL_FILE), (crew_hours, config.CREW_HOURS_MODEL_FILE),
                             (sentence, config.SENTENCE_MODEL_FILE)]:
        save_model(model, file_path)
    save_feature_schema(list(repeat_train.columns), config.REPEAT_SCHEMA_FILE)
    save_feature_schema(list(hours_train.columns), config.CREW_HOURS_SCHEMA_FILE)
    save_metrics({
        "repeat": _record(type(repeat[config.MODEL_STEP_NAME]).__name__, len(repeat_train), len(repeat_held),
                          repeat_scores, config.MINIMUM_REPEAT_ROC_AUC),
        "crew_hours": _record(type(crew_hours[config.MODEL_STEP_NAME]).__name__, len(hours_train),
                              len(hours_held), hours_scores, config.MINIMUM_CREW_HOURS_R2),
        "sentence": _record(" + ".join(type(step).__name__ for _, step in sentence.steps), len(text_train),
                            len(text_held), sentence_scores, config.MINIMUM_SENTENCE_MACRO_F1)})
    print(f"  repeat          held-out ROC AUC {repeat_scores['held_out_roc_auc']:.4f}  "
          f"(train {repeat_scores['train_roc_auc']:.4f}, gap {repeat_scores['generalisation_gap']:+.4f})")
    print(f"  crew hours      held-out R2 {hours_scores['held_out_r2']:.4f}  MAE {hours_scores['mean_absolute_error']:.4f} "
          f"hours  (train {hours_scores['train_r2']:.4f}, gap {hours_scores['generalisation_gap']:+.4f})")
    print(f"  sentence        held-out macro F1 {sentence_scores['held_out_macro_f1']:.4f}  "
          f"(train {sentence_scores['train_macro_f1']:.4f}, gap {sentence_scores['generalisation_gap']:+.4f})")
    print(f"metrics written to {config.MODEL_METRICS_FILE}")
    print("every model reaches the minimum the configuration sets for it")


if __name__ == "__main__":
    sys.exit(main())
