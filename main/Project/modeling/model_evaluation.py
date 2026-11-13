from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, mean_absolute_error, r2_score, roc_auc_score

from config import config


def _rounded(scores):
    return {name: round(float(value), config.METRIC_DECIMALS) for name, value in scores.items()}


def evaluate_classifier(model, train_features, train_target, held_out_features, held_out_target):
    train_scores = model.predict_proba(train_features)[:, 1]
    held_out_scores = model.predict_proba(held_out_features)[:, 1]
    train_auc, held_out_auc = roc_auc_score(train_target, train_scores), roc_auc_score(held_out_target, held_out_scores)
    return _rounded({"train_roc_auc": train_auc, "held_out_roc_auc": held_out_auc,
                     "generalisation_gap": train_auc - held_out_auc,
                     "brier_score": brier_score_loss(held_out_target, held_out_scores),
                     "mean_predicted_probability": held_out_scores.mean(),
                     "positive_share": held_out_target.mean()})


def evaluate_regressor(model, train_features, train_target, held_out_features, held_out_target):
    held_out_forecast = model.predict(held_out_features)
    train_r2 = r2_score(train_target, model.predict(train_features))
    held_out_r2 = r2_score(held_out_target, held_out_forecast)
    average_guess = [float(train_target.mean())] * len(held_out_target)
    return _rounded({"train_r2": train_r2, "held_out_r2": held_out_r2, "generalisation_gap": train_r2 - held_out_r2,
                     "mean_absolute_error": mean_absolute_error(held_out_target, held_out_forecast),
                     "average_estimate_mean_absolute_error": mean_absolute_error(held_out_target, average_guess),
                     "lowest_prediction": held_out_forecast.min(), "highest_prediction": held_out_forecast.max()})


def evaluate_text_classifier(model, train_text, train_target, held_out_text, held_out_target):
    held_out_predicted = model.predict(held_out_text)
    train_f1 = f1_score(train_target, model.predict(train_text), average="macro")
    held_out_f1 = f1_score(held_out_target, held_out_predicted, average="macro")
    per_class = f1_score(held_out_target, held_out_predicted, labels=model.classes_, average=None)
    scores = _rounded({"train_macro_f1": train_f1, "held_out_macro_f1": held_out_f1,
                       "generalisation_gap": train_f1 - held_out_f1,
                       "accuracy": accuracy_score(held_out_target, held_out_predicted),
                       "weighted_f1": f1_score(held_out_target, held_out_predicted, average="weighted")})
    scores["per_class_f1"] = _rounded(dict(zip(map(str, model.classes_), per_class)))
    return scores
