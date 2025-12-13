from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, f1_score

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_object_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
)

random_seed = 123


def _is_categorical(series) -> bool:
    "Determines if series contains categorical values"
    return (
        is_bool_dtype(series)
        or is_object_dtype(series)
        or is_string_dtype(series)
        or is_categorical_dtype(series)
    )


def _is_numeric(series) -> bool:
    "Determines if series contains numeric values"
    return is_numeric_dtype(series) and not is_bool_dtype(series)


def _mae_pps(df, y, model_score):
    "Calculates the baseline score for y using MAE and derives the PPS"
    df["naive"] = df[y].median()
    baseline = mean_absolute_error(df[y], df["naive"])
    ppscore = max(0, 1 - (abs(model_score) / baseline))
    return ppscore, baseline


def _f1_pps(df, y, model_score):
    "Calculates the baseline score for y using F1 score and derives the PPS"
    df["truth"] = preprocessing.LabelEncoder().fit_transform(df[y])
    df["most_common_value"] = df["truth"].value_counts().index[0]
    truth_shuffled = df["truth"].sample(frac=1, random_state=random_seed)
    baseline = max(
        f1_score(df["truth"], df["most_common_value"], average="weighted"),
        f1_score(df["truth"], truth_shuffled, average="weighted"),
    )
    ppscore = max(0, (model_score - baseline) / (1 - baseline))
    return ppscore, baseline


VALID_CALCULATIONS = {
    "regression": {
        "type": "regression",
        "model_score": None,
        "baseline_score": None,
        "ppscore": None,
        "metric_name": "mean absolute error",
        "metric_key": "neg_mean_absolute_error",
        "model": tree.DecisionTreeRegressor(),
        "score_pps": _mae_pps,
    },
    "classification": {
        "type": "classification",
        "model_score": None,
        "baseline_score": None,
        "ppscore": None,
        "metric_name": "weighted F1",
        "metric_key": "f1_weighted",
        "model": tree.DecisionTreeClassifier(),
        "score_pps": _f1_pps,
    },
    "predict_self": {
        "type": "predict_self",
        "model_score": 1,
        "baseline_score": 0,
        "ppscore": 1,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_pps": None,
    },
}


def _calculate_model_cv_score(df, target, feature, task):
    "Calculates the mean model score based on cross-validation"

    # preprocess target
    if task["type"] == "classification":
        df[target] = preprocessing.LabelEncoder().fit_transform(df[target])
    target_series = df[target]

    # preprocess feature
    array = df[feature].values.reshape(-1, 1)
    if _is_categorical(df[feature]):
        feature_input = preprocessing.OneHotEncoder().fit_transform(array)
    else:
        feature_input = array

    # evaluate model
    scores = cross_val_score(
        task["model"],
        feature_input,
        target_series,
        cv=min(4, len(df)),
        scoring=task["metric_key"],
    )

    return scores.mean()


def score(df, x, y):

    df = df[[x, y]]
    df = df.dropna()

    if is_datetime64_any_dtype(df[y]):
        df[y] = df[y].astype(int) / 1e9

    df = df.sample(n=min(10_000, len(df)), random_state=random_seed, replace=False)

    if x == y:
        case_type = "predict_self"
    elif _is_categorical(df[y]):
        case_type = "classification"
    elif _is_numeric(df[y]):
        case_type = "regression"
    else:
        raise TypeError(
            f"Cannot determine whether {df.dtypes} should be regression or classification"
        )

    task = VALID_CALCULATIONS[case_type]

    if case_type in ["classification", "regression"]:
        model_score = _calculate_model_cv_score(
            df,
            target=y,
            feature=x,
            task=task,
        )
        ppscore, baseline_score = task["score_pps"](df, y, model_score)
    else:
        model_score = task["model_score"]
        baseline_score = task["baseline_score"]
        ppscore = task["ppscore"]

    return {
        "x": x,
        "y": y,
        "ppscore": ppscore,
        "case": case_type,
        "metric": task["metric_name"],
        "baseline_score": baseline_score,
        "model_score": abs(model_score),
        "model": task["model"],
    }


def predictors(df, y=None):
    if y:
        scores = [score(df, x, y) for x in df]
    else:
        scores = [score(df, x, y) for x in df for y in df]
    return pd.DataFrame(scores)
