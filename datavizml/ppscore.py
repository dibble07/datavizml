from sklearn import logger, tree
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, f1_score

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_object_dtype,
    is_categorical_dtype as _is_categorical_dtype,
    is_string_dtype,
    is_datetime64_any_dtype,
)

random_seed = 123


def is_categorical_dtype(series) -> bool:
    "Determines if the dtype of the series represents categorical values"
    return (
        is_bool_dtype(series)
        or is_object_dtype(series)
        or is_string_dtype(series)
        or _is_categorical_dtype(series)
    )


def _normalized_mae_score(model_mae, naive_mae):
    "Normalizes the model MAE score, given the baseline score"
    out = 0 if model_mae > naive_mae else 1 - (model_mae / naive_mae)
    return out


def _mae_normalizer(df, y, model_score):
    "In case of MAE, calculates the baseline score for y and derives the PPS"
    df["naive"] = df[y].median()
    baseline_score = mean_absolute_error(df[y], df["naive"])
    ppscore = _normalized_mae_score(abs(model_score), baseline_score)
    return ppscore, baseline_score


def _normalized_f1_score(model_f1, baseline_f1):
    "Normalizes the model F1 score, given the baseline score"
    out = 0 if model_f1 < baseline_f1 else (model_f1 - baseline_f1) / (1 - baseline_f1)
    return out


def _f1_normalizer(df, y, model_score):
    "In case of F1, calculates the baseline score for y and derives the PPS"
    df["truth"] = preprocessing.LabelEncoder().fit_transform(df[y])
    df["most_common_value"] = df["truth"].value_counts().index[0]
    baseline_score = max(
        f1_score(df["truth"], df["most_common_value"], average="weighted"),
        f1_score(
            df["truth"],
            df["truth"].sample(frac=1, random_state=random_seed),
            average="weighted",
        ),
    )
    ppscore = _normalized_f1_score(model_score, baseline_score)
    return ppscore, baseline_score


VALID_CALCULATIONS = {
    "regression": {
        "type": "regression",
        "is_valid_score": True,
        "model_score": None,
        "baseline_score": None,
        "ppscore": None,
        "metric_name": "mean absolute error",
        "metric_key": "neg_mean_absolute_error",
        "model": tree.DecisionTreeRegressor(),
        "score_normalizer": _mae_normalizer,
    },
    "classification": {
        "type": "classification",
        "is_valid_score": True,
        "model_score": None,
        "baseline_score": None,
        "ppscore": None,
        "metric_name": "weighted F1",
        "metric_key": "f1_weighted",
        "model": tree.DecisionTreeClassifier(),
        "score_normalizer": _f1_normalizer,
    },
    "predict_itself": {  # remove this task entirely if possible
        "type": "predict_itself",
        "is_valid_score": True,
        "model_score": 1,
        "baseline_score": 0,
        "ppscore": 1,
        "metric_name": None,
        "metric_key": None,
        "model": None,
        "score_normalizer": None,
    },
}


def _determine_case_and_prepare_df(df, x, y):
    "Returns str with the name of the determined case based on the columns x and y"
    if x == y:
        return df, "predict_itself"

    df = df[[x, y]]
    df = df.dropna()

    if is_datetime64_any_dtype(df[y]):
        df[y] = df[y].astype(int) / 1e9

    df = df.sample(n=min(10_000, len(df)), random_state=random_seed, replace=False)

    if is_categorical_dtype(df[y]):
        return df, "classification"
    if is_numeric_dtype(df[y]) and not is_bool_dtype(df[y]):
        return df, "regression"
    else:
        raise TypeError(
            f"Cannot determine whether {df.dtypes} should be regression or classification"
        )


def _calculate_model_cv_score(df, target, feature, task):
    "Calculates the mean model score based on cross-validation"

    # preprocess target
    if task["type"] == "classification":
        df[target] = preprocessing.LabelEncoder().fit_transform(df[target])
    target_series = df[target]

    # preprocess feature
    array = df[feature].values.reshape(-1, 1)
    if is_categorical_dtype(df[feature]):
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
    df, case_type = _determine_case_and_prepare_df(df, x, y)
    task = VALID_CALCULATIONS[case_type]

    if case_type in ["classification", "regression"]:
        model_score = _calculate_model_cv_score(
            df,
            target=y,
            feature=x,
            task=task,
        )
        ppscore, baseline_score = task["score_normalizer"](df, y, model_score)
    else:
        model_score = task["model_score"]
        baseline_score = task["baseline_score"]
        ppscore = task["ppscore"]

    return {
        "x": x,
        "y": y,
        "ppscore": ppscore,
        "case": case_type,
        "is_valid_score": task["is_valid_score"],
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
