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
    df["median"] = df[y].median()
    baseline = mean_absolute_error(df[y], df["median"])
    ppscore = max(0, 1 - (abs(model_score) / baseline))
    return ppscore, baseline


def _f1_pps(df, y, model_score):
    "Calculates the baseline score for y using F1 score and derives the PPS"
    df["truth"] = preprocessing.LabelEncoder().fit_transform(df[y])
    df["mode"] = df["truth"].mode().values[0]
    truth_shuffled = df["truth"].sample(frac=1, random_state=random_seed)
    baseline = max(
        f1_score(df["truth"], df["mode"], average="weighted"),
        f1_score(df["truth"], truth_shuffled, average="weighted"),
    )
    ppscore = max(0, (model_score - baseline) / (1 - baseline))
    return ppscore, baseline


def _calculate_model_cv_score(df, target, feature, case, model, scoring):
    "Calculates the mean cross-validated model score"

    # preprocess target
    if case == "classification":
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
        model,
        feature_input,
        target_series,
        cv=min(4, len(df)),
        scoring=scoring,
    )

    return scores.mean()


def _calculate_single(df, x, y):

    # extract feature and target columns and drop null rows
    df = df[[x, y]]
    df = df.dropna()

    # convert datetime targets to compatible dtype
    if is_datetime64_any_dtype(df[y]):
        df[y] = df[y].astype(int) / 1e9

    # identify task type and calculate scores
    if x == y:
        case = "predict_self"
        metric_key = None
        ppscore, model_score, baseline_score = 1, 1, 0
    elif _is_categorical(df[y]):
        case = "classification"
        metric_key = "f1_weighted"
        model_score = _calculate_model_cv_score(
            df,
            target=y,
            feature=x,
            case=case,
            model=tree.DecisionTreeClassifier(),
            scoring=metric_key,
        )
        ppscore, baseline_score = _f1_pps(df, y, model_score)
    elif _is_numeric(df[y]):
        case = "regression"
        metric_key = "neg_mean_absolute_error"
        model_score = _calculate_model_cv_score(
            df,
            target=y,
            feature=x,
            case=case,
            model=tree.DecisionTreeRegressor(),
            scoring=metric_key,
        )
        ppscore, baseline_score = _mae_pps(df, y, model_score)
    else:
        raise TypeError(f"Cannot determine task for columns {x} and {y}")

    return {
        "x": x,
        "y": y,
        "ppscore": ppscore,
        "case": case,
        "metric": metric_key,
        "baseline_score": baseline_score,
        "model_score": abs(model_score),
    }


def calculate(df, x=None, y=None):

    # ensure feature and target names are lists
    x_all = df.columns.tolist() if x is None else [x]
    y_all = df.columns.tolist() if y is None else [y]

    # shuffle dataset
    df = df.sample(n=len(df), random_state=random_seed, replace=False)

    # calculate pps scores
    scores = pd.DataFrame(
        [_calculate_single(df, x_, y_) for x_ in x_all for y_ in y_all]
    )

    return scores
