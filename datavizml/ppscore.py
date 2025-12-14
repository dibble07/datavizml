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
    baseline_score = mean_absolute_error(df[y], df["median"])
    ppscore = max(0, 1 - (abs(model_score) / baseline_score))
    return ppscore, baseline_score


def _f1_pps(df, y, model_score):
    "Calculates the baseline score for y using F1 score and derives the PPS"
    df["truth"] = preprocessing.LabelEncoder().fit_transform(df[y])
    df["mode"] = df["truth"].mode().values[0]
    truth_shuffled = df["truth"].sample(frac=1, random_state=random_seed)
    baseline_score = max(
        f1_score(df["truth"], df["mode"], average="weighted"),
        f1_score(df["truth"], truth_shuffled, average="weighted"),
    )
    ppscore = max(0, (model_score - baseline_score) / (1 - baseline_score))
    return ppscore, baseline_score


def _calculate_model_cv_score(df, x, y, case, model, scoring):
    "Calculates the mean cross-validated model score"

    # preprocess target
    y_series = df[y]
    if case == "classification":
        y_series = preprocessing.LabelEncoder().fit_transform(y_series)

    # preprocess feature
    x_array = df[x].values.reshape(-1, 1)
    if _is_categorical(df[x]):
        x_array = preprocessing.OneHotEncoder().fit_transform(x_array)

    # evaluate model
    scores = cross_val_score(
        model,
        x_array,
        y_series,
        cv=min(4, len(df)),
        scoring=scoring,
    )

    return scores.mean()


def _calculate_single(df, x, y):
    "Calculates the ppscore for a single feature target pair"

    # extract feature and target columns and drop null rows
    df = df[[x, y]]
    df = df.dropna()

    # convert datetime targets to compatible dtype
    if is_datetime64_any_dtype(df[y]):
        df[y] = df[y].astype(int) / 1e9

    # identify task type and calculate scores
    if x == y:
        case = "predict_self"
        metric = None
        ppscore, model_score, baseline_score = 1, 1, 0
    elif _is_categorical(df[y]):
        case = "classification"
        metric = "f1_weighted"
        model_score = _calculate_model_cv_score(
            df,
            x=x,
            y=y,
            case=case,
            model=tree.DecisionTreeClassifier(),
            scoring=metric,
        )
        ppscore, baseline_score = _f1_pps(df, y, model_score)
    elif _is_numeric(df[y]):
        case = "regression"
        metric = "neg_mean_absolute_error"
        model_score = _calculate_model_cv_score(
            df,
            x=x,
            y=y,
            case=case,
            model=tree.DecisionTreeRegressor(),
            scoring=metric,
        )
        ppscore, baseline_score = _mae_pps(df, y, model_score)
    else:
        raise TypeError(f"Cannot determine task for columns {x} and {y}")

    return {
        "x": x,
        "y": y,
        "ppscore": ppscore,
        "case": case,
        "metric": metric,
        "baseline_score": baseline_score,
        "model_score": abs(model_score),
    }


def calculate(df, x=None, y=None):
    """Calculates the ppscore for all feature target pairs

    :param df: Raw data
    :type df: pandas.DataFrame
    :param x: column names to consider as features
    :type x: list, Optional
    :param y: column names to consider as targets
    :type y: list, Optional

    :return: The ppscore values and relevant calculation information for each feature-target pair
    :rtype: pandas.DataFrame
    """

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
