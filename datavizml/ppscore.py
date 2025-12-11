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


def _determine_case_and_prepare_df(df, x, y, sample=5_000):
    "Returns str with the name of the determined case based on the columns x and y"
    if x == y:
        return df, "predict_itself"

    df = df[[x, y]]
    df = df.dropna()

    if is_datetime64_any_dtype(df[y]):
        df[y] = df[y].astype(int) / 1e9

    n = min(sample, len(df)) if sample else len(df)
    df = df.sample(n=n, random_state=random_seed, replace=False)

    if is_categorical_dtype(df[y]):
        return df, "classification"
    if is_numeric_dtype(df[y]) and not is_bool_dtype(df[y]):
        return df, "regression"
    else:
        raise TypeError(
            f"Cannot determine whether {df.dtypes} should be regression or classification"
        )


def _calculate_model_cv_score(df, target, feature, task, cross_validation):
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
        cv=cross_validation,
        scoring=task["metric_key"],
    )

    return scores.mean()


def score(
    df,
    x,
    y,
    task=None,
    sample=5_000,
    cross_validation=4,
):
    if cross_validation > len(df):
        logger.warning(
            f"cross_validation value ({cross_validation}) has been reduced to number of samples present ({len(df)})"
        )
        cross_validation = len(df)

    df, case_type = _determine_case_and_prepare_df(df, x, y, sample=sample)
    task = VALID_CALCULATIONS[case_type]

    if case_type in ["classification", "regression"]:
        model_score = _calculate_model_cv_score(
            df,
            target=y,
            feature=x,
            task=task,
            cross_validation=cross_validation,
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


def _format_list_of_dicts(scores, output, sorted):
    """
    Format list of score dicts `scores`
    - maybe sort by ppscore
    - maybe return pandas.Dataframe
    - output can be one of ["df", "list"]
    """
    if sorted:
        scores.sort(key=lambda item: item["ppscore"], reverse=True)

    if output == "df":
        df_columns = [
            "x",
            "y",
            "ppscore",
            "case",
            "is_valid_score",
            "metric",
            "baseline_score",
            "model_score",
            "model",
        ]
        data = {column: [score[column] for score in scores] for column in df_columns}
        scores = pd.DataFrame.from_dict(data)

    return scores


def predictors(df, y, output="df", sorted=True, **kwargs):
    """
    Calculate the Predictive Power Score (PPS) of all the features in the dataframe
    against a target column

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    y : str
        Name of the column y which acts as the target
    output: str - potential values: "df", "list"
        Control the type of the output. Either return a pandas.DataFrame (df) or a list with the score dicts
    sorted: bool
        Whether or not to sort the output dataframe/list by the ppscore
    kwargs:
        Other key-word arguments that shall be forwarded to the pps.score method,
        e.g. `sample`, `cross_validation`

    Returns
    -------
    pandas.DataFrame or list of Dict
        Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced
        by the output argument
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    if len(df[[y]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
        )
    if not output in ["df", "list"]:
        raise ValueError(
            f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values"""
        )
    if not sorted in [True, False]:
        raise ValueError(
            f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
        )

    scores = [score(df, column, y, **kwargs) for column in df if column != y]

    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)


def matrix(df, output="df", sorted=False, **kwargs):
    """
    Calculate the Predictive Power Score (PPS) matrix for all columns in the dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe that contains the data
    output: str - potential values: "df", "list"
        Control the type of the output. Either return a pandas.DataFrame (df) or a list with the score dicts
    sorted: bool
        Whether or not to sort the output dataframe/list by the ppscore
    kwargs:
        Other key-word arguments that shall be forwarded to the pps.score method,
        e.g. `sample`, `cross_validation`

    Returns
    -------
    pandas.DataFrame or list of Dict
        Either returns a tidy dataframe or a list of all the PPS dicts. This can be influenced
        by the output argument
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    if not output in ["df", "list"]:
        raise ValueError(
            f"""The 'output' argument should be one of ["df", "list"] but you passed: {output}\nPlease adjust your input to one of the valid values"""
        )
    if not sorted in [True, False]:
        raise ValueError(
            f"""The 'sorted' argument should be one of [True, False] but you passed: {sorted}\nPlease adjust your input to one of the valid values"""
        )

    scores = [score(df, x, y, **kwargs) for x in df for y in df]

    return _format_list_of_dicts(scores=scores, output=output, sorted=sorted)
