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
    # IDEA: log.warning when values have been dropped
    df = df.dropna()

    if is_datetime64_any_dtype(df[y]):
        df[y] = df[y].astype(int) / 1e9

    df = _maybe_sample(df, sample)

    if _feature_is_id(df, x):
        return df, "feature_is_id"

    category_count = df[y].value_counts().count()
    if category_count == 1:
        # it is helpful to separate this case in order to save unnecessary calculation time
        return df, "target_is_constant"
    if is_categorical_dtype(df[y]) and (category_count == len(df[y])):
        # it is important to separate this case in order to save unnecessary calculation time
        return df, "target_is_id"

    if is_categorical_dtype(df[y]):
        return df, "classification"
    if is_numeric_dtype(df[y]):
        # this check needs to be after is_bool_dtype (which is part of is_categorical_dtype) because bool is considered numeric by pandas
        return df, "regression"

    # IDEA: show warning
    # raise Exception(
    #     f"Could not infer a valid task based on the target {y}. The dtype {df[y].dtype} is not yet supported"
    # )  # pragma: no cover
    return df, "target_data_type_not_supported"


def _feature_is_id(df, x):
    "Returns Boolean if the feature column x is an ID"
    if not is_categorical_dtype(df[x]):
        return False

    category_count = df[x].value_counts().count()
    return category_count == len(df[x])


def _maybe_sample(df, sample):
    """
    Maybe samples the rows of the given df to have at most `sample` rows
    If sample is `None` or falsy, there will be no sampling.
    If the df has fewer rows than the sample, there will be no sampling.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that might be sampled
    sample : int or `None`
        Number of rows to be sampled

    Returns
    -------
    pandas.DataFrame
        DataFrame after potential sampling
    """
    if sample and len(df) > sample:
        # this is a problem if x or y have more than sample=5000 categories
        # TODO: dont sample when the problem occurs and show warning
        df = df.sample(sample, random_state=random_seed, replace=False)
    return df


def _calculate_model_cv_score_(df, target, feature, task, cross_validation):
    "Calculates the mean model score based on cross-validation"

    # shuffle the rows
    df = df.sample(frac=1, random_state=random_seed, replace=False)

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


def _score(df, x, y, task, sample, cross_validation):
    df, case_type = _determine_case_and_prepare_df(df, x, y, sample=sample)
    task = _get_task(case_type)

    if case_type in ["classification", "regression"]:
        model_score = _calculate_model_cv_score_(
            df,
            target=y,
            feature=x,
            task=task,
            cross_validation=cross_validation,
        )
        # IDEA: the baseline_scores do sometimes change significantly, e.g. for F1 and thus change the PPS
        # we might want to calculate the baseline_score 10 times and use the mean in order to have less variance
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
        "model_score": abs(model_score),  # sklearn returns negative mae
        "model": task["model"],
    }


def score(
    df,
    x,
    y,
    task=None,
    sample=5_000,
    cross_validation=4,
):
    """
    Calculate the Predictive Power Score (PPS) for "x predicts y"
    The score always ranges from 0 to 1 and is data-type agnostic.

    A score of 0 means that the column x cannot predict the column y better than a naive baseline model.
    A score of 1 means that the column x can perfectly predict the column y given the model.
    A score between 0 and 1 states the ratio of how much potential predictive power the model achieved compared to the baseline model.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that contains the columns x and y
    x : str
        Name of the column x which acts as the feature
    y : str
        Name of the column y which acts as the target
    sample : int or `None`
        Number of rows for sampling. The sampling decreases the calculation time of the PPS.
        If `None` there will be no sampling.
    cross_validation : int
        Number of iterations during cross-validation. This has the following implications:
        For example, if the number is 4, then it is possible to detect patterns when there are at least 4 times the same observation. If the limit is increased, the required minimum observations also increase. This is important, because this is the limit when sklearn will throw an error and the PPS cannot be calculated

    Returns
    -------
    Dict
        A dict that contains multiple fields about the resulting PPS.
        The dict enables introspection into the calculations that have been performed under the hood
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"The 'df' argument should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame"
        )
    if len(df[[x]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[x]].columns)} columns with the same column name {x}\nPlease adjust the dataframe and make sure that only 1 column has the name {x}"
        )
    if len(df[[y]].columns) >= 2:
        raise AssertionError(
            f"The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}"
        )
    if task is not None:
        raise AttributeError(
            "The attribute 'task' is no longer supported because it led to confusion and inconsistencies.\nThe task of the model is now determined based on the data types of the columns. If you want to change the task please adjust the data type of the column.\nFor more details, please refer to the README"
        )
    if cross_validation > len(df):
        logger.warning(
            f"cross_validation value ({cross_validation}) has been reduced to number of samples present ({len(df)})"
        )
        cross_validation = len(df)

    return _score(
        df,
        x,
        y,
        task,
        sample,
        cross_validation,
    )


def _get_task(case_type):
    if case_type in VALID_CALCULATIONS.keys():
        return VALID_CALCULATIONS[case_type]
    else:
        raise Exception(f"case_type {case_type} is not supported")


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
