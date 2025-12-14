from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from datavizml.singledistribution import SingleDistribution
from tests.utils import expected_prediction_matrix

np.random.seed(42)


def test_improper_inputs():
    # initialise inputs
    _, ax = plt.subplots()
    list_ = [0, 1]
    x_series = pd.Series(list_, name="x")
    y_series = pd.Series(list_, name="y")
    x_series_long = pd.Series([i for i in range(10_000)], name="x")
    y_series_long = pd.Series([i for i in range(10_000)], name="y")
    dataframe = pd.DataFrame([[0, 1], [0, 1], [0, 1]])

    # check target vs target
    sd = SingleDistribution(feature=y_series, ax=ax, target=y_series)
    sd()
    assert np.isnan(sd.to_dict()["target_score"])

    # check large values use axis formatter
    sd = SingleDistribution(feature=x_series_long, ax=ax, target=y_series_long)
    sd()

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.feature = x_series
    with pytest.raises(AttributeError):
        sd.target = y_series

    # check inability to initiate with lists
    with pytest.raises(TypeError):
        SingleDistribution(feature=list_, ax=ax, target=y_series)
    with pytest.raises(TypeError):
        SingleDistribution(feature=x_series, ax=ax, target=list_)

    # check inability to initiate with 2d series
    with pytest.raises(TypeError):
        SingleDistribution(feature=x_series, ax=ax, target=dataframe)
    with pytest.raises(TypeError):
        SingleDistribution(feature=dataframe, ax=ax, target=y_series)

    # check inability to initiate with unevenly sized inputs
    with pytest.raises(ValueError):
        SingleDistribution(feature=x_series, ax=ax, target=y_series_long)
    with pytest.raises(ValueError):
        SingleDistribution(feature=x_series_long, ax=ax, target=y_series)


def test_prescribed_score():
    # initialise inputs
    _, ax = plt.subplots()
    x = pd.Series(
        [1, 1, 2, np.nan] * 4,
        name="feature_test",
    )
    y = pd.Series([0, 0, 1, 1] * 4, name="target_test")

    # check inability to initiate with target score value as string
    with pytest.raises(TypeError):
        SingleDistribution(feature=x, ax=ax, target=y, target_score="0.1")

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax, target=y, target_score=0.1)
    sd()

    # check target score value
    assert sd.to_dict()["target_score"] == 0.1


def test_deskew_symmetrical():
    # initialise inputs
    _, ax = plt.subplots()
    x = pd.Series(
        [1, 1, 1, 1] * 4,
        name="feature_test",
    )

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax, feature_deskew=True)
    sd()

    # check target score value
    assert sd.to_dict()["feature_score"] == 0


@pytest.mark.parametrize(
    "feature_deskew",
    [True, False],
)
@pytest.mark.parametrize(
    "target_rebalance",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype_target",
    ["Int64", "Float64", "string", "category", "boolean", "no target provided"],
)
@pytest.mark.parametrize(
    "dtype_feature",
    ["Int64", "Float64", "string", "category", "boolean", "datetime64[ns]"],
)
def test_combinations(dtype_feature, dtype_target, target_rebalance, feature_deskew):
    # initialise raw values - include a missing value and a modal value
    raw = [0, 1, 2, 3, 4, 4, 4, 4, np.nan] * 100

    # process raw x values based on type
    if dtype_feature == "Float64":
        # add small value to avoid being downcast to integer
        x = [i + 0.001 if not np.isnan(i) else i for i in raw]
    elif dtype_feature in ["string", "category"]:
        # category is a dtype only for pandas - start with string
        x = [str(i) if not np.isnan(i) else i for i in raw]
    elif dtype_feature == "boolean":
        # convert to boolean
        x = [i < 2.5 if not np.isnan(i) else i for i in raw]
    elif dtype_feature == "datetime64[ns]":
        # convert to datetime64
        x = [datetime(2000, 1, i + 1) if not np.isnan(i) else i for i in raw]
    elif dtype_feature == "Int64":
        x = raw

    # process raw y values based on type
    if dtype_target == "Float64":
        # add small value to avoid being downcast to integer
        y = [i + 0.01 if not np.isnan(i) else i for i in raw]
    elif dtype_target in ["string", "category"]:
        # category is a dtype only for pandas - start with string
        y = [str(i) if not np.isnan(i) else i for i in raw]
    elif dtype_target == "boolean":
        # convert to boolean
        y = [i < 2.5 if not np.isnan(i) else i for i in raw]
    elif dtype_target == "Int64":
        y = raw
    elif dtype_target == "no target provided":
        y = None

    # convert x values to inputs
    x_name = f"x_{dtype_feature}".capitalize()
    x_final = pd.Series(x, name=x_name)

    # convert y values to inputs
    y_name = f"y_{dtype_target}".capitalize()
    if y is None:
        y_final = y
    else:
        y_final = pd.Series(y, name=y_name)

    # pandas specific dtype conversion
    if dtype_feature == "category":
        x_final = x_final.astype("category")
    if dtype_target == "category":
        y_final = y_final.astype("category")

    # decide target analysis type
    if dtype_target in ["Int64", "Float64"]:
        target_analysis_type = "regression"
    elif dtype_target in ["string", "category", "boolean"]:
        target_analysis_type = "classification"

    # set expected feature score
    expected_feature_transform = None
    if dtype_feature in ["Int64", "Float64", "datetime64[ns]"]:
        if feature_deskew and dtype_feature != "datetime64[ns]":
            expected_feature_score = 0.467
            expected_feature_transform = "exp-2"
        else:
            expected_feature_score = 0.750
    elif dtype_feature in ["string", "category"]:
        expected_feature_score = 0.5
    elif dtype_feature in ["boolean"]:
        expected_feature_score = 0.625

    # set expected target score
    if dtype_target != "no target provided":
        expected_target_score = expected_prediction_matrix(
            dtype_feature, dtype_target, target_rebalance, dtype_target
        )

    # initialise object
    _, ax = plt.subplots()
    sd = SingleDistribution(
        feature=x_final,
        ax=ax,
        feature_deskew=feature_deskew,
        target=y_final,
        target_rebalance=target_rebalance,
    )

    # check printing
    captured = sd.__str__()
    feature_str = (
        "Float64"
        if feature_deskew and dtype_feature in ["Int64", "Float64"]
        else dtype_feature
    )
    target_str = (
        dtype_target
        if dtype_target == "no target provided"
        else f"{y_name} ({dtype_target} - {target_analysis_type})"
    )
    expected = f"feature: {x_name} ({feature_str}), target: {target_str}"
    assert expected == captured

    # call object
    sd()

    # extract values using summary dictionary
    summary = sd.to_dict()
    # from pprint import pp
    # pp(summary)

    # check feature parameters
    assert summary["feature_name"] == x_name
    assert summary["feature_dtype"] == feature_str
    assert np.round(summary["feature_score"], 2) == np.round(expected_feature_score, 2)
    # assert False
    assert (
        summary["feature_score_type"] == "Inter-decile skew"
        if dtype_feature in ["Int64", "Float64"]
        else "Categorical skew"
    )
    assert summary["feature_transform"] == expected_feature_transform
    assert summary["feature_nunique"] == 6 if dtype_feature != "boolean" else 3
    assert summary["feature_missing_proportion"] == 1 / 9

    # check target parameters
    if dtype_target != "no target provided":
        assert summary["target_name"] == y_name
        assert summary["target_dtype"] == dtype_target
        assert np.round(summary["target_score"], 2) == np.round(
            expected_target_score, 2
        )
        assert summary["target_score_type"] == "PPS"
    else:
        assert summary["target_name"] == None
        assert summary["target_dtype"] == None
        assert np.isnan(summary["target_score"])
        assert summary["target_score_type"] == "N/A"

    plt.close()
