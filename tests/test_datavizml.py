from datavizml.datavizml import SingleDistribution, ExploratoryDataAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest


def test_single_with_list():
    # initialise inputs
    _, ax = plt.subplots()
    x_list = [1, 2]
    y_list = [0, 1]
    x_array = np.array([1, 2])
    y_array = np.array([0, 1])
    y_array_long = np.array([0, 1, 2])

    # check inability to initiate with lists
    with pytest.raises(TypeError):
        SingleDistribution(feature=x_list, ax=ax, target=y_array)
    with pytest.raises(TypeError):
        SingleDistribution(feature=x_array, ax=ax, target=y_list)

    # check inability to initiate with unevenly sized inputs
    with pytest.raises(ValueError):
        SingleDistribution(feature=x_array, ax=ax, target=y_array_long)


def test_single_prescribed_score():
    # initialise inputs
    _, ax = plt.subplots()
    x = pd.Series(
        [1, 1, 2, np.nan] * 4,
        name="feature_test",
    )
    y = pd.Series([0, 0, 1, 1] * 4, name="target_test")

    # check inability to initiate with score value as string
    with pytest.raises(TypeError):
        SingleDistribution(feature=x, ax=ax, target=y, score="0.1")

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax, target=y, score=0.1)

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.score = 0.2

    # check score value
    assert sd.score == 0.1


@pytest.mark.parametrize(
    "dtype_feature", ["Int64", "Float64", "string", "category", "boolean"]
)
def test_single_with_series_with_boolean_target(capsys, dtype_feature):
    # initialise raw values
    x_raw = ([i for i in range(5)] + [np.nan]) * 4
    y = [True, True, True, False, False, False] * 4

    # process raw values based on type
    if dtype_feature == "Float64":
        x = [i + 0.01 if not np.isnan(i) else i for i in x_raw]
    elif dtype_feature in ["string", "category"]:
        x = [str(i) if not np.isnan(i) else i for i in x_raw]
    elif dtype_feature == "boolean":
        x = [i > 2.5 if not np.isnan(i) else i for i in x_raw]
    else:
        x = x_raw

    # convert values to inputs
    x_series = pd.Series(x, name="feature_test")
    if dtype_feature == "category":
        x_series = x_series.astype("category")
    y_series = pd.Series(y, name="target_test")

    # initialise object
    _, ax = plt.subplots()
    sd = SingleDistribution(feature=x_series, ax=ax, target=y_series)

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = f"feature: feature_test ({dtype_feature}), target: target_test (boolean - Classification), score: not calculated"
    assert expected == captured.out

    # check missing proportion value
    assert sd.missing_proportion == 1 / 6

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.target = y

    # call object
    sd()

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = f"feature: feature_test ({dtype_feature}), target: target_test (boolean - Classification), score: 1.0"
    assert expected == captured.out

    # check score
    assert sd.score == 1.0


def test_single_with_boolean_pandas_with_string_target(capsys):
    # initialise inputs
    _, ax = plt.subplots()
    x = pd.Series(
        [False, False, True, np.nan] * 4,
        name="feature_test",
    )
    y = pd.Series(["one", "one", "two", "two"] * 4, name="target_test")

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax, target=y)

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: feature_test (boolean), target: target_test (string - Classification), score: not calculated"
    assert expected == captured.out

    # check missing proportion value
    assert sd.missing_proportion == 0.25

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.target = y

    # call object
    sd()

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: feature_test (boolean), target: target_test (string - Classification), score: 1.0"
    assert expected == captured.out

    # check score
    assert sd.score == 1.0


def test_single_with_boolean_pandas_with_categorical_target(capsys):
    # initialise inputs
    _, ax = plt.subplots()
    x = pd.Series(
        [False, False, True, np.nan] * 4,
        name="feature_test",
    )
    y = pd.Series(["one", "one", "two", "two"] * 4, name="target_test").astype(
        "category"
    )

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax, target=y)

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: feature_test (boolean), target: target_test (category - Classification), score: not calculated"
    assert expected == captured.out

    # check missing proportion value
    assert sd.missing_proportion == 0.25

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.target = y

    # call object
    sd()

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: feature_test (boolean), target: target_test (category - Classification), score: 1.0"
    assert expected == captured.out

    # check score
    assert sd.score == 1.0


def test_single_with_interger_array_without_target(capsys):
    # initialise inputs
    _, ax = plt.subplots()
    x = np.array(list(range(16 - 1)) + [np.nan]) * 1000
    x_fail = np.repeat(x.reshape(-1, 1), 3, axis=1)

    # check inability to reset values
    with pytest.raises(ValueError):
        SingleDistribution(feature=x_fail, ax=ax)

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax)

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = (
        "feature: unnamed (Int64), target: no target provided, score: not calculated"
    )
    assert expected == captured.out

    # check missing proportion value
    assert sd.missing_proportion == 1 / 16

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.feature = x

    # call object
    sd()

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: unnamed (Int64), target: no target provided, score: 0.0"
    assert expected == captured.out

    # check score
    assert sd.score == 0.0


def test_single_with_float_pandas_with_float_target(capsys):
    # initialise inputs
    _, ax = plt.subplots()
    x = pd.Series(
        [(x - 10) / 2 for x in range(20)] * 10 + [np.nan],
        name="feature_test",
    )
    y = x * x
    y.name = "target_test"

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax, target=y)

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: feature_test (Float64), target: target_test (Float64 - Regression), score: not calculated"
    assert expected == captured.out

    # call object
    sd()

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: feature_test (Float64), target: target_test (Float64 - Regression), score: 1.0"
    assert expected == captured.out

    # check score
    assert sd.score == 1.0


def test_multi_with_float_series_with_float_target(capsys):
    # initialise inputs
    _, ax = plt.subplots()
    x = pd.Series(
        [(x - 10) / 2 for x in range(20)] * 10 + [np.nan],
        name="feature_test",
    )
    y = x * x
    y.name = "target_test"

    # initialise object
    eda = ExploratoryDataAnalysis(data=x, target=y, ncols=1)

    # run object
    eda()

    # check printing
    print(eda, end="")
    captured = capsys.readouterr()
    expected = "features: feature_test (Float64)\ntarget: target_test (Float64)"
    assert expected == captured.out


def test_multi_with_float_string_dataframe_with_string_target(capsys):
    # initialise inputs
    x = pd.DataFrame(
        {
            "feature_float": [i / 10 for i in range(10)],
            "feature_string": [str(i) for i in range(10)],
        }
    )
    y = pd.Series([str(i) for i in range(10)], name="target_test")

    # initialise object
    eda = ExploratoryDataAnalysis(data=x, target=y, ncols=2)

    # run object
    eda()

    # check printing
    print(eda, end="")
    captured = capsys.readouterr()
    expected = "features: feature_float, feature_string (Float64, string)\ntarget: target_test (string)"
    assert expected == captured.out


def test_multi_with_int_category_dataframe_without_target(capsys):
    # initialise inputs
    x = pd.DataFrame(
        {
            "feature_int": [i for i in range(10)],
            "feature_bool": [i % 2 == 0 for i in range(10)],
            "feature_category": [str(i) for i in range(10)],
        }
    ).astype({"feature_category": "category"})

    # initialise object
    eda = ExploratoryDataAnalysis(data=x, ncols=2)

    # run object
    eda()

    # check printing
    print(eda, end="")
    captured = capsys.readouterr()
    expected = "features: feature_int, feature_bool, feature_category (Int64, boolean, category)\ntarget: no target provided"
    assert expected == captured.out
