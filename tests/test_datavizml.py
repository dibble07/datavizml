from datavizml.singledistribution import SingleDistribution
from datavizml.exploratorydataanalysis import ExploratoryDataAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest

np.random.seed(42)


def test_single_improper_inputs():
    # initialise inputs
    _, ax = plt.subplots()
    list_ = [0, 1]
    array = np.array(list)
    array_long = np.arange(10_000)
    array_wide = np.array([[0, 1], [0, 1], [0, 1]])

    # check large values use axis formatter
    sd = SingleDistribution(feature=array_long, ax=ax, target=array_long)
    sd()

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.feature = array
    with pytest.raises(AttributeError):
        sd.target = array

    # check inability to initiate with lists
    with pytest.raises(TypeError):
        SingleDistribution(feature=list_, ax=ax, target=array)
    with pytest.raises(TypeError):
        SingleDistribution(feature=array, ax=ax, target=list_)

    # check inability to initiate with 2d arrays
    with pytest.raises(ValueError):
        SingleDistribution(feature=array, ax=ax, target=array_wide)
    with pytest.raises(ValueError):
        SingleDistribution(feature=array_wide, ax=ax, target=array)

    # check inability to initiate with unevenly sized inputs
    with pytest.raises(ValueError):
        SingleDistribution(feature=array, ax=ax, target=array_long)
    with pytest.raises(ValueError):
        SingleDistribution(feature=array_long, ax=ax, target=array)


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
    "dtype_target",
    ["Int64", "Float64", "string", "category", "boolean", "no target provided"],
)
@pytest.mark.parametrize("type_target", ["array", "series"])
@pytest.mark.parametrize(
    "dtype_feature", ["Int64", "Float64", "string", "category", "boolean"]
)
@pytest.mark.parametrize("type_feature", ["array", "series"])
def test_single_with_target(type_feature, dtype_feature, type_target, dtype_target):
    # check config is testable - category is only a dtype for numpy arrays
    config_testable = not (
        type_feature == "array" and dtype_feature == "category"
    ) and not (type_target == "array" and dtype_target == "category")

    if config_testable:
        # initialise raw values - include a missing value and a modal value
        raw = ([0, 1, 2, 3, 4, 4, 4, 4, np.nan]) * 100

        # process raw x values based on type
        if dtype_feature == "Float64":
            # add small value to avoid being downcast to integer
            x = [i + 0.01 if not np.isnan(i) else i for i in raw]
        elif dtype_feature in ["string", "category"]:
            # category is a dtype only for pandas - start with string
            x = [str(i) if not np.isnan(i) else i for i in raw]
        elif dtype_feature == "boolean":
            # convert to boolean - convert missing values for arrays as boolean arrays aren't nullable
            if type_feature == "array":
                x = [i < 2.5 for i in raw]
            else:
                x = [i < 2.5 if not np.isnan(i) else i for i in raw]
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
            # convert to boolean - convert missing values for arrays as boolean arrays aren't nullable
            if type_target == "array":
                y = [i < 2.5 for i in raw]
            else:
                y = [i < 2.5 if not np.isnan(i) else i for i in raw]
        elif dtype_target == "Int64":
            y = raw
        elif dtype_target == "no target provided":
            y = None

        # convert x values to inputs
        x_name = f"x_{dtype_feature}".capitalize()
        if type_feature == "array":
            x_final = np.array(x)
        elif type_feature == "series":
            x_final = pd.Series(x, name=x_name)

        # convert y values to inputs
        y_name = f"y_{dtype_target}".capitalize()
        if y is None:
            y_final = y
        elif type_target == "array":
            y_final = np.array(y)
        elif type_target == "series":
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

        # set expected score
        if dtype_target == "no target provided":
            if dtype_feature in ["Int64", "Float64"]:
                expected_score = 0.139
            elif dtype_feature in ["string", "category"]:
                if type_feature == "array":
                    expected_score = 0.444
                else:
                    expected_score = 0.5
            elif dtype_feature in ["boolean"]:
                if type_feature == "array":
                    expected_score = 0.667
                else:
                    expected_score = 0.625
        else:
            if dtype_feature == "boolean":
                if dtype_target == "boolean":
                    expected_score = 1
                elif dtype_target in ["string", "category"]:
                    # performance is slightly worse for arrays because the nulls are considered a category
                    if type_feature == "array" and type_target == "array":
                        expected_score = 0.189
                    else:
                        expected_score = 0.260
                elif dtype_target in ["Int64", "Float64"]:
                    expected_score = 0.634
            else:
                expected_score = 1

        # initialise object
        _, ax = plt.subplots()
        sd = SingleDistribution(
            feature=(x_final, x_name),
            ax=ax,
            target=(y_final, y_name)
            if type_target == "array" and dtype_target != "no target provided"
            else y_final,
        )

        # check printing
        captured = sd.__str__()
        target_str = (
            dtype_target
            if dtype_target == "no target provided"
            else f"{y_name} ({dtype_target} - {target_analysis_type})"
        )
        expected = f"feature: {x_name} ({dtype_feature}), target: {target_str}, score: not calculated"
        assert expected == captured

        # check missing proportion value
        if type_feature == "array" and dtype_feature in ["boolean", "string"]:
            expected_missing_proportion = 0
        else:
            expected_missing_proportion = 1 / 9
        assert sd.missing_proportion == expected_missing_proportion

        # call object
        sd()

        # check score
        assert np.round(sd.score, 3) == expected_score

        # check printing
        captured = sd.__str__()
        target_str = (
            dtype_target
            if dtype_target == "no target provided"
            else f"{y_name} ({dtype_target} - {target_analysis_type})"
        )
        expected = f"feature: {x_name} ({dtype_feature}), target: {target_str}, score: {expected_score:0.3f}"
        assert expected == captured

        plt.close()


def test_multi_with_float_series_with_float_target():
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
    captured = eda.__str__()
    expected = "features: feature_test (Float64)\ntarget: target_test (Float64)"
    assert expected == captured


def test_multi_with_float_string_dataframe_with_string_target():
    # initialise inputs
    x = pd.DataFrame(
        {
            "feature_float": [i / 10 for i in range(10)],
            "feature_string": [str(i) for i in range(10)],
        }
    )
    y = (np.array([str(i) for i in range(10)]), "target_test")

    # initialise object
    eda = ExploratoryDataAnalysis(data=x, target=y, ncols=2)

    # run object
    eda()

    # check inability to reset values
    with pytest.raises(AttributeError):
        eda.data = x
    with pytest.raises(AttributeError):
        eda.target = y

    # check printing
    captured = eda.__str__()
    expected = "features: feature_float, feature_string (Float64, string)\ntarget: target_test (string)"
    assert expected == captured


def test_multi_with_int_category_dataframe_without_target():
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

    # check inability to reset values
    with pytest.raises(AttributeError):
        eda.data = x

    # check printing
    captured = eda.__str__()
    expected = "features: feature_int, feature_bool, feature_category (Int64, boolean, category)\ntarget: no target provided"
    assert expected == captured


def test_multi_with_list():
    # initialise inputs
    x_list = [[1, 2], [1, 2]]
    y_list = [0, 1]
    x_array = np.array([[1, 2], [1, 2]])
    y_array = np.array([0, 1])
    x_array_long = np.array([[1, 2], [1, 2], [1, 2]])
    y_array_long = np.array([0, 1, 2])

    # check inability to initiate with lists
    with pytest.raises(TypeError):
        ExploratoryDataAnalysis(data=x_list, ncols=2, target=y_array)
    with pytest.raises(TypeError):
        ExploratoryDataAnalysis(data=x_array, ncols=2, target=y_list)

    # check inability to initiate with unevenly sized inputs
    with pytest.raises(ValueError):
        ExploratoryDataAnalysis(data=x_array, ncols=2, target=y_array_long)
    with pytest.raises(ValueError):
        ExploratoryDataAnalysis(data=x_array_long, ncols=2, target=y_array)
