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
def test_single(type_feature, dtype_feature, type_target, dtype_target):
    # check config is testable - category is only a dtype for numpy arrays
    config_testable = not (
        type_feature == "array" and dtype_feature == "category"
    ) and not (type_target == "array" and dtype_target == "category")

    if config_testable:
        # initialise raw values - include a missing value and a modal value
        raw = [0, 1, 2, 3, 4, 4, 4, 4, np.nan] * 100

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


def test_multi_improper_inputs():
    # initialise inputs
    x_list = [[1, 2], [1, 2]]
    y_list = [0, 1]
    x_array = np.array([[1, 2], [1, 2]])
    y_array = np.array([0, 1])
    x_array_long = np.array([[1, 2], [1, 2], [1, 2]])
    y_array_long = np.array([0, 1, 2])

    # check inability to reset values
    eda = ExploratoryDataAnalysis(data=x_array, ncols=2, target=y_array)
    with pytest.raises(AttributeError):
        eda.data = x_array
    with pytest.raises(AttributeError):
        eda.target = y_array

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


@pytest.mark.parametrize(
    "dtype_target",
    ["Int64", "Float64", "string", "category", "boolean", "no target provided"],
)
@pytest.mark.parametrize("type_target", ["array", "series"])
@pytest.mark.parametrize("type_data", ["dataframe", "series"])  # "array",
def test_multi(type_data, type_target, dtype_target):
    # check config is testable - category is only a dtype for numpy arrays
    config_testable = not (type_target == "array" and dtype_target == "category")

    if config_testable:
        # initialise raw values - include a missing value and a modal value
        raw = [0, 1, 2, 3, 4, 4, 4, 4, np.nan] * 100

        # process raw x values based on type
        x = {
            "Int64": raw,
            "Float64": [i + 0.01 if not np.isnan(i) else i for i in raw],
            "string": [str(i) if not np.isnan(i) else i for i in raw],
            "category": [str(i) if not np.isnan(i) else i for i in raw],
            "boolean": [i < 2.5 for i in raw]
            if type_data == "array"
            else [i < 2.5 if not np.isnan(i) else i for i in raw],
        }

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
        x_names_raw = [f"x_{i}".capitalize() for i in x.keys()]
        # if type_data == "array":
        #     x_final = np.array(list(x.values()))
        # el
        if type_data == "dataframe":
            x_final_list = [pd.DataFrame(x)]
            x_final_list[0].columns = x_names_raw
            x_names_list = [x_names_raw]
            x_types_list = [x.keys()]
        elif type_data == "series":
            x_final_list = [
                pd.Series(val, name=name) for val, name in zip(x.values(), x_names_raw)
            ]
            x_names_list = [[i] for i in x_names_raw]
            x_types_list = [[i] for i in x.keys()]

        # convert y values to inputs
        y_name = f"y_{dtype_target}".capitalize()
        if y is None:
            y_final = y
        elif type_target == "array":
            y_final = np.array(y)
        elif type_target == "series":
            y_final = pd.Series(y, name=y_name)

        # pandas specific dtype conversion
        if type_data == "dataframe":
            x_final_list[0] = x_final_list[0].astype(
                {name.capitalize(): type_ for type_, name in zip(x.keys(), x_names_raw)}
            )
        elif type_data == "series":
            x_final_list = [i.astype(type_) for type_, i in zip(x.keys(), x_final_list)]
        if dtype_target == "category":
            y_final = y_final.astype("category")

        # initialise objects
        eda_list = []
        for x_final, x_names in zip(x_final_list, x_names_list):
            eda_list.append(
                ExploratoryDataAnalysis(
                    data=(x_final, x_names) if type_data == "array" else x_final,
                    ncols=2,
                    target=(y_final, y_name)
                    if type_target == "array" and dtype_target != "no target provided"
                    else y_final,
                )
            )

        # loop over all eda objects created
        for eda, x_names, x_types in zip(eda_list, x_names_list, x_types_list):
            # check indexing
            assert isinstance(eda[0], SingleDistribution)

            # check printing
            captured = eda.__str__()
            data_str = (
                ", ".join(x_names),
                ", ".join(sorted(x_types)),
            )
            target_str = (
                dtype_target
                if dtype_target == "no target provided"
                else f"{y_name} ({dtype_target})"
            )
            expected = f"features: {data_str[0]} ({data_str[1]})\ntarget: {target_str}"
            assert expected == captured

            # call object
            eda()

            # close figure to save memory
            plt.close(eda.fig)
