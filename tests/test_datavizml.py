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
    x_series = pd.Series(list_, name="x")
    y_series = pd.Series(list_, name="y")
    x_series_long = pd.Series([i for i in range(10_000)], name="x")
    y_series_long = pd.Series([i for i in range(10_000)], name="y")
    dataframe = pd.DataFrame([[0, 1], [0, 1], [0, 1]])

    # check target vs target
    sd = SingleDistribution(feature=y_series, ax=ax, target=y_series)
    sd()
    assert np.isnan(sd.target_score)

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


def test_single_prescribed_score():
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

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.target_score = 0.2

    # check target score value
    assert sd.target_score == 0.1


@pytest.mark.parametrize(
    "dtype_target",
    ["Int64", "Float64", "string", "category", "boolean", "no target provided"],
)
@pytest.mark.parametrize(
    "dtype_feature", ["Int64", "Float64", "string", "category", "boolean"]
)
def test_single(dtype_feature, dtype_target):
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
        # convert to boolean
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
    if dtype_feature in ["Int64", "Float64"]:
        expected_feature_score = 0.139
    elif dtype_feature in ["string", "category"]:
        expected_feature_score = 0.5
    elif dtype_feature in ["boolean"]:
        expected_feature_score = 0.625

    # set expected feature score
    if dtype_feature == "boolean":
        if dtype_target == "boolean":
            expected_target_score = 1
        elif dtype_target in ["string", "category"]:
            expected_target_score = 0.260
        elif dtype_target in ["Int64", "Float64"]:
            expected_target_score = 0.634
    else:
        expected_target_score = 1

    # initialise object
    _, ax = plt.subplots()
    sd = SingleDistribution(
        feature=x_final,
        ax=ax,
        target=y_final,
    )

    # check printing
    captured = sd.__str__()
    target_str = (
        dtype_target
        if dtype_target == "no target provided"
        else f"{y_name} ({dtype_target} - {target_analysis_type})"
    )
    expected = f"feature: {x_name} ({dtype_feature}), target: {target_str}"
    assert expected == captured

    # call object
    sd()

    # extract values using summary dictionary
    summary = sd.to_dict()

    # check feature parameters
    assert summary["feature_name"] == x_name
    assert summary["feature_dtype"] == dtype_feature
    assert np.round(summary["feature_score"], 3) == expected_feature_score
    assert (
        summary["feature_score_type"] == "Inter-quartile skew"
        if sd.feature_is_numeric and not sd.feature_is_bool
        else "Categorical skew"
    )
    assert summary["feature_nunique"] == 6 if dtype_feature != "boolean" else 3
    assert summary["feature_missing_proportion"] == 1 / 9

    # check target parameters
    if sd.has_target:
        assert summary["target_name"] == y_name
        assert summary["target_dtype"] == dtype_target
        assert np.round(summary["target_score"], 3) == expected_target_score
        assert summary["target_score_type"] == "PPS"
    else:
        assert summary["target_name"] == None
        assert summary["target_dtype"] == None
        assert np.isnan(summary["target_score"])
        assert summary["target_score_type"] == "N/A"

    plt.close()


def test_multi_improper_inputs():
    # initialise inputs
    x_list = [[1, 2], [1, 2]]
    y_list = [0, 1]
    x_dataframe = pd.DataFrame([[1, 2], [1, 2]])
    y_series = pd.Series([0, 1], name="y")
    x_dataframe_long = pd.DataFrame([[1, 2], [1, 2], [1, 2]])
    y_series_long = pd.Series([0, 1, 2], name="y")

    # check inability to reset values
    eda = ExploratoryDataAnalysis(data=x_dataframe, ncols=2, target=y_series)
    with pytest.raises(AttributeError):
        eda.data = x_dataframe
    with pytest.raises(AttributeError):
        eda.target = y_series

    # check inability to initiate with lists
    with pytest.raises(TypeError):
        ExploratoryDataAnalysis(data=x_list, ncols=2, target=y_series)
    with pytest.raises(TypeError):
        ExploratoryDataAnalysis(data=x_dataframe, ncols=2, target=y_list)

    # check inability to initiate with unevenly sized inputs
    with pytest.raises(ValueError):
        ExploratoryDataAnalysis(data=x_dataframe, ncols=2, target=y_series_long)
    with pytest.raises(ValueError):
        ExploratoryDataAnalysis(data=x_dataframe_long, ncols=2, target=y_series)


@pytest.mark.parametrize(
    "matrix_full",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype_target",
    ["Int64", "Float64", "string", "category", "boolean", "no target provided"],
)
@pytest.mark.parametrize("type_data", ["dataframe", "series"])
def test_multi(type_data, dtype_target, matrix_full):
    # initialise raw values - include a missing value and a modal value
    raw = [0, 1, 2, 3, 4, 4, 4, 4, np.nan] * 100

    # process raw x values based on type
    x = {
        "Int64": raw,
        "Float64": [i + 0.01 if not np.isnan(i) else i for i in raw],
        "string": [str(i) if not np.isnan(i) else i for i in raw],
        "category": [str(i) if not np.isnan(i) else i for i in raw],
        "boolean": [i < 2.5 if not np.isnan(i) else i for i in raw],
    }

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
    x_names_raw = [f"x_{i}" for i in x.keys()]
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
    y_name = f"y_{dtype_target}"
    if y is None:
        y_final = y
    else:
        y_final = pd.Series(y, name=y_name)

    # pandas specific dtype conversion
    if type_data == "dataframe":
        x_final_list[0] = x_final_list[0].astype(
            {name: type_ for type_, name in zip(x.keys(), x_names_raw)}
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
                data=x_final,
                ncols=2,
                target=y_final,
                prediction_matrix_full=matrix_full,
            )
        )

    # loop over all eda objects created
    for eda, x_names, x_types in zip(eda_list, x_names_list, x_types_list):
        # check indexing
        assert isinstance(eda[0], SingleDistribution)

        # set expected prediction matrix
        expected_prediction_matrix = pd.DataFrame(
            [
                [
                    1.0,
                    0.26,
                    0.634,
                    0.634,
                    0.26,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            ],
            index=["boolean", "category", "Float64", "Int64", "string"],
            columns=["boolean", "category", "Float64", "Int64", "string"],
        )

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

        # check single distribution pps scores are correct
        for sd in eda.single_distributions:
            if sd.has_target:
                assert (
                    np.round(sd.target_score, 3)
                    == expected_prediction_matrix.loc[
                        sd.feature.name[2:], sd.target.name[2:]
                    ]
                )

        # check summary dataframe - structure only as values tested in singledistribution
        summary = eda.summary()
        assert isinstance(summary, pd.DataFrame)
        assert (
            summary.columns
            == [
                "feature_name",
                "feature_dtype",
                "feature_score",
                "feature_score_type",
                "feature_nunique",
                "feature_missing_proportion",
                "target_name",
                "target_dtype",
                "target_score",
                "target_score_type",
            ]
        ).all()
        assert summary.shape[0] == len(x_names)

        # check prediction matrix values
        if not eda.has_target and not eda.prediction_matrix_full:
            assert eda.prediction_matrix == None
        else:
            captured_prediction_matrix = eda.prediction_matrix.pivot(
                index="x", columns="y", values="ppscore"
            ).round(3)
            for col_name, col in captured_prediction_matrix.items():
                for row_name, captured_val in col.items():
                    expected_val = expected_prediction_matrix.loc[
                        row_name[2:], col_name[2:]
                    ]
                    assert expected_val == captured_val

        # checks prediction heatmap plotting
        fig, ax = plt.subplots()
        if not eda.has_target and not eda.prediction_matrix_full:
            with pytest.raises(TypeError):
                eda.prediction_score_plot(ax=ax)
        else:
            eda.prediction_score_plot(ax=ax)

        # close figure to save memory
        plt.close(eda.fig)
        plt.close(fig)
