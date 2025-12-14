from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from datavizml.exploratorydataanalysis import ExploratoryDataAnalysis
from datavizml.singledistribution import SingleDistribution
from tests.utils import expected_prediction_matrix

np.random.seed(42)


def test_improper_inputs():
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


def test_transforms():
    # initialise inputs
    raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] * 100
    data_transform = pd.DataFrame(
        {
            "raw": raw,
            "square": np.sqrt(raw),
            "square-root": np.square(raw),
            "log-2": np.exp2(raw),
            "exp-2": np.log2(raw),
        }
    )

    # check ability to transform
    ExploratoryDataAnalysis(
        data=data_transform,
        ncols=5,
        data_deskew=True,
        figure_width=18,
        axes_height=3,
    )()

    # check ability to transform
    ExploratoryDataAnalysis(
        data=data_transform,
        ncols=5,
        data_deskew=["square", "square-root"],
        figure_width=18,
        axes_height=3,
    )()


@pytest.mark.parametrize(
    "target_rebalance",
    [True, False],
)
@pytest.mark.parametrize(
    "matrix_full",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype_target",
    ["Int64", "Float64", "string", "category", "boolean", "no target provided"],
)
@pytest.mark.parametrize("type_data", ["dataframe", "series"])
def test_combinations(type_data, dtype_target, matrix_full, target_rebalance):
    # initialise raw values - include a missing value and a modal value
    raw = [0, 1, 2, 3, 4, 4, 4, 4, np.nan] * 100

    # process raw x values based on type
    x = {
        "Int64": raw,
        "Float64": [i + 0.001 if not np.isnan(i) else i for i in raw],
        "string": [str(i) if not np.isnan(i) else i for i in raw],
        "category": [str(i) if not np.isnan(i) else i for i in raw],
        "boolean": [i < 2.5 if not np.isnan(i) else i for i in raw],
        "datetime64[ns]": [
            datetime(2000, 1, i + 1) if not np.isnan(i) else i for i in raw
        ],
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
                target_rebalance=target_rebalance,
                prediction_matrix_full=matrix_full,
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

        # check single distribution pps scores are correct
        for sd in eda.single_distributions:
            if dtype_target != "no target provided":
                assert np.round(sd.to_dict()["target_score"], 2) == np.round(
                    expected_prediction_matrix(
                        sd.feature.name[2:],
                        sd.target.name[2:],
                        target_rebalance,
                        dtype_target,
                    ),
                    2,
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
                "feature_transform",
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
        if dtype_target == "no target provided" and not matrix_full:
            assert eda.prediction_matrix == None
        else:
            captured_prediction_matrix = eda.prediction_matrix.pivot(
                index="x", columns="y", values="ppscore"
            ).round(2)
            for col_name, col in captured_prediction_matrix.items():
                for row_name, captured_val in col.items():
                    expected_val = np.round(
                        expected_prediction_matrix(
                            row_name[2:],
                            col_name[2:],
                            target_rebalance and dtype_target != "no target provided",
                            dtype_target,
                        ),
                        2,
                    )
                    assert (expected_val == captured_val) or (
                        np.isnan(expected_val) and np.isnan(captured_val)
                    )

        # checks prediction heatmap plotting
        fig, ax = plt.subplots()
        if not dtype_target != "no target provided" and not matrix_full:
            with pytest.raises(TypeError):
                eda.prediction_score_plot(ax=ax)
        else:
            eda.prediction_score_plot(ax=ax)

        # close figure to save memory
        plt.close(eda.fig)
        plt.close(fig)
