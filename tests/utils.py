# set expected prediction matrix
from typing import List, Union

import pandas as pd

expected_prediction_matrix_raw = pd.DataFrame(
    data=[
        [1.000, 0.260, 0.634, 0.634, 0.634, 0.260],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    ],
    index=["boolean", "category", "Float64", "Int64", "datetime64[ns]", "string"],
    columns=["boolean", "category", "Float64", "Int64", "datetime64[ns]", "string"],
)

expected_prediction_matrix_balanced_multi = pd.DataFrame(
    data=[
        [1.000, 0.043, 0.497, 0.497, 0.497, 0.043],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    ],
    index=["boolean", "category", "Float64", "Int64", "datetime64[ns]", "string"],
    columns=["boolean", "category", "Float64", "Int64", "datetime64[ns]", "string"],
)

expected_prediction_matrix_balanced_binary = pd.DataFrame(
    data=[
        [1.000, 0.2254, 0.625, 0.625, 0.625, 0.2254],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    ],
    index=["boolean", "category", "Float64", "Int64", "datetime64[ns]", "string"],
    columns=["boolean", "category", "Float64", "Int64", "datetime64[ns]", "string"],
)


def expected_prediction_matrix(
    row: Union[str, List[str]],
    col: Union[str, List[str]],
    target_rebalance: bool,
    type_: bool,
) -> pd.DataFrame:
    if not target_rebalance or type_ in ["Int64", "Float64"]:
        return expected_prediction_matrix_raw.loc[row, col]
    elif type_ in ["string", "category"]:
        return expected_prediction_matrix_balanced_multi.loc[row, col]
    elif type_ in ["boolean"]:
        return expected_prediction_matrix_balanced_binary.loc[row, col]
