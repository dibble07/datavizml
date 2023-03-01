# set expected prediction matrix
import pandas as pd

expected_prediction_matrix_raw = pd.DataFrame(
    data=[
        [1.000, 0.260, 0.634, 0.634, 0.260],
        [1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000],
    ],
    index=["boolean", "category", "Float64", "Int64", "string"],
    columns=["boolean", "category", "Float64", "Int64", "string"],
)

expected_prediction_matrix_balanced = pd.DataFrame(
    data=[
        [1.000, 0.043, 0.497, 0.497, 0.043],
        [1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 1.000, 1.000, 1.000],
    ],
    index=["boolean", "category", "Float64", "Int64", "string"],
    columns=["boolean", "category", "Float64", "Int64", "string"],
)


def expected_prediction_matrix(row, col, balanced):
    if balanced:
        return expected_prediction_matrix_balanced.loc[row, col]
    else:
        return expected_prediction_matrix_raw.loc[row, col]
