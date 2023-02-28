# set expected prediction matrix
import pandas as pd

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
