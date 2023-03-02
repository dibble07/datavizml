import numpy as np
import pandas as pd
import warnings


# convert to series
def to_series(input):
    """A function to check inputs are a pandas series"""
    # convert array to series
    if isinstance(input, pd.Series):
        return input
    else:
        raise TypeError(
            f"Input is of {input.__class__.__name__} type which is not valid"
        )


# convert to frame
def to_frame(input):
    """A function to convert inputs into a pandas dataframe"""
    # convert array to frame
    if isinstance(input, pd.DataFrame):
        return input
    elif isinstance(input, pd.Series):
        return input.to_frame()
    else:
        raise TypeError(
            f"Input is of {input.__class__.__name__} type which is not valid"
        )


# classify type of data
def classify_type(input):
    """A function to classify pandas series"""
    # drop null values
    no_null = input.dropna().convert_dtypes()
    is_bool = pd.api.types.is_bool_dtype(no_null)
    is_numeric = pd.api.types.is_numeric_dtype(no_null)
    return is_bool, is_numeric, no_null.dtype


# rebalance classes
def class_rebalance(x, y):
    """A function to reduce class imbalance"""

    # combine into one dataset
    df = pd.concat([x, y], axis=1)

    # calculate how much each class needs to be duplicate
    class_counts = y.value_counts()
    class_multiples = (class_counts.max() / class_counts).round().astype(int)

    # repeat each class's data
    class_balanced_data = []
    for class_, multiple in class_multiples.items():
        df_class = df.loc[y == class_]
        class_balanced_data.extend([df_class] * multiple)

    # create combined rebalanced dataframe
    df_balanced = pd.concat(class_balanced_data).sample(
        frac=1, replace=False, random_state=42
    )
    x_balanced = df_balanced.drop(columns=y.name)
    y_balanced = df_balanced[y.name]

    # check results
    len_before = len(df.dropna().drop_duplicates())
    len_after = len(df_balanced.dropna().drop_duplicates())
    assert len_before == len_after
    size_increase = len(df_balanced) / len(df)
    if size_increase > 2:
        warnings.warn(
            f"Balancing the classes has increases the dataset by a factor of {size_increase:.1f}, this will reduce target score calculation speed"
        )

    return x_balanced, y_balanced
