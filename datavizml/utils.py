import numpy as np
import pandas as pd
from scipy import stats
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


# inter-quartile skew
def inter_quartile_skew(data):
    """A function to calculate inter-quartile skew"""

    lower, median, upper = np.quantile(data.dropna(), [0.25, 0.5, 0.75])
    middle = (upper + lower) / 2
    range_ = abs(upper - lower)
    if range_ != 0:
        feature_score = abs((median - middle)) / range_ / 2
    else:
        feature_score = 0
    feature_score_type = "Inter-quartile skew"

    return feature_score, feature_score_type


# reduce skew
def reduce_skew(df):
    """A function to transform the data to reduce skew"""
    # make all data positive
    min_ = min(df)
    if min_ <= 0:
        df_pos = df - min_ + 0.001
    else:
        df_pos = df

    # define transformers
    transformers = {
        "square": np.square,
        "square-root": np.sqrt,
        "log-2": np.log2,
        "exp-2": np.exp2,
        "yeojohnson": lambda x: stats.yeojohnson(x)[0],
    }

    # initiate outputs and skew
    skew_ = abs(df_pos.skew())
    transformer_name = None
    transformed_df = df_pos

    # evaluate all samples
    for name, trans in transformers.items():
        # calculate values for current transformer
        if name == "exp-2":
            # suppress overflow warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                temp_df = pd.Series(trans(df_pos), name=df.name)
        else:
            temp_df = pd.Series(trans(df_pos), name=df.name)

        # update if skew has been reduced
        temp_skew = abs(temp_df.skew())
        if temp_skew < skew_:
            skew_ = temp_skew
            transformer_name = name
            transformed_df = temp_df

    return transformer_name, transformed_df
