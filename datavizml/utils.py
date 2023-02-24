import pandas as pd


# convert to series
def to_series(input):
    """A method to check inputs are a pandas series"""
    # convert array to series
    if isinstance(input, pd.Series):
        return input
    else:
        raise TypeError(
            f"Input is of {input.__class__.__name__} type which is not valid"
        )


# convert to frame
def to_frame(input):
    """A method to convert inputs into a pandas dataframe"""
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
    """A method to classify pandas series"""
    # drop null values
    no_null = input.dropna().convert_dtypes()
    is_bool = pd.api.types.is_bool_dtype(no_null)
    is_numeric = pd.api.types.is_numeric_dtype(no_null)
    return is_bool, is_numeric, no_null.dtype
