from datavizml.datavizml import SingleDistribution
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest


def test_single_with_list():
    # initialise inputs
    _, ax = plt.subplots()
    x_list = [1, 2]
    y_list = [0, 1]
    x_array = np.array([1, 2])
    y_array = np.array([0, 1])

    # check inability to initiate with lists
    with pytest.raises(TypeError):
        SingleDistribution(feature=x_list, ax=ax, target=y_array)
    with pytest.raises(TypeError):
        SingleDistribution(feature=x_array, ax=ax, target=y_list)


def test_prescribed_score():
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


# def test_single_with_boolean_pandas_with_target(capsys):
#     # initialise inputs
#     _, ax = plt.subplots()
#     x = pd.Series(
#         [False, False, True, np.nan] * 4,
#         name="feature_test",
#     )
#     y = pd.Series([0, 0, 1, 1] * 4, name="target_test")

#     # initialise object
#     sd = SingleDistribution(feature=x, ax=ax, target=y)

#     # check printing
#     print(sd, end="")
#     captured = capsys.readouterr()
#     expected = "feature: feature_test, target: target_test, score: not calculated"
#     assert expected == captured.out

#     # check missing proportion value
#     assert sd.missing_proportion == 0.25

#     # check inability to reset values
#     with pytest.raises(AttributeError):
#         sd.target = y

#     # call object
#     sd()

#     # check printing
#     print(sd, end="")
#     captured = capsys.readouterr()
#     expected = "feature: feature_test, target: target_test, score: 1.0"
#     assert expected == captured.out

#     # check score
#     assert sd.score == 1.0


def test_single_with_interger_array_without_target(capsys):
    # initialise inputs
    _, ax = plt.subplots()
    x = np.array(list(range(16 - 1)) + [np.nan]) * 1000

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax)

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: unnamed, target: no target provided, score: not calculated"
    assert expected == captured.out

    # check missing proportion value
    assert sd.missing_proportion == 1 / 16

    # check inability to reset values
    with pytest.raises(AttributeError):
        sd.feature = x

    # call object
    sd()

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: unnamed, target: no target provided, score: 0.0"
    assert expected == captured.out

    # check score
    assert sd.score == 0.0
