import matplotlib.pyplot as plt
import pandas as pd
from datavizml.datavizml import SingleDistribution


def test_1(capsys):
    # initialise inputs
    fig, ax = plt.subplots()
    x = pd.Series([0, 1, 2, 3], name="feature_test")
    y = pd.Series(["f", "f", "t", "t"], name="target_test")

    # initialise object
    sd = SingleDistribution(feature=x, ax=ax, target=y)

    # check printing
    print(sd, end="")
    captured = capsys.readouterr()
    expected = "feature: feature_test, target: target_test, score: not calculated"
    assert expected in captured.out
