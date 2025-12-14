Distribution scoring metrics
============================

- `Predictive Power Score (PPS) <https://github.com/8080labs/ppscore#cases-and-their-score-metrics>`_
    - The improvement in an evaluation metric from a naive model to a model based on the selected feature, relative to the potential improvement achieved from a perfect feature
        - For classifications, the naive model is the better of the mode or the shuffled target values and the metric is F1 Score
        - For regressions, the naive model is the median and the metric is Mean Absolute Error
    - e.g. a model that uses the modal classes achieves an F1 score of 0.6 and a model using the selected feature achieves an F1 score of 0.9, the PPS score is 0.75
- Categorical skew (categorical features)
    - The proportion of data points with the modal value
    - e.g. a dataset of 6 class `A`, 3 class `B` and one class `C` has 60% of data points as the modal value
- Inter-quartile skew (numerical features)
    - The difference between the median and the midpoint of the upper and lower quantiles, as a proportion of half the inter-quartile range
    - e.g. if the lower and upper quartile were 1 and 3 and the median was 1.5, the inter-quartile skew would be 0.5