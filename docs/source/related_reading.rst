Distribution scoring metrics
===============

- `Predictive Power Score (PPS) <https://github.com/8080labs/ppscore#cases-and-their-score-metrics>`_ is the improvement in an evaluation metric from a naive model to a model based on the selected feature, relative to the potential improvement achieved from a perfect feature
    - For classifications, the naive model is the mode and the metric is F1 Score
    - For regressions, the naive model is the median and the metric is Mean Absolute Error
    - e.g. a model that uses the modal classes achieves an F1 score of 0.6 and a model using the selected feature achieves an F1 score of 0.9, the PPS score is 0.75