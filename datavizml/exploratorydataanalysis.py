from typing import Any, Dict, Optional, Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from datavizml import ppscore as pps
from datavizml import singledistribution as sd
from datavizml import utils


class ExploratoryDataAnalysis:
    """A graphical summary of all given features and their relationship to a target

    :param data: Features to be analysed
    :type data: pandas Series of pandas DataFrame
    :param ncols: Number of columns to use in figure
    :type ncols: float, optional
    :param data_deskew: Reduce data skew, trialling: squaring, rooting, logging, exponents and Yeo-Johnson
    :type data_deskew: bool for all features, string or list of strings for selective features, dict of column names and transforms for selective features and transforms, optional
    :param target: Target to be predicted
    :type target: pandas Series, optional
    :param target_rebalance: Rebalance target
    :type target_rebalance: bool, optional
    :param metric: Metric used for prevalence, "count" or "prop" (default)
    :type metric: string, optional
    :param prediction_matrix_full: Full or reduced prediction matrix
    :type prediction_matrix_full: bool, optional
    :param figure_width: Width of figure
    :type figure_width: int, optional
    :param axes_height: Height of axes
    :type axes_height: int, optional
    """

    FIGURE_WIDTH = 18  # width of figure
    AXES_HEIGHT = 3  # height of each axis

    def __init__(
        self,
        data: Any,
        ncols: int,
        data_deskew: Union[bool, dict, list, str] = False,
        target: Optional[Any] = None,
        target_rebalance: bool = False,
        metric: str = "prop",
        prediction_matrix_full: bool = False,
        figure_width: Union[int, float] = FIGURE_WIDTH,
        axes_height: Union[int, float] = AXES_HEIGHT,
    ) -> None:
        """Constructor method"""
        # input variables
        self.data = data
        self.__data_deskew = (
            [data_deskew] if isinstance(data_deskew, str) else data_deskew
        )
        self.__has_target = target is not None
        if self.__has_target:
            self.target = target
            self.__target_rebalance = target_rebalance
        self.__ncols = ncols
        self.__prediction_matrix_full = prediction_matrix_full
        self.__figure_width = figure_width
        self.__axes_height = axes_height
        self.__metric = metric

        # calculate general use variables
        self.__nrows = -(-(self.data.shape[1]) // self.__ncols)

        # classify inputs
        self.__data_dtypes = set(
            [utils.classify_type(x)[3] for _, x in self.data.items()]
        )
        if self.__has_target:
            (
                self.__target_is_bool,
                self.__target_is_numeric,
                _,
                self.__target_dtype,
            ) = utils.classify_type(self.target)
            if self.__target_is_numeric and not self.__target_is_bool:
                self.__target_type = "regression"
            else:
                self.__target_type = "classification"

        # check input
        if self.__has_target:
            if self.data.shape[0] != self.target.shape[0]:
                raise ValueError(
                    f"Dimension mismatch, features have {self.data.shape[0]} elements but the target has {self.target.shape[0]}"
                )

        # initialise figure and axes
        self.__init_figure()

        # calculate prediction matrix
        self.__calculate_prediction_matrix()

        # initialise figure and axes
        self.__init_single_distributions()

    def __str__(self) -> str:
        """Returns a string representation of the instance

        :return: A string containing: feature name and data type; target name and data type; and relationship score if available
        :rtype: str
        """

        # conditional strings
        feature_vals = (
            ", ".join(self.data.columns),
            ", ".join(sorted([str(x) for x in self.__data_dtypes])),
        )
        target_val = (
            f"{self.target.name} ({self.__target_dtype})"
            if self.__has_target
            else "no target provided"
        )

        # attribute related strings
        feature_str = f"features: {feature_vals[0]} ({feature_vals[1]})"
        target_str = f"target: {target_val}"

        return "\n".join([feature_str, target_str])

    def __getitem__(self, ind: int) -> sd.SingleDistribution:
        """Get the distribution plot at the given index

        :param ind: The index of the distribution plot to retrieve
        :type ind: int

        :return: The SingleDistribution object at the given index, or None if the index is out of range
        :rtype: SingleDistribution or None
        """
        return self.single_distributions[ind]

    def __call__(self) -> matplotlib.figure.Figure:
        """Generates and decorates the plots for each feature

        :return: A figure with the plots for each feature
        :rtype: matplotlib.figure.Figure
        """
        # call the plot for each object
        for plot in (pbar := tqdm(self, total=len(self.single_distributions))):  # type: ignore
            pbar.set_description(plot.feature.name)
            plot()

        return self.fig

    # initialise figure
    def __init_figure(self) -> None:
        """Initialise a figure with the required size and axes for the exploratory data analysis"""
        # create figure of required size with the required axes
        figsize = (self.__figure_width, self.__axes_height * self.__nrows)
        fig, ax = plt.subplots(
            nrows=self.__nrows, ncols=self.__ncols, squeeze=False, figsize=figsize
        )

        # assign to object
        self.fig: matplotlib.figure.Figure = fig
        self.ax = ax

    # calculate prediction matrix
    def __calculate_prediction_matrix(self) -> None:
        "Calculate prediction matrix for specified combinations of features/targets"
        # combine feature and target
        if self.__has_target:
            # rebalance classes
            if self.__target_type == "classification" and self.__target_rebalance:
                x_balanced, y_balanced = utils.class_rebalance(self.data, self.target)
                df = pd.concat([x_balanced, y_balanced], axis=1)
            else:
                df = pd.concat([self.data, self.target], axis=1)

        else:
            df = self.data

        # calculate full matrix
        if self.__prediction_matrix_full:
            self.__prediction_matrix = pps.calculate(
                df=df,
            )
        else:
            # calculate reduced matrix
            if self.__has_target:
                self.__prediction_matrix = pps.calculate(
                    df=df,
                    y=self.target.name,
                )
            else:
                self.__prediction_matrix = None

    # initialise distribution plot
    def __init_single_distributions(self) -> None:
        """Initialise a single distribution object for each feature"""
        # initialise all single distribution objects
        self.single_distributions = []
        for (_, feature), ax in zip(self.data.items(), self.ax.flatten()):
            if isinstance(self.__data_deskew, bool):
                feature_deskew = self.__data_deskew
            elif isinstance(self.__data_deskew, list):
                feature_deskew = feature.name in self.__data_deskew
            elif isinstance(self.__data_deskew, dict):
                if feature.name in self.__data_deskew:
                    feature_deskew = self.__data_deskew[feature.name]
                else:
                    feature_deskew = False
            self.single_distributions.append(
                sd.SingleDistribution(
                    feature=feature,
                    ax=ax,
                    feature_deskew=feature_deskew,
                    target=self.target if self.__has_target else None,
                    target_score=(
                        self.prediction_matrix.pivot(
                            index="x", columns="y", values="ppscore"
                        ).loc[feature.name, self.target.name]
                        if self.__has_target
                        else None
                    ),
                    metric=self.__metric,
                )
            )

    # create summary dataframe
    def summary(self) -> pd.DataFrame:
        """Summarise analysis

        :return: A dataframe summarising each of the features and their relationship to the target
        :rtype: pd.DataFrame
        """
        data = [sd.to_dict() for sd in self.single_distributions]
        return pd.DataFrame(data=data)

    # create prediction power plot
    def prediction_score_plot(self, ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
        """Plot the prediction scores as a heatmap

        :param ax: Axes to plot on
        :type ax: matplotlib Axes

        :return: The heatmap plot
        :rtype: matplotlib Axes
        """
        # extract data and plot heatmap
        if self.prediction_matrix is not None:
            data = self.prediction_matrix.rename(
                columns={"x": "x (predictor)", "y": "y (predictee)"}
            )
            data = data.pivot(
                index="x (predictor)", columns="y (predictee)", values="ppscore"
            )
            sns.heatmap(
                data=data,
                vmin=0,
                vmax=1,
                cmap="GnBu",
                annot=True,
                fmt=".2f",
                ax=ax,
            )
        else:
            raise TypeError(
                f"No appropriate matrix is present. This most likely is because a reduced dataframe was calculated with no target"
            )

        return ax

    # data getter
    @property
    def data(self) -> Union[pd.Series, pd.DataFrame]:
        """The feature data"""
        return self.__data

    # data setter
    @data.setter
    def data(self, data: Any) -> None:
        if hasattr(self, "data"):
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # convert to series and set
            self.__data = utils.to_frame(data)

    # target getter
    @property
    def target(self) -> pd.Series:
        """The target data"""
        self.__target: pd.Series
        return self.__target

    # target setter
    @target.setter
    def target(self, target: Any) -> None:
        if hasattr(self, "target") or not self.__has_target:
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # convert to series and set
            self.__target = utils.to_series(target)

    # prediction matrix getter
    @property
    def prediction_matrix(self) -> pd.DataFrame:
        """The prediction matrix data"""
        return self.__prediction_matrix
