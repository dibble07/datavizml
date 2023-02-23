import matplotlib
from matplotlib import ticker
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ppscore as pps
import scipy
from statsmodels.stats.proportion import proportion_confint


# convert to series
def to_series(input, name):
    """A method to convert inputs into a pandas series"""
    # convert array to series
    if isinstance(input, pd.Series):
        return input
    elif isinstance(input, np.ndarray):
        output = np.squeeze(input)
        ndim = output.ndim
        if ndim > 1:
            raise ValueError(f"Input has {ndim} dimensions but only 1 is allowed")
        return pd.Series(output, name=name)
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
    elif isinstance(input, np.ndarray):
        return pd.DataFrame(np.squeeze(input))
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


class SingleDistribution:
    """A graphical summary of a given feature and its relationship to a target

    :param feature: Feature to be analysed
    :type feature: pandas Series, numpy array or tuple(numpy array, name)
    :param ax: Axes to plot on
    :type ax: matplotlib Axes
    :param target: Target to be predicted
    :type target: pandas Series, numpy array or tuple(numpy array, name), optional
    :param score: Precomputed score to avoid recalculation
    :type score: float, optional
    :param binning_threshold: Maximum number of distinct values in the column before binning, defaults to 12
    :type binning_threshold: int, optional
    """

    BINNING_THRESHOLD_DEFAULT = 12  # distinct values for binning
    CI_SIGNIFICANCE_DEFAULT = 0.05  # confidence interval significance
    COLOUR_FEATURE_DEFAULT = "grey"  # colour used for feature
    COLOURMAP_TARGET_DEFAULT = "tab10"  # colour map used for target

    def __init__(self, feature, ax, target=None, score=None, binning_threshold=None):
        """Constructor method"""
        # input variables
        self.ax_feature = ax
        self.feature = feature
        self.has_target = target is not None
        if self.has_target:
            self.target = target
        if score is not None:
            self.score = score
        self.binning_threshold = (
            binning_threshold
            if binning_threshold
            else SingleDistribution.BINNING_THRESHOLD_DEFAULT
        )

        # check input
        if self.has_target:
            if self.feature.shape[0] != self.target.shape[0]:
                raise ValueError(
                    f"Dimension mismatch, feature has {self.feature.shape[0]} elements but the target has {self.target.shape[0]}"
                )

        # classify inputs
        (
            self.feature_is_bool,
            self.feature_is_numeric,
            self.feature_dtype,
        ) = classify_type(self.feature)
        if self.has_target:
            (
                self.target_is_bool,
                self.target_is_numeric,
                self.target_dtype,
            ) = classify_type(self.target)
            if self.target_is_numeric and not self.target_is_bool:
                self.target_type = "regression"
            else:
                self.target_type = "classification"

        # supplementary/reusable variables
        missing_proportion = self.feature.isna().value_counts(normalize=True)
        self.__missing_proportion = (
            missing_proportion[True] if True in missing_proportion.index else 0
        )
        if self.has_target:
            self.ax_target = self.ax_feature.twinx()

    def __str__(self):
        """Returns a string representation of the instance

        :return: A string containing: feature name and data type; target name and data type; and relationship score if available
        :rtype: str
        """

        # conditional strings
        target_val = (
            f"{self.target.name} ({self.target_dtype} - {self.target_type})"
            if self.has_target
            else "no target provided"
        )
        score_val = f"{self.score:0.3f}" if hasattr(self, "score") else "not calculated"

        # attribute related strings
        feature_str = f"feature: {self.feature.name} ({self.feature_dtype})"
        target_str = f"target: {target_val}"
        score_str = f"score: {score_val}"

        return ", ".join([feature_str, target_str, score_str])

    def __call__(
        self,
        ci_significance=CI_SIGNIFICANCE_DEFAULT,
        colour_feature=COLOUR_FEATURE_DEFAULT,
        colourmap_target=COLOURMAP_TARGET_DEFAULT,
    ):
        """Generates and decorates the plot

        : param ci_significance: Significance level for the target confidence interval calculation, defaults to 0.05
        : type ci_significance: float, optional
        : param colour_feature: Colour used for the feature plot, defaults to "grey"
        : type colour_feature: str, optional
        : param colourmap_target: Colour map used for the target plot, defaults to "tab10"
        : type colourmap_target: str, optional
        """

        # load colourmap
        self.__cmap = matplotlib.colormaps[colourmap_target]

        # calculate score
        if not hasattr(self, "score"):
            self.calculate_score()

        # summarise feature
        if not hasattr(self, "feature_summary"):
            self.summarise_feature()

        # plot feature frequency
        markerline, stemlines, baseline = self.ax_feature.stem(
            self.__feature_summary.index, self.__feature_summary["count"]
        )
        markerline.set_color(colour_feature)
        stemlines.set_color(colour_feature)
        baseline.set_color(colour_feature)

        # plot target values and uncertainty
        if self.has_target:
            # regression specific calculations
            if self.target_type == "regression":
                z_crit = scipy.stats.norm.ppf(1 - ci_significance / 2)
                ci_diff_all = {None: self.__feature_summary["std"] * z_crit}
                y_plot_all = {None: self.__feature_summary["mean"]}

            # classification specific calculations
            elif self.target_type == "classification":
                # calculate values for each class
                ci_diff_all = {}
                y_plot_all = {}
                for class_name, values in self.__feature_summary.drop(
                    columns="count"
                ).items():
                    mean = values / self.__feature_summary["count"]
                    ci_lo, ci_hi = proportion_confint(
                        values, self.__feature_summary["count"], ci_significance
                    )
                    ci_diff_all[class_name] = 100 * np.concatenate(
                        (
                            (mean - ci_lo).values.reshape(1, -1),
                            (ci_hi - mean).values.reshape(1, -1),
                        )
                    )
                    y_plot_all[class_name] = mean * 100

                # drop false class for boolean
                if self.target_is_bool:
                    ci_diff_all.pop(False)
                    y_plot_all.pop(False)

            # plot errorbars
            for (class_name, ci_diff), (_, y_plot), colour_target in zip(
                ci_diff_all.items(),
                y_plot_all.items(),
                [self.__cmap(i) for i in range(len(y_plot_all))],
            ):
                self.ax_target.errorbar(
                    self.__feature_summary.index,
                    y_plot,
                    yerr=ci_diff,
                    color=colour_target,
                    elinewidth=2,
                    capsize=3,
                    capthick=2,
                    label=class_name,
                    ls="",
                    marker="D",
                    markersize=3,
                )

        # decorate x axis
        self.ax_feature.set_xlabel(self.feature.name)
        if self.feature_is_numeric and not self.feature_is_bool:
            _, ax_max = self.ax_feature.get_xlim()
            if ax_max > 1000:
                self.ax_feature.xaxis.set_major_formatter(
                    ticker.StrMethodFormatter("{x:,.0f}")
                )
        else:
            self.ax_feature.tick_params(axis="x", labelrotation=90)

        # decorate first y axis
        self.ax_feature.set_ylabel("Frequency")
        self.ax_feature.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

        # decorate second y axis
        if self.has_target:
            twin_y_colour = (
                "k"
                if len(y_plot_all) > 1 and not self.target_is_bool
                else colour_target
            )
            self.ax_target.set_ylabel(self.target.name, color=twin_y_colour)
            self.ax_target.tick_params(axis="y", labelcolor=twin_y_colour)
            if self.target_type == "classification":
                self.ax_target.yaxis.set_major_formatter(ticker.PercentFormatter())
                if not self.target_is_bool:
                    self.ax_target.legend()

        # add title
        self.ax_feature.set_title(
            f"{self.__score_type} = {self.score:.2f}\n({100*self.missing_proportion:.1f}% missing)"
        )

    def calculate_score(self):
        """Calculate the score for the feature based on its predictive power or skewness.

        If a target has been specified for the feature, the score is calculated based on the predictive power score
        (PPS). Otherwise, the score is calculated based on the skewness of the median towards quartiles (for
        numerical features) or the categorical skew (for categorical features).
        """

        if self.has_target:
            # calculate predictive power score
            self.score = pps.score(
                df=pd.concat([self.feature, self.target], axis=1),
                x=self.feature.name,
                y=self.target.name,
                sample=None,
                invalid_score=np.nan,
            )["ppscore"]
            self.__score_type = "PPS"

        else:
            if self.feature_is_numeric and not self.feature_is_bool:
                # calculate skew of median towards quartiles
                lower, median, upper = np.quantile(
                    self.feature.dropna(), [0.25, 0.5, 0.75]
                )
                middle = (upper + lower) / 2
                range_ = abs(upper - lower)
                self.score = abs((median - middle)) / range_ / 2
                self.__score_type = "Inter-quartile skew"
            else:
                self.score = self.feature.value_counts(normalize=True).max()
                self.__score_type = "Categorical skew"

    def summarise_feature(self):
        """Summarise the feature by calculating summary statistics for each distinct value and binning if there are too many distinct values"""
        # join feature and target intro single dataframe
        if self.has_target:
            all_data = pd.concat([self.feature, self.target], axis=1)
        else:
            all_data = self.feature.to_frame()

        # bin target variable if there are too many distinct values
        if self.feature.nunique() > self.binning_threshold and self.feature_is_numeric:
            bin_boundaries = np.linspace(
                self.feature.min(), self.feature.max(), self.binning_threshold + 1
            )
            all_data[self.feature.name] = pd.cut(self.feature, bin_boundaries).apply(
                lambda x: x.mid
            )

        # calculate summary statistics for each distinct target variable
        if self.has_target:
            if self.target_type == "regression":
                self.__feature_summary = all_data.groupby(self.feature.name).agg(
                    {"count", "mean", "std"}
                )
                self.__feature_summary.columns = (
                    self.__feature_summary.columns.droplevel()
                )
            elif self.target_type == "classification":
                self.__feature_summary = pd.pivot_table(
                    all_data.value_counts().to_frame("count"),
                    values="count",
                    index=self.feature.name,
                    columns=self.target.name,
                    fill_value=0,
                )
                self.__feature_summary["count"] = self.__feature_summary.sum(axis=1)
        else:
            self.__feature_summary = all_data.value_counts().to_frame("count")
            self.__feature_summary.index = self.__feature_summary.index.map(
                lambda x: x[0]
            )

        # convert index to string from boolean for printing purposes
        if self.feature_is_bool:
            self.__feature_summary.index = self.__feature_summary.index.map(
                {True: "True", False: "False"}
            )

    # feature getter
    @property
    def feature(self):
        """The feature data"""
        return self.__feature

    # feature setter
    @feature.setter
    def feature(self, feature):
        if hasattr(self, "feature"):
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # unpack value and name
            if isinstance(feature, tuple):
                feature, name = feature
            else:
                name = "unnamed_feature"

            # convert to series and set
            self.__feature = to_series(feature, name=name)

    # target getter
    @property
    def target(self):
        """The target data"""
        return self.__target

    # target setter
    @target.setter
    def target(self, target):
        if hasattr(self, "target") or not self.has_target:
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # unpack value and name
            if isinstance(target, tuple):
                target, name = target
            else:
                name = "unnamed_target"

            # convert to series and set
            self.__target = to_series(target, name=name)

    # score getter
    @property
    def score(self):
        """The score value"""
        return self.__score

    # score setter
    @score.setter
    def score(self, score):
        if hasattr(self, "score"):
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # only accept numerical values
            if isinstance(score, (int, float)):
                self.__score = score
            else:
                raise TypeError(
                    f"score is of {score.__class__.__name__} type which is not valid"
                )

    # missing proportion getter
    @property
    def missing_proportion(self):
        """The proportion of values that are missing"""
        return self.__missing_proportion


class ExploratoryDataAnalysis:
    """A graphical summary of all given features and their relationship to a target

    :param data: Features to be analysed
    :type data: pandas Series
    :param target: Target to be predicted
    :type target: pandas Series, optional
    :param ncols: Number of columns to use in figure
    :type ncols: float, optional
    """

    FIGURE_WIDTH = 18  # width of figure
    AXES_HEIGHT = 3  # height of each axis

    def __init__(
        self,
        data,
        ncols,
        target=None,
        figure_width=FIGURE_WIDTH,
        axes_height=AXES_HEIGHT,
    ):
        """Constructor method"""
        # input variables
        self.data = data
        self.has_target = target is not None
        if self.has_target:
            self.target = target
        self.__ncols = ncols
        self.__figure_width = figure_width
        self.__axes_height = axes_height

        # calculate general use variables
        self.__nrows = -(-(self.data.shape[1]) // self.__ncols)

        # classify inputs
        self.data_dtypes = set([classify_type(x)[2] for _, x in self.data.items()])
        if self.has_target:
            _, _, self.target_dtype = classify_type(self.target)

        # check input
        if self.has_target:
            if self.data.shape[0] != self.target.shape[0]:
                raise ValueError(
                    f"Dimension mismatch, features have {self.data.shape[0]} elements but the target has {self.target.shape[0]}"
                )

        # initialise figure and axes
        self.init_figure()

        # initialise figure and axes
        self.init_single_distributions()

    def __str__(self):
        """Returns a string representation of the instance

        :return: A string containing: feature name and data type; target name and data type; and relationship score if available
        :rtype: str
        """

        # conditional strings
        feature_vals = (
            ", ".join(self.data.columns),
            ", ".join(sorted([str(x) for x in self.data_dtypes])),
        )
        target_val = (
            f"{self.target.name} ({self.target_dtype})"
            if self.has_target
            else "no target provided"
        )

        # attribute related strings
        feature_str = f"features: {feature_vals[0]} ({feature_vals[1]})"
        target_str = f"target: {target_val}"

        return "\n".join([feature_str, target_str])

    def __getitem__(self, ind):
        """Get the distribution plot at the given index

        :param ind: The index of the distribution plot to retrieve
        :type ind: int

        :return: The SingleDistribution object at the given index, or None if the index is out of range
        :rtype: SingleDistribution or None
        """
        return self.single_distributions[ind]

    def __call__(self):
        """Generates and decorates the plots for each feature

        :return: A figure with the plots for each feature
        :rtype: matplotlib.figure.Figure
        """
        # call the plot for each object
        for plot in self:
            plot()

        return self.fig

    # initialise figure
    def init_figure(self):
        """Initialise a figure with the required size and axes for the exploratory data analysis"""
        # create figure of required size with the required axes
        figsize = (self.__figure_width, self.__axes_height * self.__nrows)
        fig, ax = plt.subplots(
            nrows=self.__nrows, ncols=self.__ncols, squeeze=False, figsize=figsize
        )

        # assign to object
        self.fig = fig
        self.ax = ax

    # initialise distribution plot
    def init_single_distributions(self):
        """Initialise a single distribution object for each feature"""
        # initialise all single distribution objects
        self.single_distributions = []
        for (_, feature), ax in zip(self.data.items(), self.ax.flatten()):
            self.single_distributions.append(
                SingleDistribution(
                    feature=feature,
                    target=self.target if self.has_target else None,
                    ax=ax,
                )
            )

    # data getter
    @property
    def data(self):
        """The feature data"""
        return self.__data

    # data setter
    @data.setter
    def data(self, data):
        if hasattr(self, "data"):
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # convert to series and set
            self.__data = to_frame(data)

    # target getter
    @property
    def target(self):
        """The target data"""
        return self.__target

    # target setter
    @target.setter
    def target(self, target):
        if hasattr(self, "target") or not self.has_target:
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # unpack value and name
            if isinstance(target, tuple):
                target, name = target
            else:
                name = "unnamed_target"

            # convert to series and set
            self.__target = to_series(target, name=name)
