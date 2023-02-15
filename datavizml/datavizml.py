from matplotlib import ticker
import numpy as np
import pandas as pd
import ppscore as pps
from statsmodels.stats.proportion import proportion_confint


class SingleDistribution:
    """A graphical summary of a given feature and its relationship to a target

    :param feature: Feature to be analysed
    :type feature: pandas Series
    :param ax: Axes to plot on
    :type ax: matplotlib Axes
    :param target: Target to be predicted
    :type target: pandas Series, optional
    :param score: Precomputed score to avoid recalculation
    :type score: float, optional
    :param binning_threshold: Maximum number of distinct values in the column before binning, defaults to 12
    :type binning_threshold: int, optional
    """

    BINNING_THRESHOLD_DEFAULT = 12  # distinct values for binning
    CI_SIGNIFICANCE_DEFAULT = 0.05  # confidence interval significance
    COLOUR_FEATURE_DEFAULT = "grey"  # colour used for feature
    COLOUR_TARGET_DEFAULT = "tab:blue"  # colour used for target

    def __init__(self, feature, ax, target=False, score=False, binning_threshold=False):
        """Constructor method"""
        # input variables
        self.ax_feature = ax
        self.feature = feature
        self.has_target = target is not False
        if self.has_target:
            self.target = target
        if score is not False:
            self.score = score
        self.binning_threshold = (
            binning_threshold
            if binning_threshold
            else SingleDistribution.BINNING_THRESHOLD_DEFAULT
        )

        # supplementary/reusable variables
        feature_no_null = self.feature.dropna().convert_dtypes()
        self.is_bool = pd.api.types.is_bool_dtype(feature_no_null)
        self.is_numeric = pd.api.types.is_numeric_dtype(feature_no_null)
        missing_proportion = self.feature.isna().value_counts(normalize=True)
        self.__missing_proportion = (
            missing_proportion[True] if True in missing_proportion.index else 0
        )
        if self.has_target:
            self.ax_target = self.ax_feature.twinx()

    def __str__(self):
        """Returns a string representation of the instance

        :return: A string containing the feature name, target name, and score if available
        :rtype: str
        """

        # conditional strings
        target_val = self.target.name if self.has_target else "no target provided"
        score_val = self.score if hasattr(self, "score") else "not calculated"

        # attribute related strings
        feature_str = f"feature: {self.feature.name}"
        target_str = f"target: {target_val}"
        score_str = f"score: {score_val}"

        return ", ".join([feature_str, target_str, score_str])

    def __call__(
        self,
        ci_significance=CI_SIGNIFICANCE_DEFAULT,
        colour_feature=COLOUR_FEATURE_DEFAULT,
        colour_target=COLOUR_TARGET_DEFAULT,
    ):
        """Generates and decorates the plot

        : param ci_significance: Significance level for the target confidence interval calculation, defaults to 0.05
        : type ci_significance: float, optional
        : param colour_feature: Colour used for the feature plot, defaults to "grey"
        : type colour_feature: str, optional
        : param colour_target: Colour used for the target plot, defaults to "tab:blue"
        : type colour_target: str, optional
        """

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

        # plot target proportion
        if self.has_target:
            ci_lo, ci_hi = proportion_confint(
                self.__feature_summary["mean"] * self.__feature_summary["count"],
                self.__feature_summary["count"],
                ci_significance,
            )
            ci_diff = np.concatenate(
                (
                    (self.__feature_summary["mean"] - ci_lo).values.reshape(1, -1),
                    (ci_hi - self.__feature_summary["mean"]).values.reshape(1, -1),
                )
            )
            self.ax_target.errorbar(
                self.__feature_summary.index,
                self.__feature_summary["mean"] * 100,
                yerr=ci_diff * 100,
                color=colour_target,
                elinewidth=2,
                capsize=3,
                capthick=2,
                ls="",
                marker="D",
                markersize=3,
            )

        # decorate x axis
        self.ax_feature.set_xlabel(self.feature.name)
        if self.is_numeric and not self.is_bool:
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
            self.ax_target.set_ylabel(self.target.name, color=colour_target)
            self.ax_target.tick_params(axis="y", labelcolor=colour_target)
            self.ax_target.yaxis.set_major_formatter(ticker.PercentFormatter())

        # add title
        if self.has_target:
            self.ax_feature.set_title(
                f"PPS = {self.score:.2f}\n({100*self.missing_proportion:.1f}% missing)"
            )
        else:
            self.ax_feature.set_title(
                f"Inter-quartile skew = {self.score:.2f}\n({100*self.missing_proportion:.1f}% missing)"
            )

    def calculate_score(self):
        """Calculate the score for the feature based on its predictive power or skewness.

        If a target has been specified for the feature, the score is calculated based on the predictive power score
        (PPS). Otherwise, the score is calculated based on the skewness of the median towards quartiles.
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

        else:
            # calculate skew of median towards quartiles
            lower, median, upper = np.quantile(self.feature.dropna(), [0.25, 0.5, 0.75])
            middle = (upper + lower) / 2
            range_ = abs(upper - lower)
            self.score = abs((median - middle)) / range_ / 2

    def summarise_feature(self):
        """Summarise the feature by calculating summary statistics for each distinct value and binning if there are too many distinct values"""
        # join feature and target intro single dataframe
        if self.has_target:
            self.__feature_summary = pd.concat([self.feature, self.target], axis=1)
        else:
            self.__feature_summary = self.feature.to_frame()

        # bin target variable if there are too many distinct values
        if self.feature.nunique() > self.binning_threshold and self.is_numeric:
            bin_boundaries = np.linspace(
                self.feature.min(), self.feature.max(), self.binning_threshold + 1
            )
            self.__feature_summary[self.feature.name] = pd.cut(
                self.feature, bin_boundaries
            ).apply(lambda x: x.mid)
        # calculate summary statistics for each distinct target variable
        if self.has_target:
            self.__feature_summary = self.__feature_summary.groupby(
                self.feature.name
            ).agg({"count", "mean", "std"})
            self.__feature_summary.columns = self.__feature_summary.columns.droplevel()
        else:
            self.__feature_summary = self.__feature_summary.value_counts().to_frame(
                "count"
            )
            self.__feature_summary.index = self.__feature_summary.index.map(
                lambda x: x[0]
            )

        # convert index to string from boolean for printing purposes
        if self.is_bool:
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
            # convert to series and set
            self.__feature = self.__to_series(feature)

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
            # convert to series and set
            self.__target = self.__to_series(target)

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

    # convert to series
    @staticmethod
    def __to_series(input):
        """A method to convert inputs into a pandas series"""
        # extract original class name
        class_name = input.__class__.__name__

        # convert array to series
        if isinstance(input, np.ndarray):
            output = np.squeeze(input)
            output = pd.Series(output, name="unnamed")
        else:
            output = input

        # only accept pandas series object
        if isinstance(output, pd.Series):
            return output
        else:
            raise TypeError(f"input is of {class_name} type which is not valid")
