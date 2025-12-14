import logging
from typing import Any, Dict, Optional, Union

import matplotlib
import numpy as np
import pandas as pd
import scipy
from matplotlib import dates, ticker
from statsmodels.stats.proportion import proportion_confint

from datavizml import ppscore as pps
from datavizml import utils


class SingleDistribution:
    """A graphical summary of a given feature and its relationship to a target

    :param feature: Feature to be analysed
    :type feature: pandas Series
    :param ax: Axes to plot on
    :type ax: matplotlib Axes
    :param feature_deskew: reduce feature skew, trialling: squaring, rooting, logging, exponents and Yeo-Johnson
    :type feature_deskew: bool, optional
    :param target: Target to be predicted
    :type target: pandas Series, optional
    :param target_score: Precomputed score to avoid recalculation
    :type target_score: float, optional
    :param target_rebalance: reduce class imbalance in target score
    :type target_rebalance: bool, optional
    :param binning_threshold: Maximum number of distinct values in the column before binning, defaults to 12
    :type binning_threshold: int, optional
    :param metric: Metric used for prevalence, "count" or "prop" (default)
    :type metric: string, optional
    """

    BINNING_THRESHOLD_DEFAULT = 12  # distinct values for binning
    CI_SIGNIFICANCE_DEFAULT = 0.05  # confidence interval significance
    COLOUR_FEATURE_DEFAULT = "grey"  # colour used for feature
    COLOURMAP_TARGET_DEFAULT = "tab10"  # colour map used for target

    def __init__(
        self,
        feature: Any,
        ax: Any,
        feature_deskew: Union[bool, list, str] = False,
        target: Optional[Any] = None,
        target_score: Optional[float] = None,
        target_rebalance: bool = False,
        binning_threshold: Optional[int] = None,
        metric: str = "prop",
    ) -> None:
        """Constructor method"""
        # input variables
        self.ax_feature = ax
        self.__feature_deskew = feature_deskew
        self.feature = feature
        self.__has_target = target is not None
        if self.__has_target:
            self.target = target
            self.__target_rebalance = target_rebalance
            if self.feature.name == self.target.name:
                # clear target if the same as feature
                del self.__target
                self.__has_target = False
        if isinstance(target_score, (int, float, np.integer, np.floating)):
            self.__target_score = target_score
            self.__target_score_type = "PPS"
        elif target_score is not None:
            raise TypeError(
                f"target_score is of {target_score.__class__.__name__} type which is not valid"
            )
        self.__binning_threshold = (
            binning_threshold
            if binning_threshold
            else SingleDistribution.BINNING_THRESHOLD_DEFAULT
        )
        self.__metric = metric

        # check input
        if self.__has_target:
            if self.feature.shape[0] != self.target.shape[0]:
                raise ValueError(
                    f"Dimension mismatch, feature has {self.feature.shape[0]} elements but the target has {self.target.shape[0]}"
                )

        # classify inputs
        (
            self.__feature_is_bool,
            self.__feature_is_numeric,
            self.__feature_is_datetime,
            self.__feature_dtype,
        ) = utils.classify_type(self.feature)
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

        # supplementary/reusable variables
        self.__feature_nunique = self.feature.nunique(dropna=False)
        missing_proportion = self.feature.isna().value_counts(normalize=True)
        self.__missing_proportion = (
            missing_proportion[True] if True in missing_proportion.index else 0
        )
        if self.__has_target:
            self.ax_target = self.ax_feature.twinx()

    def __str__(self) -> str:
        """Returns a string representation of the instance

        :return: A string containing the feature and target name and their data types
        :rtype: str
        """

        # conditional strings
        target_val = (
            f"{self.target.name} ({self.__target_dtype} - {self.__target_type})"
            if self.__has_target
            else "no target provided"
        )

        # attribute related strings
        feature_str = f"feature: {self.feature.name} ({self.__feature_dtype})"
        target_str = f"target: {target_val}"

        return ", ".join([feature_str, target_str])

    def __call__(
        self,
        ci_significance: float = CI_SIGNIFICANCE_DEFAULT,
        colour_feature: str = COLOUR_FEATURE_DEFAULT,
        colourmap_target: str = COLOURMAP_TARGET_DEFAULT,
    ) -> None:
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

        # calculate target score
        if not hasattr(self, "_SingleDistribution__target_score"):
            self.calculate_target_score()

        # calculate feature score
        if not hasattr(self, "_SingleDistribution__feature_score"):
            self.calculate_feature_score()

        # summarise feature
        if not hasattr(self, "_SingleDistribution__feature_summary"):
            self.summarise_feature()

        # plot feature frequency
        markerline, stemlines, baseline = self.ax_feature.stem(
            self.__feature_summary.index, self.__feature_summary[self.__metric]
        )
        markerline.set_color(colour_feature)
        stemlines.set_color(colour_feature)
        baseline.set_color(colour_feature)

        # plot target values and uncertainty
        if self.__has_target:
            ci_diff_all: Dict[Any, Any]
            y_plot_all: Dict[Any, Any]
            # regression specific calculations
            if self.__target_type == "regression":
                z_crit = scipy.stats.norm.ppf(1 - ci_significance / 2)
                ci_diff_all = {None: self.__feature_summary["std"] * z_crit}
                y_plot_all = {None: self.__feature_summary["mean"]}

            # classification specific calculations
            elif self.__target_type == "classification":
                # calculate values for each class
                ci_diff_all = {}
                y_plot_all = {}
                for class_name, values in self.__feature_summary.drop(
                    columns=["count", "prop"]
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
                if self.__target_is_bool:
                    del ci_diff_all[False]
                    del y_plot_all[False]

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
        if self.__feature_is_numeric and not self.__feature_is_bool:
            self.ax_feature.xaxis.set_minor_locator(ticker.MaxNLocator(integer=True))
            self.ax_feature.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
            if not self.__feature_is_datetime:
                # decorate depending on transform
                if self.__feature_transform is None:
                    _, ax_max = self.ax_feature.get_xlim()
                    if ax_max > 1000:
                        self.ax_feature.xaxis.set_major_formatter(
                            ticker.StrMethodFormatter("{x:,.0f}")
                        )
                elif self.__feature_transform == "square":
                    self.ax_feature.xaxis.set_major_formatter(
                        ticker.StrMethodFormatter("$\sqrt{{{x:.0f}}}$")
                    )
                elif self.__feature_transform == "square-root":
                    self.ax_feature.xaxis.set_major_formatter(
                        ticker.StrMethodFormatter("${{{x:.0f}}}^2$")
                    )
                elif self.__feature_transform == "log-2":
                    self.ax_feature.xaxis.set_major_formatter(
                        ticker.StrMethodFormatter("$2^{{{x:.0f}}}$")
                    )
                elif self.__feature_transform == "exp-2":
                    self.ax_feature.xaxis.set_major_formatter(
                        ticker.StrMethodFormatter("$\log_2{{{x:.0f}}}$")
                    )
                elif self.__feature_transform == "yeojohnson":
                    self.ax_feature.xaxis.set_ticklabels([])
        elif self.__feature_is_datetime:
            self.ax_feature.xaxis.set_minor_locator(dates.AutoDateLocator())
            self.ax_feature.xaxis.set_major_locator(dates.AutoDateLocator(maxticks=5))
        else:
            if self.__feature_nunique > self.__binning_threshold:
                self.ax_feature.set_xticklabels([])
            else:
                self.ax_feature.tick_params(axis="x", labelrotation=90)

        # decorate first y axis
        if self.__metric == "count":
            self.ax_feature.set_ylabel("Frequency")
            self.ax_feature.yaxis.set_major_formatter(
                ticker.StrMethodFormatter("{x:,.0f}")
            )
        elif self.__metric == "prop":
            self.ax_feature.set_ylabel("Frequency density")
            self.ax_feature.yaxis.set_major_formatter(
                ticker.StrMethodFormatter("{x:,.2f}")
            )

        # decorate second y axis
        if self.__has_target:
            twin_y_colour = (
                "k"
                if len(y_plot_all) > 1 and not self.__target_is_bool
                else colour_target
            )
            self.ax_target.set_ylabel(self.target.name, color=twin_y_colour)
            self.ax_target.tick_params(axis="y", labelcolor=twin_y_colour)
            if self.__target_type == "classification":
                self.ax_target.yaxis.set_major_formatter(ticker.PercentFormatter())
                if not self.__target_is_bool:
                    self.ax_target.legend()

        # add title
        if self.__has_target:
            score_type, score = self.__target_score_type, self.__target_score
        else:
            score_type, score = self.__feature_score_type, self.__feature_score
        self.ax_feature.set_title(
            f"{score_type} = {score:.2f}\n({100*self.__missing_proportion:.1f}% missing)"
        )

    def calculate_feature_score(self) -> None:
        """Calculate the score for the feature based on its skewness"""
        self.__feature_score: pd.DataFrame
        self.__feature_score_type: Union[None, str]
        if (
            self.__feature_is_numeric or self.__feature_is_datetime
        ) and not self.__feature_is_bool:
            # calculate skew of median towards deciles
            feature = (
                self.feature
                if not self.__feature_is_datetime
                else (self.feature - self.feature.min()).dt.total_seconds()
            )
            self.__feature_score, self.__feature_score_type = utils.inter_decile_skew(
                feature
            )
        else:
            # calculate skew towards the mode
            self.__feature_score = self.feature.value_counts(normalize=True).max()
            self.__feature_score_type = "Categorical skew"

    def calculate_target_score(self) -> None:
        """Calculate the score for the feature based on its predictive power"""
        if self.__has_target:
            # rebalance classes
            if self.__target_type == "classification" and self.__target_rebalance:
                x_balanced, y_balanced = utils.class_rebalance(
                    self.feature, self.target
                )
                df = pd.concat([x_balanced, y_balanced], axis=1)
            else:
                df = pd.concat([self.feature, self.target], axis=1)

            ## calculate score
            self.__target_score = pps.calculate(
                df=df,
                x=self.feature.name,
                y=self.target.name,
            ).iloc[0]["ppscore"]
            self.__target_score_type = "PPS"
        else:
            self.__target_score = np.nan
            self.__target_score_type = "N/A"

    def summarise_feature(self) -> None:
        """Summarise the feature by calculating summary statistics for each distinct value and binning if there are too many distinct values"""
        # join feature and target intro single dataframe
        if self.__has_target:
            all_data = pd.concat([self.feature, self.target], axis=1)
        else:
            all_data = self.feature.to_frame()

        # bin target variable if there are too many distinct values
        if self.__feature_nunique > self.__binning_threshold and (
            self.__feature_is_numeric or self.__feature_is_datetime
        ):
            feature = (
                self.feature
                if not self.__feature_is_datetime
                else (self.feature - self.feature.min()).dt.total_seconds()
            )
            bin_boundaries = np.linspace(
                feature.min(), feature.max(), self.__binning_threshold + 1
            )
            all_data[self.feature.name] = pd.cut(feature, bin_boundaries).apply(
                lambda x: x.mid
            )
            if self.__feature_is_datetime:
                all_data[self.feature.name] = (
                    pd.to_timedelta(all_data[self.feature.name], unit="s")
                    + self.feature.min()
                )

        # calculate summary statistics for each distinct target variable
        if self.__has_target:
            if self.__target_type == "regression":
                self.__feature_summary: pd.DataFrame = all_data.groupby(
                    self.feature.name
                ).agg({"count", "mean", "std"})
                self.__feature_summary.columns = (
                    self.__feature_summary.columns.droplevel()
                )
            elif self.__target_type == "classification":
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
        self.__feature_summary["prop"] = (
            self.__feature_summary["count"] / self.__feature_summary["count"].sum()
        )

        # convert index to string from boolean for printing purposes
        if self.__feature_is_bool:
            self.__feature_summary.index = self.__feature_summary.index.map(
                {True: "True", False: "False"}
            )

    def to_dict(self) -> dict:
        "Summarise as a dictionary"
        summary = {
            "feature_name": self.feature.name,
            "feature_dtype": self.__feature_dtype,
            "feature_score": self.__feature_score,
            "feature_score_type": self.__feature_score_type,
            "feature_transform": self.__feature_transform,
            "feature_nunique": self.__feature_nunique,
            "feature_missing_proportion": self.__missing_proportion,
            "target_name": self.target.name if self.__has_target else None,
            "target_dtype": self.__target_dtype if self.__has_target else None,
            "target_score": self.__target_score,
            "target_score_type": self.__target_score_type,
        }
        return summary

    # feature getter
    @property
    def feature(self) -> pd.Series:
        """The feature data"""
        return self.__feature

    # feature setter
    @feature.setter
    def feature(self, feature: Any) -> None:
        if hasattr(self, "feature"):
            # do not allow changing of data
            raise AttributeError("This attribute has already been set")

        else:
            # convert to series and set
            data = utils.to_series(feature)

            is_bool, is_numeric, _, _ = utils.classify_type(data)

            # reduce feature skew
            if self.__feature_deskew and (is_numeric and not is_bool):
                self.__feature_transform, self.__feature = utils.reduce_skew(
                    data,
                    transforms=(
                        None
                        if isinstance(self.__feature_deskew, bool)
                        else self.__feature_deskew
                    ),
                )
            else:
                self.__feature_transform, self.__feature = None, data

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
