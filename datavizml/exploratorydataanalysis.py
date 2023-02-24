from matplotlib import pyplot as plt
from datavizml import singledistribution as sd, utils


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
        self.data_dtypes = set(
            [utils.classify_type(x)[2] for _, x in self.data.items()]
        )
        if self.has_target:
            _, _, self.target_dtype = utils.classify_type(self.target)

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
                sd.SingleDistribution(
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
            self.__data = utils.to_frame(data)

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
            self.__target = utils.to_series(target, name=name)
