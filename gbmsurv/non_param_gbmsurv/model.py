import numpy as np
import xgboost as xgb
from scipy.special import softmax

from .loss import surv_grad_hess, surv_loss
from sksurv.metrics import concordance_index_censored
from scipy.integrate import trapezoid


class GBMSurvivalModel:

    """
    Gradient Boosting Model for Non-Parametric Survival Analysis.

    This model uses gradient boosting (based on XGBoost) to predict event probabilities within discrete time intervals. 
    It is designed to work with right-censored survival data and supports custom regularization.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Learning rate for boosting.

    n_estimators : int, default=100
        Number of boosting rounds (trees).

    max_depth : int, default=7
        Maximum tree depth for base learners.

    random_seed : int, default=42
        Random seed for reproducibility.

    lambda_val : float, default=1
        L2 regularization term on weights.

    alpha : float, default=0
        L1 regularization term on weights.

    num_intervals : int, default=10
        Number of discrete time intervals into which the timeline is divided. 
        The model predicts the probability of an event occurring within each of these intervals.

    interval_grid : str, default='time_distribution_based'
        Strategy for dividing the time axis into intervals. Determines how the time intervals 
        are constructed for predicting event probabilities. Available options are:

        - 'unique_times': Each unique time point in the dataset is used as the left boundary 
        of an interval. The number of intervals is equal to the number of unique times.
        - 'uniform': The time range is divided into `num_intervals` equally spaced intervals.
        - 'time_distribution_based': The time range is divided based on percentiles of the 
        time distribution, resulting in intervals that reflect the data's empirical distribution.

    colsample_bytree : float, default=1
        Subsample ratio of columns when constructing each tree.

    colsample_bylevel : float, default=1
        Subsample ratio of columns for each level.

    colsample_bynode : float, default=1
        Subsample ratio of columns for each split.

    max_leaves : int, default=0
        Maximum number of leaves for tree growth (used when grow_policy="lossguide").

    max_bin : int, default=256
        Number of histogram bins for histogram-based split finding.

    min_child_weight : float, default=1
        Minimum sum of instance weight (hessian) needed in a child.

    subsample : float, default=1
        Subsample ratio of the training instances.

    initial_params : bool, default=False
        If True, initializes params using the mean and var of event time.

    Attributes
    ----------
    _model : xgb.Booster
        The trained XGBoost model.

    _results : dict
        Training history and logged metrics.

    _interval_bounds : np.ndarray
        Array of time interval boundaries used to split the time axis. Each value represents 
        the left boundary of a time interval. The intervals are defined based on the selected 
        `interval_grid` strategy and are used to compute event probabilities per interval.

    """

    def __init__(
            self,
            learning_rate=0.1,
            n_estimators=100,
            max_depth=7,
            random_seed=42,
            lambda_val=1,
            alpha=0,
            num_intervals=10,
            interval_grid='time_distribution_based',
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
            max_leaves=0,
            max_bin=256,
            min_child_weight=1,
            subsample=1,
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.lambda_val = lambda_val
        self.alpha = alpha
        self.num_intervals = num_intervals
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.interval_grid = interval_grid

    def prepare_target(self, y):
        """Function for preparing data before training model
        y should have format (delta, time)."""

        self._interval_bounds = np.array([])

        delta_str = y.dtype.names[0]
        time_str = y.dtype.names[1]

        delta = y[delta_str]
        time = y[time_str]

        if self.interval_grid == 'uniform':
            self._interval_bounds = np.linspace(
                time.min(), time.max(), self.num_intervals)

        elif self.interval_grid == 'time_distribution_based':
            percentiles = np.linspace(0, 100, self.num_intervals)
            self._interval_bounds = np.percentile(time, percentiles)

        elif self.interval_grid == 'unique_times':
            self._interval_bounds = np.unique(time)
            self.num_intervals = len(self._interval_bounds)

        time_ind = np.digitize(time, self._interval_bounds) - 1

        target = np.zeros((self.num_intervals, len(time_ind)))
        target[0] = time_ind
        target[1] = delta
        target = target.transpose()

        return target

    def fit(self, X, y):

        self._model = xgb.Booster()
        self._results = dict()

        target = self.prepare_target(y)

        d_train = xgb.DMatrix(X, label=target, enable_categorical=True)

        self._model = xgb.train(
            {
                'tree_method': 'hist',
                'seed': self.random_seed,
                'disable_default_eval_metric': 1,
                'multi_strategy': "multi_output_tree",
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'lambda': self.lambda_val,
                'alpha': self.alpha,
                'nthread': 8,
                'colsample_bytree': self.colsample_bytree,
                'colsample_bylevel': self.colsample_bylevel,
                'colsample_bynode': self.colsample_bynode,
                'min_child_weight': self.min_child_weight,
                'max_leaves': self.max_leaves,
                'max_bin': self.max_bin,
                'subsample': self.subsample,
            },
            dtrain=d_train,
            num_boost_round=self.n_estimators,
            obj=surv_grad_hess,
            custom_metric=surv_loss,
            # evals=[(d_train, 'd_train')],
            evals_result=self._results,
        )

        return self

    def predict(self, X):

        d_test = xgb.DMatrix(X, enable_categorical=True)
        predicted_proba = softmax(self._model.predict(d_test), axis=1)

        return predicted_proba

    def _step_function(self, times, survival_function):

        if isinstance(times, (int, float)):
            times = [times]

        survs = []

        for time in times:
            if time < 0:
                raise ValueError("Time can't have negative value")

            if time < self._interval_bounds[0]:
                survs.append(1)
            elif time >= self._interval_bounds[-1]:
                survs.append(survival_function[-1])
            else:
                for i, bound in enumerate(self._interval_bounds):
                    if time < bound:
                        survs.append(survival_function[i - 1])
                        break

        return survs

    def predict_survival_function(self, X):

        d_test = xgb.DMatrix(X, enable_categorical=True)
        predicted_proba = softmax(self._model.predict(d_test), axis=1)

        assert np.allclose(np.sum(predicted_proba, axis=1), 1.0)

        cumulative_proba = np.cumsum(predicted_proba, axis=1)

        cumulative_proba[cumulative_proba > 1.0] = 1.0

        survival_functions = 1 - cumulative_proba

        step_functions = np.array([
            lambda x, sf=sf: self._step_function(x, sf) for sf in survival_functions
        ])

        return step_functions

    def score(self, X, y):

        delta_str = y.dtype.names[0]
        time_str = y.dtype.names[1]

        delta = y[delta_str]
        time = y[time_str]

        d_test = xgb.DMatrix(X, enable_categorical=True)
        predicted_proba = softmax(self._model.predict(d_test), axis=1)

        cumulative_proba = np.cumsum(predicted_proba, axis=1)

        cumulative_proba[cumulative_proba > 1.0] = 1.0

        survival_function = 1 - cumulative_proba

        integrated_сum_proba = np.array([trapezoid(
            survival_function[i], self._interval_bounds) for i in range(survival_function.shape[0])])

        c_index = concordance_index_censored(
            delta.astype(bool), time, -integrated_сum_proba)

        return c_index[0]

    def get_params(self, deep=True):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
