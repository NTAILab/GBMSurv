import numpy as np
import xgboost as xgb
from typing import Tuple
from .utils import safe_log
from scipy.stats import norm

def prepare_data_for_computing(preds: np.ndarray, dtrain: xgb.DMatrix):

    """
    Prepares intermediate variables required for computing the gradient, Hessian, and loss 
    in a parametric survival model based on the Log-Normal distribution.
    """

    y = dtrain.get_label()
    y = y.reshape(preds.shape)

    time = y[:, 0].astype(np.float64)
    censoring = y[:, 1].astype(np.float64)
    
    N, C = y.shape

    preds = preds.astype(np.float64)
    preds = np.maximum(preds, 10e-20) 

    time_intervals = np.sort(np.unique(time))
    time_indices = np.searchsorted(time_intervals, time, side='right') - 1
    mask = (time_indices == np.max(time_indices))

    k = 0 # increase interval to lenght of 2 * k + 1

    valid_left_indices = np.maximum(time_indices[~mask] - k, 0)
    valid_right_indices = np.minimum(time_indices[~mask] + 1 + k, len(time_intervals) - 1)

    time_left = time_intervals[valid_left_indices]
    time_right = time_intervals[valid_right_indices]

    mu = preds[:, 0]
    
    sigma = preds[:, 1]
    sigma = np.maximum(sigma, 10e-3)

    log_time_right = np.log(time_right)
    log_time_left = np.log(time_left)

    log_t_mu_diff_right = log_time_right - mu[~mask]
    log_t_mu_diff_left = log_time_left - mu[~mask]

    phi_values_right = norm.pdf(log_time_right, loc=mu[~mask], scale=sigma[~mask])
    phi_values_left = norm.pdf(log_time_left, loc=mu[~mask], scale=sigma[~mask])

    laplas_func_values_right = norm.cdf(log_time_right, loc=mu[~mask], scale=sigma[~mask])
    laplas_func_values_left = norm.cdf(log_time_left, loc=mu[~mask], scale=sigma[~mask])

    valid_left_masked_indices = np.maximum(time_indices[mask] - k, 0)
    time_left_masked = time_intervals[valid_left_masked_indices]
    log_time_left_masked = np.log(time_left_masked)
    log_t_mu_diff_left_masked = log_time_left_masked - mu[mask]
    phi_values_left_masked = norm.pdf(log_time_left_masked, loc=mu[mask], scale=sigma[mask])
    laplas_func_values_left_masked = norm.cdf(log_time_left_masked, loc=mu[mask], scale=sigma[mask])

    return y.shape, censoring, mask, sigma, log_t_mu_diff_right, phi_values_right, laplas_func_values_right, log_t_mu_diff_left, phi_values_left, laplas_func_values_left, log_t_mu_diff_left_masked, phi_values_left_masked, laplas_func_values_left_masked


def surv_grad_hess(preds: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Computes the first- and second-order derivatives (gradient and Hessian) of the 
    survival loss with respect to the predicted parameters of the Log-Normal distribution.
    """

    shape, censoring, mask, sigma, log_t_mu_diff_right, phi_values_right, laplas_func_values_right, log_t_mu_diff_left, phi_values_left, laplas_func_values_left, log_t_mu_diff_left_masked, phi_values_left_masked, laplas_func_values_left_masked = prepare_data_for_computing(preds, dtrain)

    # gradient

    grad = np.zeros(shape) 

    laplas_func_diff = np.maximum(laplas_func_values_right - laplas_func_values_left, 10e-20)

    laplas_func_one_diff_right = np.maximum(1 - laplas_func_values_right, 10e-20)
    laplas_func_one_diff_left_masked = np.maximum(1 - laplas_func_values_left_masked, 10e-20)

    # Calculate gradients for all but the last interval
    grad[~mask, 0] += censoring[~mask] * (phi_values_right - phi_values_left) / (laplas_func_diff)
    grad[~mask, 1] += censoring[~mask] * ((log_t_mu_diff_right / sigma[~mask]) * phi_values_right - (log_t_mu_diff_left / sigma[~mask]) * phi_values_left) / (laplas_func_diff)

    grad[~mask, 0] -= (1 - censoring[~mask]) * phi_values_right / laplas_func_one_diff_right
    grad[~mask, 1] -= (1 - censoring[~mask]) * (log_t_mu_diff_right / sigma[~mask]) * phi_values_right / laplas_func_one_diff_right

    # Handle the last interval separately
    grad[mask, 0] -= censoring[mask] * phi_values_left_masked / laplas_func_one_diff_left_masked
    grad[mask, 1] -= censoring[mask] * (log_t_mu_diff_left_masked / sigma[mask]) * phi_values_left_masked / laplas_func_one_diff_left_masked 
    
    # hessian

    hess = np.zeros(shape) 
    
    # Calculate hessians for all but the last interval
    hess[~mask, 0] += censoring[~mask] * (((phi_values_right - phi_values_left) / (laplas_func_diff)) ** 2 \
                                          + ((log_t_mu_diff_right / sigma[~mask] ** 2) * phi_values_right - (log_t_mu_diff_left / sigma[~mask] ** 2) * phi_values_left) / laplas_func_diff)
    
    u = 1 / (laplas_func_diff)
    v = ((log_t_mu_diff_right / sigma[~mask]) * phi_values_right - (log_t_mu_diff_left / sigma[~mask]) * phi_values_left)

    u_derivative = v / (u ** 2)
    v_derivative = (log_t_mu_diff_right / (sigma[~mask] ** 2)) * phi_values_right * ((log_t_mu_diff_right / sigma[~mask]) ** 2 - 2) - (log_t_mu_diff_left / (sigma[~mask] ** 2)) * phi_values_left * ((log_t_mu_diff_left / sigma[~mask]) ** 2 - 2)

    hess[~mask, 1] += censoring[~mask] * (u_derivative * v + u * v_derivative)

    hess[~mask, 0] -= (1 - censoring[~mask]) * ((log_t_mu_diff_right / (sigma[~mask] ** 2)) * phi_values_right / laplas_func_one_diff_right \
                                                - (phi_values_right / laplas_func_one_diff_right) ** 2)
    hess[~mask, 1] -= (1 - censoring[~mask]) * (log_t_mu_diff_right ** 3 / sigma[~mask] ** 4 * phi_values_right / laplas_func_one_diff_right \
                                                - 2 * log_t_mu_diff_right / sigma[~mask] ** 2 * phi_values_right / laplas_func_one_diff_right \
                                                - (log_t_mu_diff_right / sigma[~mask]) ** 2 * (phi_values_right / laplas_func_one_diff_right) ** 2)

    # Handle the last interval separately
    hess[mask, 0] -= censoring[mask] * ((log_t_mu_diff_left_masked / (sigma[mask] ** 2)) * phi_values_left_masked / laplas_func_one_diff_left_masked \
                                        - (phi_values_left_masked / laplas_func_one_diff_left_masked) ** 2)
    hess[mask, 1] -= censoring[mask] * (log_t_mu_diff_left_masked ** 3 / sigma[mask] ** 4 * phi_values_left_masked / laplas_func_one_diff_left_masked \
                                        - 2 * log_t_mu_diff_left_masked / sigma[mask] ** 2 * phi_values_left_masked / laplas_func_one_diff_left_masked \
                                        - (log_t_mu_diff_left_masked / sigma[mask]) ** 2 * (phi_values_left_masked / laplas_func_one_diff_left_masked) ** 2)

    return grad.ravel(), hess.ravel()

def surv_loss(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:

    """
    Computes the average negative log-likelihood loss for a parametric Log-Normal 
    survival model on a right-censored dataset.
    """

    shape, censoring, mask, _, _, _, laplas_func_values_right, _, _, laplas_func_values_left, _, _, laplas_func_values_left_masked = prepare_data_for_computing(preds, dtrain)

    N, _ = shape

    loss = 0

    loss -= np.sum(censoring[~mask] * safe_log(laplas_func_values_right - laplas_func_values_left))
    loss -= np.sum((1 - censoring[~mask]) * safe_log(1 - laplas_func_values_right))
    loss -= np.sum(censoring[mask] * safe_log(1 - laplas_func_values_left_masked))

    loss = loss / N

    return 'SurvivalLoss', loss