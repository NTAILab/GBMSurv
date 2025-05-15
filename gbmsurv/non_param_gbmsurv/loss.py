import numpy as np
import xgboost as xgb
from typing import Tuple
from scipy.special import softmax


def surv_grad_hess(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the first- and second-order derivatives (gradient and Hessian) of the 
    survival loss with respect to the predicted event probabilities in each time interval.
    """

    y = dtrain.get_label()
    y = y.reshape(preds.shape)

    correct_class = y[:, 0].astype(int)
    censoring = y[:, 1].astype(int)

    s = softmax(preds, axis=1)

    N, C = s.shape

    preds = preds.astype(np.float128)
    preds = np.maximum(preds, 10e-20)

    # gradient

    grad_1 = np.zeros((N, C))

    delta_ik = np.eye(C)[correct_class.reshape(-1)]
    grad_1 = - censoring[:, None] * (delta_ik - s)

    grad_0 = np.zeros((N, C))

    indices = np.arange(C).reshape(1, -1)
    indices_last_class = np.full((1, C), C-1)

    mask_i_is_not_last_class = indices_last_class != correct_class.reshape(
        -1, 1)
    mask_less_equal_i = (
        indices <= correct_class.reshape(-1, 1)) == mask_i_is_not_last_class
    mask_more_i = ~ mask_less_equal_i == mask_i_is_not_last_class

    exp_preds = np.exp(preds)

    exp_preds_not_after_i = exp_preds * mask_less_equal_i
    exp_preds_after_i = exp_preds * mask_more_i

    sum_exp_l = np.sum(exp_preds_not_after_i, axis=1)
    sum_exp_r = np.sum(exp_preds_after_i, axis=1)

    sum_exp_r_nonzero = np.where(sum_exp_r == 0, 1e-10, sum_exp_r)

    grad_0[mask_less_equal_i] = -s[mask_less_equal_i]
    grad_0[mask_more_i] = (
        s * (sum_exp_l/sum_exp_r_nonzero)[:, None])[mask_more_i]

    grad_0 = - (1 - censoring[:, None]) * grad_0

    grad = grad_1 + grad_0

    # hessian

    hess_1 = censoring[:, None] * s * (1.0 - s)

    hess_0 = np.zeros((N, C))

    hess_0[mask_less_equal_i] = - \
        s[mask_less_equal_i] * (1 - s[mask_less_equal_i])
    hess_0[mask_more_i] = (s * (1 - s) * (sum_exp_l/sum_exp_r_nonzero)[:, None])[mask_more_i] + (
        s * (-1) * exp_preds * (sum_exp_l / (sum_exp_r_nonzero ** 2))[:, None])[mask_more_i]

    hess_0 = - (1 - censoring[:, None]) * hess_0

    hess = hess_1 + hess_0

    return grad.ravel(), hess.ravel()


def surv_loss(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """
    Computes the average negative log-likelihood loss for a non-parametric 
    survival model on a right-censored dataset.
    """

    y = dtrain.get_label()
    y = y.reshape(preds.shape)

    correct_class = y[:, 0].astype(int)
    censoring = y[:, 1].astype(int)

    s = softmax(preds, axis=1)

    N, C = s.shape

    loss = 0

    indices = np.arange(N)
    mask_s_correct_class = np.zeros_like(s, dtype=bool)
    mask_s_correct_class[indices, correct_class] = True
    mask_after_correct_class = np.tile(
        np.arange(C), (N, 1)) > correct_class[:, None]

    loss -= np.sum(censoring * np.log(s[mask_s_correct_class]))

    sum_s_after_correct_class = np.sum(s * mask_after_correct_class, axis=1)
    sum_s_after_correct_class_nonzero = np.where(
        sum_s_after_correct_class == 0, 1e-0, sum_s_after_correct_class)

    loss -= np.sum((1 - censoring) * np.log(sum_s_after_correct_class_nonzero))

    loss = loss / N

    return 'SurvivalLoss', loss
