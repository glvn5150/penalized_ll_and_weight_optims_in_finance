import numpy as np
from math import log, sqrt, pi
from numpy.linalg import norm

""""""" the gradient descent and training algorithm """""""

def calculate_generalized_nll(observed_data, predicted_mean, predicted_var):
    T = len(observed_data)
    predicted_var = np.maximum(predicted_var, 1e-9)
    term1 = (T / 2.0) * log(2 * pi)
    term2 = 0.5 * np.sum(np.log(predicted_var))
    term3 = np.sum((observed_data - predicted_mean)**2 / (2 * predicted_var))
    return term1 + term2 + term3

def simplex_proj(y):
    n_features = len(y)
    u = np.sort(y)[::-1]
    css = np.cumsum(u)
    ind = np.arange(n_features) + 1
    cond = u - (css - 1.0) / ind > 0
    rho = ind[cond][-1]
    theta = (css[cond][-1] - 1.0) / rho
    return np.maximum(y - theta, 0)

def proj(x, eta):
    w_simplex = simplex_proj(x)
    if np.count_nonzero(w_simplex) > eta:
        threshold = np.sort(w_simplex)[-eta]
        w_simplex[w_simplex < threshold] = 0
        w_simplex /= (np.sum(w_simplex) + 1e-12)
    return w_simplex

def gradient_f3(w, gamma, T, S_train):
    B = np.identity(T) - np.ones([T,T])/T
    B0 = np.matmul(B, S_train[:T])
    B1 = np.matmul(B, S_train[1:])
    B0_t, B1_t = B0.T, B1.T
    b0, b1 = np.matmul(B0, w), np.matmul(B1, w)
    b01 = np.inner(b0, b1)
    b0_Frob = max(np.inner(b0, b0), 1e-9)
    if gamma == 0:
        c = b01 / b0_Frob
        bcb = b1 - c * b0
        grad = (np.matmul(B1_t, bcb) - c * np.matmul(B0_t, bcb)) / max(np.inner(bcb, bcb), 1e-9)
    else:
        b1_Frob = np.inner(b1, b1)
        p = sqrt(max(b0_Frob**2 - 4 * (gamma**2) * (b0_Frob * b1_Frob - b01**2), 1e-9))
        q = b0_Frob - p
        grad = (1/max(q, 1e-9)) * np.matmul(B0_t, b0)
    return grad

def run_projected_gradient_descent(S_train, w_init, params):
    w = w_init.copy()
    T = S_train.shape[0] - 1
    for i in range(params['max_iter']):
        grad = gradient_f3(w, params['gamma'], T, S_train)
        step = params['stepsize'] * grad
        w_new = proj(w - step, params['eta'])
        if norm(w_new - w) < 1e-6:
            break
        w = w_new
    return w

def portfolio_nll(w, S, delta_t, engine_func, params):
    x = np.matmul(S, w)
    x_current, x_next_actual = x[:-1], x[1:]
    pred_mean, pred_var = engine_func(x_current, delta_t, params)
    return calculate_generalized_nll(x_next_actual, pred_mean, pred_var)