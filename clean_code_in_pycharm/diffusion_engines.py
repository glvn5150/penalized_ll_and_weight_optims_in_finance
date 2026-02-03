import numpy as np
""""""" primary diffusion moments for model """""""

def engine_ou(x_current, delta_t, params):
    mu, theta, sigma_sq = params['mu'], params['theta'], params['sigma_sq']
    phi = np.exp(-mu * delta_t)
    m_next = x_current * phi + theta * (1 - phi)
    v_next = (sigma_sq / (2 * mu)) * (1 - phi**2)
    return m_next, v_next

def engine_vasicek(x_t, dt, params):
    a, b, sigma = params['a'], params['b'], params['sigma']
    phi = np.exp(-a * dt)
    m_next = x_t * phi + b * (1 - phi)
    v_next = (sigma**2 / (2 * a)) * (1 - phi**2)
    return m_next, v_next

def engine_cir(x_t, dt, params):
    a, b, sigma = params['a'], params['b'], params['sigma']
    phi = np.exp(-a * dt)
    m_next = x_t * phi + b * (1 - phi)
    v_next = x_t * (sigma**2 / a) * (phi - phi**2) + (b * sigma**2 / (2 * a)) * (1 - phi)**2
    return m_next, v_next

def engine_hw(x_t, dt, params):
    a, theta, sigma = params['a'], params['theta'], params['sigma']
    phi = np.exp(-a * dt)
    m_next = x_t * phi + (theta / a) * (1 - phi)
    v_next = (sigma**2 / (2 * a)) * (1 - phi**2)
    return m_next, v_next

def engine_bdt(x_t, dt, params):
    theta, sigma = params['theta'], params['sigma']
    m_ln = np.log(x_t) + theta * dt
    v_ln = sigma**2 * dt
    m_next = np.exp(m_ln + 0.5 * v_ln)
    v_next = (np.exp(v_ln) - 1) * np.exp(2 * m_ln + v_ln)
    return m_next, v_next

def engine_bk(x_t, dt, params):
    a, theta, sigma = params['a'], params['theta'], params['sigma']
    phi = np.exp(-a * dt)
    m_ln = np.log(x_t) * phi + (theta / a) * (1 - phi)
    v_ln = (sigma**2 / (2 * a)) * (1 - phi**2)
    m_next = np.exp(m_ln + 0.5 * v_ln)
    v_next = (np.exp(v_ln) - 1) * np.exp(2 * m_ln + v_ln)
    return m_next, v_next

def engine_ho_lee(x_current, delta_t, params):
    drift, sigma_sq = params['drift'], params['sigma_sq']
    m_next = x_current + drift * delta_t
    v_next = sigma_sq * delta_t #linear_growth
    return m_next, v_next