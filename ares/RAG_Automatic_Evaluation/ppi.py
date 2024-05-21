import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import binom, norm
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from joblib import delayed, Parallel
import pdb

"""
    IID Concentration Bounds
"""
def binomial_iid(N: int, alpha: float, muhat: float) -> np.ndarray:
    """
    Calculate the binomial confidence interval for independent and identically distributed (IID) samples.

    Parameters:
    N (int): The number of trials.
    alpha (float): The significance level.
    muhat (float): The estimated mean.

    Returns:
    np.ndarray: A numpy array containing the lower and upper bounds of the confidence interval.
    """
    
    def invert_upper_tail(mu: float) -> float:
        """Calculate the upper tail inversion."""
        return binom.cdf(N * muhat, N, mu) - (alpha / 2)
    
    def invert_lower_tail(mu: float) -> float:
        """Calculate the lower tail inversion."""
        return binom.cdf(N * muhat, N, mu) - (1 - alpha / 2)
    
    u = brentq(invert_upper_tail, 0, 1)
    l = brentq(invert_lower_tail, 0, 1)
    
    return np.array([l, u])

def bentkus_iid(N: int, alpha: float, muhat: float) -> np.ndarray:
    """
    Calculate the Bentkus confidence interval for independent and identically distributed (IID) samples.

    Parameters:
    N (int): The number of trials.
    alpha (float): The significance level.
    muhat (float): The estimated mean.

    Returns:
    np.ndarray: A numpy array containing the lower and upper bounds of the confidence interval.
    """
    return binomial_iid(N, alpha / np.e, muhat)

def clt_iid(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate the confidence interval for the mean of IID samples using the Central Limit Theorem (CLT).

    Parameters:
    x (np.ndarray): The array of sample data.
    alpha (float): The significance level.

    Returns:
    np.ndarray: A numpy array containing the lower and upper bounds of the confidence interval.
    """
    n = x.shape[0]  # Number of samples
    sigmahat = x.std()  # Sample standard deviation
    w = norm.ppf(1 - alpha / 2) * sigmahat / np.sqrt(n)  # Margin of error
    return np.array([x.mean() - w, x.mean() + w])  # Confidence interval

def wsr_iid(x_n: np.ndarray, alpha: float, grid: np.ndarray, num_cpus: int = 10, parallelize: bool = False, 
            intersection: bool = True, theta: float = 0.5, c: float = 0.75) -> np.ndarray:
    """
    Calculate the Weighted Sequential Rank (WSR) confidence interval for IID samples.

    Parameters:
    x_n (np.ndarray): The array of sample data.
    alpha (float): The significance level.
    grid (np.ndarray): The grid of values to search for the confidence interval.
    num_cpus (int, optional): The number of CPUs to use for parallel processing. Default is 10.
    parallelize (bool, optional): Whether to parallelize the computation. Default is False.
    intersection (bool, optional): Whether to use the intersection method for the confidence interval. Default is True.
    theta (float, optional): The weighting parameter. Default is 0.5.
    c (float, optional): The scaling parameter. Default is 0.75.

    Returns:
    np.ndarray: A numpy array containing the lower and upper bounds of the confidence interval.
    """
    n = x_n.shape[0]
    t_n = np.arange(1, n + 1)
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(x_n - muhat_n, 2))) / (1 + t_n)
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[:-1])
    assert np.all(sigma2hat_tminus1_n > 0), "All elements of sigma2hat_tminus1_n must be greater than 0"
    lambda_n = np.sqrt(2 * np.log(2 / alpha) / (n * sigma2hat_tminus1_n))

    def M(m: float) -> np.ndarray:
        lambdaplus_n = np.minimum(lambda_n, c / m)
        lambdaminus_n = np.minimum(lambda_n, c / (1 - m))
        return np.maximum(
            theta * np.exp(np.cumsum(np.log(1 + lambdaplus_n * (x_n - m)))),
            (1 - theta) * np.exp(np.cumsum(np.log(1 - lambdaminus_n * (x_n - m))))
        )

    if parallelize:
        M_vectorized = np.vectorize(M)
        M_list = Parallel(n_jobs=num_cpus)(delayed(M_vectorized)(m) for m in grid)
        indicators_gxn = np.array(M_list) < 1 / alpha
    else:
        indicators_gxn = np.zeros([grid.size, n])
        found_lb = False
        for m_idx, m in enumerate(grid):
            m_n = M(m)
            indicators_gxn[m_idx] = m_n < 1 / alpha
            if not found_lb and np.prod(indicators_gxn[m_idx]):
                found_lb = True
            if found_lb and not np.prod(indicators_gxn[m_idx]):
                break  # Since interval, once find a value that fails, stop searching

    if intersection:
        ci_full = grid[np.where(np.prod(indicators_gxn, axis=1))[0]]
    else:
        ci_full = grid[np.where(indicators_gxn[:, -1])[0]]

    if ci_full.size == 0:  # Grid may be too coarse
        idx = np.argmax(np.sum(indicators_gxn, axis=1))
        if idx == 0:
            return np.array([grid[0], grid[1]])
        return np.array([grid[idx - 1], grid[idx]])

    return np.array([ci_full.min(), ci_full.max()])  # Only output the interval

"""
    Mean estimation confidence intervals
"""

def pp_mean_iid_asymptotic(Y_labeled: np.ndarray, Yhat_labeled: np.ndarray, Yhat_unlabeled: np.ndarray, alpha: float) -> list:
    """
    Compute the mean estimation confidence interval for IID data using an asymptotic approach.

    Parameters:
    Y_labeled (np.ndarray): Labeled data array.
    Yhat_labeled (np.ndarray): Predicted values for the labeled data.
    Yhat_unlabeled (np.ndarray): Predicted values for the unlabeled data.
    alpha (float): Significance level for the confidence interval.

    Returns:
    list: A list containing the lower and upper bounds of the confidence interval.
    """
    n = Y_labeled.shape[0]  # Number of labeled samples
    N = Yhat_unlabeled.shape[0]  # Number of unlabeled samples
    
    if n == 0 or N == 0:
        return [0, 0]

    # Mean of the predicted values for the unlabeled data
    tildethetaf = Yhat_unlabeled.mean()

    # Mean of the residuals (difference between predicted and actual values for labeled data)
    rechat = (Yhat_labeled - Y_labeled).mean()
    
    # Point estimate for the parameter of interest
    thetahatPP = tildethetaf - rechat

    # Standard deviation of the predicted values for the unlabeled data
    sigmaftilde = np.std(Yhat_unlabeled)

    # Standard deviation of the residuals
    sigmarec = np.std(Yhat_labeled - Y_labeled)

    # Half-width of the confidence interval
    hw = norm.ppf(1 - alpha / 2) * np.sqrt((sigmaftilde**2 / N) + (sigmarec**2 / n))

    # Return the confidence interval as a list
    return [thetahatPP - hw, thetahatPP + hw]

"""
    OLS algorithm with sandwich variance estimator
"""

def ols(features: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """
    Perform Ordinary Least Squares (OLS) regression.

    Parameters:
    features (np.ndarray): The input feature matrix (design matrix).
    outcome (np.ndarray): The outcome vector (response variable).

    Returns:
    np.ndarray: The OLS coefficients.
    """
    ols_coeffs = np.linalg.pinv(features).dot(outcome)
    return ols_coeffs

def classical_ols_interval(X: np.ndarray, Y: np.ndarray, alpha: float, 
                           return_stderr: bool = False, sandwich: bool = True):
    """
    Compute the confidence interval for OLS regression coefficients using the sandwich variance estimator.

    Parameters:
    X (np.ndarray): The input feature matrix (design matrix).
    Y (np.ndarray): The outcome vector (response variable).
    alpha (float): Significance level for the confidence interval.
    return_stderr (bool): If True, return the standard errors instead of the confidence interval. Default is False.
    sandwich (bool): If True, use the sandwich variance estimator. Default is True.

    Returns:
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The standard errors if return_stderr is True, otherwise the lower and upper bounds of the confidence interval.
    """
    n = X.shape[0]  # Number of samples
    thetahat = ols(X, Y)  # OLS coefficients
    Sigmainv = np.linalg.inv(1/n * X.T @ X)  # Inverse of the covariance matrix

    if sandwich:
        M = 1/n * (X.T * ((Y - X @ thetahat) ** 2)[None, :]) @ X
    else:
        M = 1/n * ((Y - X @ thetahat) ** 2).mean() * X.T @ X

    V = Sigmainv @ M @ Sigmainv  # Variance-covariance matrix
    stderr = np.sqrt(np.diag(V))  # Standard errors

    if return_stderr:
        return stderr

    halfwidth = norm.ppf(1 - alpha / 2) * stderr / np.sqrt(n)  # Half-width of the confidence interval
    return thetahat - halfwidth, thetahat + halfwidth  # Confidence interval

def pp_ols_interval(X_labeled: np.ndarray, X_unlabeled: np.ndarray, Y_labeled: np.ndarray, 
                    Yhat_labeled: np.ndarray, Yhat_unlabeled: np.ndarray, alpha: float, 
                    sandwich: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the confidence interval for post-processed OLS regression coefficients.

    Parameters:
    X_labeled (np.ndarray): The labeled input feature matrix (design matrix).
    X_unlabeled (np.ndarray): The unlabeled input feature matrix (design matrix).
    Y_labeled (np.ndarray): The labeled outcome vector (response variable).
    Yhat_labeled (np.ndarray): The predicted outcomes for the labeled data.
    Yhat_unlabeled (np.ndarray): The predicted outcomes for the unlabeled data.
    alpha (float): Significance level for the confidence interval.
    sandwich (bool): If True, use the sandwich variance estimator. Default is True.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The lower and upper bounds of the confidence interval.
    """
    n = X_labeled.shape[0]  # Number of labeled samples
    N = X_unlabeled.shape[0]  # Number of unlabeled samples

    # Compute OLS coefficients for the unlabeled data
    thetatildef = ols(X_unlabeled, Yhat_unlabeled)
    
    # Compute the rectifier OLS coefficients for the labeled data
    rectifierhat = ols(X_labeled, Y_labeled - Yhat_labeled)
    
    # Post-processed OLS coefficients
    pp_thetahat = thetatildef + rectifierhat
    
    # Compute standard errors for the unlabeled and labeled data
    stderr_tildef = classical_ols_interval(X_unlabeled, Yhat_unlabeled, 0.001 * alpha, return_stderr=True, sandwich=sandwich)
    stderr_rec = classical_ols_interval(X_labeled, Y_labeled - Yhat_labeled, 0.999 * alpha, return_stderr=True, sandwich=sandwich)
    
    # Compute the half-width of the confidence interval
    halfwidth = norm.ppf(1 - alpha / 2) * np.sqrt((stderr_rec**2 / n) + (stderr_tildef**2 / N))
    
    # Return the confidence interval
    return pp_thetahat - halfwidth, pp_thetahat + halfwidth

"""
    Logistic regression algorithm
"""
def logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Perform logistic regression on the given data.

    Parameters:
    X (np.ndarray): The input feature matrix (design matrix).
    y (np.ndarray): The binary outcome vector (response variable).

    Returns:
    np.ndarray: The coefficients of the logistic regression model.
    """
    # Initialize and fit the logistic regression model
    clf = LogisticRegression(
        penalty='none', 
        solver='lbfgs', 
        max_iter=10000, 
        tol=1e-15, 
        fit_intercept=False
    ).fit(X, y)
    
    # Return the coefficients of the fitted model
    return clf.coef_.squeeze()

def product(*args: tuple, **kwds: dict) -> iter:
    """
    Cartesian product of input iterables.

    Parameters:
    *args (tuple): Variable length argument list of input iterables.
    **kwds (dict): Variable length keyword arguments (not used in this function).

    Yields:
    iter: An iterator over tuples representing the Cartesian product of the input iterables.

    Examples:
    ---------
    product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    """
    pools = map(tuple, args)  # Convert each input iterable to a tuple
    result = [[]]  # Initialize the result with an empty list

    # Generate the Cartesian product
    for pool in pools:
        result = [x + [y] for x in result for y in pool]

    # Yield each product as a tuple
    for prod in result:
        yield tuple(prod)

def classical_logistic_interval(X: np.ndarray, Y: np.ndarray, alpha: float, num_grid: int = 500) -> list:
    """
    Calculate the classical logistic confidence interval.

    Parameters:
    X (np.ndarray): The input feature matrix (design matrix).
    Y (np.ndarray): The binary outcome vector (response variable).
    alpha (float): The significance level for the confidence interval.
    num_grid (int, optional): The number of grid points to use for the interval calculation. Default is 500.

    Returns:
    list: A list containing the minimum and maximum values of the confidence interval.
    """
    n = X.shape[0]  # Number of samples
    d = X.shape[1]  # Number of features
    Y = (Y >= 0.5).astype(int)  # Convert Y to binary (0 or 1)
    
    # Obtain the point estimate using logistic regression
    point_estimate = logistic(X, Y)

    # Create a grid of theta values around the point estimate
    theta_grid = np.concatenate([
        np.linspace(-3 * point_estimate, point_estimate, num_grid // 2),
        np.linspace(point_estimate, 3 * point_estimate, num_grid // 2)[1:]
    ])

    # Calculate the expected value of the logistic function
    mu = expit(X @ theta_grid.T)
    
    # Calculate the gradient
    g = 1 / n * X.T @ (mu - Y[:, None])

    # Estimate the standard error
    sigmahat_err = np.std(X[:, :, None] * (mu - Y[:, None])[:, None, :], axis=0)
    
    # Calculate the half-width of the gradient confidence interval
    grad_halfwidth = norm.ppf(1 - alpha / (2 * d)) * sigmahat_err / np.sqrt(n)

    # Determine the condition for the confidence interval
    condition = np.all(np.abs(g) <= grad_halfwidth, axis=0)

    # Extract the confidence interval points
    Cpp = theta_grid[condition]

    # TODO: If all positive, make grid wider
    
    # Ensure the condition is met at the boundaries
    assert (condition[0] == False) & (condition[-1] == False)

    # Return the minimum and maximum values of the confidence interval
    return [Cpp.min(axis=0), Cpp.max(axis=0)]

def pp_logistic_interval(X_labeled: np.ndarray, X_unlabeled: np.ndarray, Y_labeled: np.ndarray, 
                         Yhat_labeled: np.ndarray, Yhat_unlabeled: np.ndarray, alpha: float, 
                         num_grid: int = 500) -> list:
    """
    Calculate the logistic confidence interval using both labeled and unlabeled data.

    Parameters:
    X_labeled (np.ndarray): The labeled input feature matrix.
    X_unlabeled (np.ndarray): The unlabeled input feature matrix.
    Y_labeled (np.ndarray): The binary outcome vector for labeled data.
    Yhat_labeled (np.ndarray): The predicted probabilities for labeled data.
    Yhat_unlabeled (np.ndarray): The predicted probabilities for unlabeled data.
    alpha (float): The significance level for the confidence interval.
    num_grid (int, optional): The number of grid points to use for the interval calculation. Default is 500.

    Returns:
    list: A list containing the minimum and maximum values of the confidence interval.
    """
    
    # Combine labeled and unlabeled feature matrices
    X = np.concatenate([X_labeled, X_unlabeled], axis=0)
    
    # Number of labeled samples
    n = X_labeled.shape[0]
    
    # Number of features
    d = X_labeled.shape[1]
    
    # Number of unlabeled samples
    N = X_unlabeled.shape[0]
    
    # Clip predicted probabilities to be within [0, 1]
    Yhat_labeled = np.clip(Yhat_labeled, 0, 1)
    Yhat_unlabeled = np.clip(Yhat_unlabeled, 0, 1)

    # Combine predicted probabilities
    Yhat = np.concatenate([Yhat_labeled, Yhat_unlabeled], axis=0)
    
    # Obtain the point estimate using logistic regression on labeled data
    point_estimate = logistic(X_labeled, (Y_labeled > 0.5).astype(int))

    # Calculate the rechat term
    rechat = 1/n * X_labeled.T @ (Yhat_labeled - Y_labeled)
    
    # Estimate the standard error for the rechat term
    sigmahat_rec = np.std(X_labeled * (Yhat_labeled - Y_labeled)[:, None], axis=0)

    # Create a grid of theta values around the point estimate
    theta_grid = np.concatenate([
        np.linspace(-3 * point_estimate, point_estimate, num_grid // 2),
        np.linspace(point_estimate, 3 * point_estimate, num_grid // 2)[1:]
    ])

    # Calculate the expected value of the logistic function for the unlabeled data
    mu = expit(X_unlabeled @ theta_grid.T)
    
    # Calculate the gradient for the unlabeled data
    g = 1/N * X_unlabeled.T @ (mu - Yhat_unlabeled[:, None])

    # Estimate the standard error for the gradient
    sigmahat_err = np.std(X_unlabeled[:, :, None] * (mu - Yhat_unlabeled[:, None])[:, None, :], axis=0)

    # Calculate the half-width of the confidence interval
    halfwidth = norm.ppf(1 - alpha / (2 * d)) * np.sqrt(sigmahat_rec[:, None]**2 / n + sigmahat_err**2 / N)

    # Determine the condition for the confidence interval
    condition = np.all(np.abs(g + rechat[:, None]) <= halfwidth, axis=0)

    # Extract the confidence interval points
    Cpp = theta_grid[condition]

    # TODO: If all positive, make grid wider
    assert (condition[0] == False) & (condition[-1] == False)

    # Return the minimum and maximum values of the confidence interval
    return [Cpp.min(axis=0), Cpp.max(axis=0)]

"""
    DISCRETE L_p ESTIMATION RATES
"""

def linfty_dkw(N: int, K: int, alpha: float) -> float:
    """
    Calculate the L-infinity Dvoretzky-Kiefer-Wolfowitz (DKW) bound.

    Parameters:
    N (int): The number of samples.
    K (int): The number of bins or categories.
    alpha (float): The significance level.

    Returns:
    float: The L-infinity DKW bound.
    """
    return np.sqrt(2 / N * np.log(2 / alpha))

def linfty_binom(N: int, K: int, alpha: float, qhat: np.ndarray) -> float:
    """
    Calculate the L-infinity binomial bound.

    Parameters:
    N (int): The number of samples.
    K (int): The number of bins or categories.
    alpha (float): The significance level.
    qhat (np.ndarray): The estimated probabilities for each bin.

    Returns:
    float: The L-infinity binomial bound.
    """
    epsilon = 0
    for k in np.arange(K):
        bci = binomial_iid(N, alpha / K, qhat[k])
        epsilon = np.maximum(epsilon, np.abs(bci - qhat[k]).max())
    return epsilon
"""
	SAMPLING WITHOUT REPLACEMENT
"""
def clt_swr(x: np.ndarray, N: int, alpha: float) -> np.ndarray:
    """
    Calculate the confidence interval for the mean of a sample without replacement using the Central Limit Theorem (CLT).

    Parameters:
    x (np.ndarray): The sample data as a numpy array.
    N (int): The total population size.
    alpha (float): The significance level.

    Returns:
    np.ndarray: A numpy array containing the lower and upper bounds of the confidence interval.
    """
    n = x.shape[0]  # Sample size
    point_estimate = x.mean()  # Mean of the sample
    # Standard deviation of the sample adjusted for finite population correction
    fluctuations = x.std() * norm.cdf(1 - alpha / 2) * np.sqrt((N - n) / (N * n))
    # Return the confidence interval as a numpy array
    return np.array([point_estimate - fluctuations, point_estimate + fluctuations])

def wsr_swr(x: np.ndarray, N: int, alpha: float, grid: np.ndarray, num_cpus: int = 10, intersection: bool = True) -> np.ndarray:
    """
    Calculate the confidence interval for a bounded sequence using weighted sampling without replacement.

    Parameters:
    x (np.ndarray): A [0,1] bounded sequence.
    N (int): The total population size.
    alpha (float): The significance level.
    grid (np.ndarray): The grid of values to evaluate.
    num_cpus (int, optional): The number of CPUs to use for parallel processing. Default is 10.
    intersection (bool, optional): Whether to use intersection in the calculation. Default is True.

    Returns:
    np.ndarray: A numpy array containing the lower and upper bounds of the confidence interval.
    """
    n = x.shape[0]  # Sample size

    def mu(m: float, i: int) -> np.ndarray:
        """Calculate the mean adjusted for finite population correction."""
        return (N * m - np.concatenate([np.array([0]), np.cumsum(x[:i-1])])) / (N - (np.arange(i) + 1) + 1)

    muhats = (1/2 + np.cumsum(x)) / (np.arange(n) + 1)  # Estimated means
    sigmahat2s = (1/4 + np.cumsum((x - muhats) ** 2)) / (np.arange(n) + 1)  # Estimated variances
    lambdas = np.concatenate([np.array([1]), np.sqrt(2 * np.log(2 / alpha) / (n * sigmahat2s))[:-1]])  # Lambda values

    def M(m: float, i: int) -> float:
        """Calculate the M value for the given m and i."""
        return 1/2 * np.maximum(
            np.prod(1 + np.minimum(lambdas[:i], 0.5 / mu(m, i)) * (x[:i] - mu(m, i))),
            np.prod(1 - np.minimum(lambdas[:i], 0.5 / (1 - mu(m, i))) * (x[:i] - mu(m, i)))
        )

    M = np.vectorize(M)  # Vectorize the M function for parallel processing

    if intersection:
        M_list = Parallel(n_jobs=num_cpus)(delayed(M)(grid, i) for i in range(1, n + 1))
    else:
        M_list = [M(grid, n)]

    ci_full = grid[np.where(np.prod(np.stack(M_list, axis=1) < 1 / alpha, axis=1))[0]]
    return np.array([ci_full.min(), ci_full.max()])  # Only output the interval
