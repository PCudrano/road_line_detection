import numpy as np
import warnings
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from scipy.interpolate import UnivariateSpline

class Fitting:
    min_n_pts_for_fitting_extra = 2

    class FittingException(Exception):
        pass

    @classmethod
    def fit_poly(cls, x, y, order, ret_cov=False):
        cls.check_enough_points(x.shape[0], order)
        fit, cov = np.polyfit(x, y, order, cov=True)
        if ret_cov:
            return np.poly1d(fit), cov
        return np.poly1d(fit)

    @classmethod
    def fit_poly_robust(cls, x, y, order, epsilon_arr=(1.35,), max_iter=100, tol=1e-3, alpha=100):
        cls.check_enough_points(x.shape[0], order)
        points_poly_robust_features = np.vander(x, order + 1)[:, :-1]  # -1: remove x^0, cause I fit intercept separately
        # epsilon_arr = [1.35]  # [1.1, 1.2, 1.35, 1.5]
        eps_i = 0
        huber_converged = False
        while not huber_converged and eps_i < len(epsilon_arr):
            epsilon = epsilon_arr[eps_i]
            try:
                huber = HuberRegressor(epsilon=epsilon, max_iter=max_iter, tol=tol, fit_intercept=True, alpha=alpha)  # tol: 1 mm
                huber.fit(points_poly_robust_features, y)
                fit = np.hstack((huber.coef_, huber.intercept_))
                huber_converged = True
            except ValueError as e:
                warnings.warn("Huber with epsilon {} not converged, relaxing epsilon...".format(epsilon))
            eps_i += 1
        if not huber_converged:
            warnings.warn("No Huber model converged, using OLS")
            return cls.fit_poly(x, y, order)
        return np.poly1d(fit)

    @classmethod
    def fit_poly_reg(cls, x, y, order, reg=0.0):
        x = np.asarray(x) + 0.0
        y = np.asarray(y) + 0.0
        deg = np.asarray(order)
        # check arguments.
        if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
            raise TypeError("deg must be an int or non-empty 1-D array of int")
        if deg.min() < 0:
            raise ValueError("expected deg >= 0")
        if x.ndim != 1:
            raise TypeError("expected 1D vector for x")
        if x.size == 0:
            raise TypeError("expected non-empty vector for x")
        if y.ndim < 1 or y.ndim > 2:
            raise TypeError("expected 1D or 2D array for y")
        if len(x) != len(y):
            raise TypeError("expected x and y to have same length")
        lmax = deg
        ord = lmax + 1
        van = np.polynomial.polynomial.polyvander(x, lmax)
        # set up the least squares matrices in transposed form
        lhs = van.T
        rhs = y.T
        # set rcond
        rcond = len(x) * np.finfo(x.dtype).eps
        scl = np.sqrt(np.square(lhs).sum(1))
        scl[scl == 0] = 1
        # Solve the least squares problem.
        # c, resids, rank, s = np.linalg.lstsq(lhs.T/scl, rhs.T, rcond) # TO-DO modify
        lhs_fit = lhs.T / scl
        rhs_fit = rhs.T
        c = np.linalg.solve(lhs_fit.T.dot(lhs_fit) + reg * np.identity(lhs_fit.shape[1]), lhs_fit.T.dot(rhs_fit))
        c = (c.T / scl).T
        # warn on rank reduction
        # if rank != ord:
        #     msg = "The fit may be poorly conditioned"
        #     warnings.warn(msg, np.polynomial.polyutils.RankWarning, stacklevel=2)
        return c[::-1]

    @classmethod
    def fit_spline(cls, x, y, order, smoothing_factor=None, weights=None):
        cls.check_enough_points(x.shape[0], order)
        x_sort_i = np.argsort(x)
        xpts_sorted = x[x_sort_i]
        ypts_sorted = y[x_sort_i]
        # remove duplicates (can't fit spline with duplicate absissae
        duplicates = np.argwhere(np.diff(xpts_sorted) == 0)
        xpts_sorted = np.delete(xpts_sorted, duplicates)
        ypts_sorted = np.delete(ypts_sorted, duplicates)
        if xpts_sorted.size < order + 1:
            raise Fitting.FittingException(
                "After removing points with same abscissae, "
                "not enough points to build the model: given {} points, needed >={} points"
                    .format(xpts_sorted.size, order + 1))
        try:
            return UnivariateSpline(xpts_sorted, ypts_sorted, k=order, s=smoothing_factor, w=weights)
        except Exception as e:
            raise Fitting.FittingException("Exception in fitting spline: {}".format(e))

    @classmethod
    def fit_poly_bayesian(cls, x, y, order):
        cls.check_enough_points(x.shape[0], order)
        bayesian_model = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06,
                                       lambda_2=1e-06, fit_intercept=True)
        points_poly_features = np.vander(x, order + 1)[:, :-1]  # -1: remove x^0, cause I fit intercept separately
        bayesian_model.fit(points_poly_features, y)
        return np.poly1d(np.hstack((bayesian_model.coef_, bayesian_model.intercept_)))  # create poly1d with coefficients of mean polynomial estimated

    @classmethod
    def check_enough_points(cls, n_points, order):
        min_required = order + cls.min_n_pts_for_fitting_extra
        if n_points < min_required:
            raise Fitting.FittingException("Not enough points to build the model: given {} points, required >={} points"
                    .format(n_points, min_required))

