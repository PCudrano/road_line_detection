#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
from math import cos, sin, pi, atan2, ceil, floor, radians, degrees, modf
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
import scipy.stats as stats
import scipy.optimize as scopt
from scipy.integrate import quadrature
from numpy import exp, abs
from datetime import timedelta
import time

#configs = type('obj', (object,), {'use_cpp' : False});


#from py_configs import configs
from includes.utils import spline_neumann, rotate_image_fit, distance_point_from_line_through_points, polyfit_reg
# from includes.TrackedLine import WorldLineArcLengthParamUpdatableMockup

from matplotlib import pyplot as plt

class Line(object):
    """
    Line parent class, contains the coordinates of some points and fits on them a model of the form:
    line: x = f(y).
    The class can be further specialized as ImageLine or WorldLine.
    Notice that, usually, in order to avoid numerical issues, image models are fit considering the image y coord
    as abscissae and the image x coord as ordinate, but this aspect is not considered in this class: see class
    ImageLine for the appropriate references.
    """
    class LineException(Exception):
        pass

    def __init__(self, points, model_type="poly", model_order=2, flipXY=False, id=None, spline_s=1, min_n_pts_for_fitting=None):
        """
        Creates a line object.
        :param points: points to be used for the model fitting, passed as np.array([[x1,y1],...[xn,yn]])
        :param model_type: "poly" or "spline", only two models currently available
        :param model_order: order of the model (e.g. 2 indicates a second order model)
        :param flipXY: default False. If True, saves the points as passed, but fits the model as line: x = f(y). This is the standard
        fitting method for road lines in images.
        :param id: line id, optional
        """
        self._points = points # [[x1,y1],...[xn,yn]]
        self.model_type = model_type
        self.model_order = model_order
        self.min_n_pts_for_fitting = min_n_pts_for_fitting if min_n_pts_for_fitting is not None else model_order + 2
        self.flipXY = flipXY
        self._setupFitPoints()
        self.id = id
        self.spline_s = spline_s
        self._model = self._fitModel()

    def getPoints(self):
        return self._points

    def getFittingPoints(self):
        return self._fit_points

    def getModel(self):
        return self._model

    def getModelType(self):
        return self.model_type

    def getModelOrder(self):
        return self.model_order

    ## Private methods

    def _setupFitPoints(self):
        if self.flipXY:
            self._fit_points = np.fliplr(self._points)
        else:
            self._fit_points = self._points

    def _fitModel(self):
        self._checkEnoughPoints()

        if self.model_type == "poly":
            fit, self.cov = np.polyfit(self._fit_points[:, 0], self._fit_points[:, 1], self.model_order, cov=True)
            return np.poly1d(fit)
        elif self.model_type == "poly_robust":
            points_poly_robust_features = np.vander(self._fit_points[:, 0], self.model_order + 1)[:, :-1]  # -1: remove x^0, cause I fit intercept separately
            epsilon_arr = [1.35] #[1.1, 1.2, 1.35, 1.5]
            eps_i = 0
            huber_converged = False
            while not huber_converged and eps_i < len(epsilon_arr):
                epsilon = epsilon_arr[eps_i]
                try:
                    huber = HuberRegressor(epsilon=epsilon, max_iter=100, tol=1e-3, fit_intercept=True, alpha=100)  # tol: 1 mm
                    huber.fit(points_poly_robust_features, self._fit_points[:, 1])
                    fit = np.hstack((huber.coef_, huber.intercept_))
                    huber_converged = True
                except ValueError as e:
                    pass
                eps_i += 1
            if not huber_converged:
                fit = np.polyfit(points_poly_robust_features, self._fit_points[:, 1], self.getModelOrder())
            return np.poly1d(fit)
        elif self.model_type == "spline":
            xpts_sorted = self._fit_points[np.argsort(self._fit_points[:, 0]), 0]
            ypts_sorted = self._fit_points[np.argsort(self._fit_points[:, 0]), 1]
            # remove duplicates (can't fit spline with duplicate absissae
            duplicates = np.argwhere(np.diff(xpts_sorted) == 0)
            xpts_sorted = np.delete(xpts_sorted, duplicates)
            ypts_sorted = np.delete(ypts_sorted, duplicates)
            if xpts_sorted.size < self.model_order+1:
                raise Line.LineException(
                    "After removing points with same abscissae, "
                    "not enough points to build the model: given {} points, needed >={} points"
                                         .format(xpts_sorted.size, self.model_order + 1))
            try:
                return UnivariateSpline(xpts_sorted, ypts_sorted, k=self.model_order, s=self.spline_s)
            except:
                # print("SplineException")
                raise Line.LineException("Exception in fitting spline")
        elif self.model_type == "bayesian_poly":
            self._bayesian_model = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06,
                                          lambda_2=1e-06, fit_intercept=True)
            points_poly_features = np.vander(self._fit_points[:, 0], self.model_order + 1)[:, :-1] # -1: remove x^0, cause I fit intercept separately
            self._bayesian_model.fit(points_poly_features, self._fit_points[:, 1])
            return np.poly1d(np.hstack((self._bayesian_model.coef_, self._bayesian_model.intercept_))) # create poly1d with coefficients of mean polynomial estimated
        elif self.model_type.startswith("arc_length_parametrization"):
            raise NotImplementedError("Model arc_length_parametrization not implemented for standard Line. Use class WorldLineArcLengthParam instead.")
        else:
            raise NotImplementedError("Line model {:} not allowed.".format(self.model_type))

    def _checkEnoughPoints(self):
        # check number of points enough to fit a polynomial model
        if self._fit_points.shape[0] < self.min_n_pts_for_fitting:
            raise Line.LineException("Not enough points to build the model: given {} points, needed >={} points"
                                     .format(self._points.shape[0], self.min_n_pts_for_fitting))

    def _computeOrdinateGivenAbscissae(self, abs_pts):
        return self._model(abs_pts)

    def computeConfidenceInterval(self, abs_pts, confidence=0.95):
        """
        Two tails confidence interval. Only for model_type == "bayesian_poly"
        :param abs_pts: abscissae points for which we want to compute the condifence of the estimate
        :param confidence: condidence level (also found as 1-alpha in literature). Default: 0.95
        :return: array of arrays, each row is [min,max] of the confidence interval (i.e. mean +/ margin of error)
        """
        assert self.model_type in ["bayesian_poly", "poly"], "Confidence estimation is available only with bayesian_poly and poly models"
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)  # 2 tails
        if self.model_type == "bayesian_poly":
            abs_pts_poly_features = np.vander(abs_pts, self.model_order + 1)[:, :-1]  # -1: remove x^0, cause I fit intercept separately
            y_mean, y_std = self._bayesian_model.predict(abs_pts_poly_features, return_std=True)
            return np.vstack((y_mean-z*y_std, y_mean+z*y_std)).T
        elif self.model_type == "poly":
            y_mean = self._computeOrdinateGivenAbscissae(abs_pts)
            abs_pts_poly_features = np.vander(abs_pts, self.model_order + 1)
            y_var = np.diagonal(abs_pts_poly_features.dot(self.cov).dot(abs_pts_poly_features.T)) # var(y^) = x^4 var(a^) + x^2 var(b^) + 1 var(c^)
            y_std = np.sqrt(y_var)
            return np.vstack((y_mean - z * y_std, y_mean + z * y_std)).T

    def computeCurvature(self):
        """
        Computes an approzimated curvature and its sign
        :return: curvature (in meters), curvature_side (+1 left, -1 right)
        """
        assert self.model_type in ["poly",
                                   "bayesian_poly"], "Curvature estimation available only for poly and bayesian_poly and poly models"
        n_points = 100
        x_pts = np.linspace(np.min(self._points[:, 0]), np.max(self._points[:, 1]), n_points)
        y_pts = self._computeOrdinateGivenAbscissae(x_pts)
        points = np.vstack((x_pts, y_pts)).T
        dx_dt = np.gradient(points[:, 0])
        dy_dt = np.gradient(points[:, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        signed_curvatures = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        curvatures = np.abs(signed_curvatures)
        curvature = np.mean(curvatures)
        # curvature_side = np.sign(np.mean(signed_curvatures))
        (curvature_sides, curvature_sides_counts) = np.unique(np.sign(signed_curvatures), return_counts=True)
        curvature_side = curvature_sides[np.argmax(curvature_sides_counts)]
        return curvature, curvature_side

    def computeRadiusOfCurvature(self):
        """
        Computes an approzimated radius of curvature and its sign
        :return: radius_of_curvature (in meters), curvature_side (+1 left, -1 right)
        """
        assert self.model_type in ["poly", "bayesian_poly"], "Curvature estimation available only for poly and bayesian_poly and poly models"
        curvature, curvature_side = self.computeCurvature()
        radius_of_curvature = 1.0 / curvature
        return radius_of_curvature, curvature_side

class ImageLine(Line):
    """
    Line with forced flipXY=True (model fit using image y coord as abscissae and image x coord as ordinate)
    """
    def __init__(self, points, model_type="poly", model_order=2, id=None):
        super(ImageLine, self).__init__(points, model_type=model_type, model_order=model_order, flipXY=True, id=id)

    def computeXGivenY(self, y_pts):
        return self._computeOrdinateGivenAbscissae(y_pts)

class WorldLine(Line):
    def __init__(self, bev_obj, points, model_type="poly", model_order=2, id=None, spline_s=1,
                 min_n_pts_for_fitting=None, min_m_range_for_fitting=None):
        self.bev_obj = bev_obj
        self.min_m_range_for_fitting = min_m_range_for_fitting
        super(WorldLine, self).__init__(points, model_type=model_type, model_order=model_order, flipXY=False, id=id, spline_s=spline_s,
                                        min_n_pts_for_fitting=min_n_pts_for_fitting)
        if len(self.getPoints()) > 0:
            self.line_points_x_limit = (np.min(self.getPoints()[:, 0]), np.max(self.getPoints()[:, 0])) # interpolation interval

    def computeYGivenX(self, world_y_pts):
        return super(WorldLine, self)._computeOrdinateGivenAbscissae(world_y_pts)

    def getBevLineProxy(self, **kwargs):
        return BevLineProxy(self, **kwargs)

    def getBevObj(self):
        return self.bev_obj

    def getPointsXLimits(self):
        return self.line_points_x_limit

    # Private
    def _checkEnoughPoints(self):
        super(WorldLine, self)._checkEnoughPoints()
        if self.min_m_range_for_fitting:
            # check that range of points is larger than min_m_range_for_fitting
            pts_x_range = np.max(self._fit_points[:,0]) - np.min(self._fit_points[:,0])
            if pts_x_range < self.min_m_range_for_fitting:
                raise Line.LineException("X points range not large enough: given range {} m, needed range {} m"
                                         .format(pts_x_range, self.min_n_pts_for_fitting))

class WeightedSplineLine(WorldLine):

    def __init__(self, bev_obj, points, weights, **kwargs):
        self.weights = weights
        super(WeightedSplineLine, self).__init__(bev_obj, points, model_type="spline", **kwargs)

    def _fitModel(self):
        self._checkEnoughPoints()
        # fit spline
        xpts_sorted = self._fit_points[np.argsort(self._fit_points[:, 0]), 0]
        ypts_sorted = self._fit_points[np.argsort(self._fit_points[:, 0]), 1]
        # remove duplicates (can't fit spline with duplicate absissae
        duplicates = np.argwhere(np.diff(xpts_sorted) == 0)
        xpts_sorted = np.delete(xpts_sorted, duplicates)
        ypts_sorted = np.delete(ypts_sorted, duplicates)
        if xpts_sorted.size < self.model_order + 1:
            raise Line.LineException(
                "After removing points with same abscissae, "
                "not enough points to build the model: given {} points, needed >={} points"
                    .format(xpts_sorted.size, self.model_order + 1))
        try:
            return UnivariateSpline(xpts_sorted, ypts_sorted, k=self.model_order, s=self.spline_s, w=self.weights)
        except:
            # print("SplineException")
            raise Line.LineException("Exception in fitting spline")

class WorldLineArcLengthParam(WorldLine):

    LOOKUP_TABLE_S = 0
    LOOKUP_TABLE_THETA = 1
    LOOKUP_TABLE_X = 2
    LOOKUP_TABLE_Y = 3

    def __init__(self, bev_obj, points, model_type="arc_length_parametrization", model_order=2, model_lsq_reg=None, id=None,
                 recenter_first_pt=False, fixed_origin_x=None, min_n_pts_for_fitting=None, min_m_range_for_fitting=None, delta_pts=0.01):
        # points: [[x,y],...]
        self.recenter_first_pt = recenter_first_pt
        self.fixed_origin_x = fixed_origin_x
        self.model_lsq_reg = model_lsq_reg
        self.delta_pts = delta_pts  # 0.25 # FIXME 0.1
        super(WorldLineArcLengthParam, self).__init__(bev_obj, points, model_type=model_type, model_order=model_order, id=id,
                                                      min_n_pts_for_fitting=min_n_pts_for_fitting, min_m_range_for_fitting=min_m_range_for_fitting)
        self.min_n_pts = 20 # 2 m given delta_pts is 0.1 m
        self._initLookupTable(delta_pts=self.delta_pts, min_n_pts=self.min_n_pts)
        self.reached_limit_max = False
        self.reached_limit_min = False
    def clone(self):
        return WorldLineArcLengthParam(self.bev_obj,
                                       self.getPoints(),
                                       model_order=self.getModelOrder(),
                                       id=self.id,
                                       recenter_first_pt=self.recenter_first_pt,
                                       fixed_origin_x=self.fixed_origin_x)

    def getModel(self):
        return self._model

    def getModelCutoffParams(self):
        if "cutoff" in self.getModelType():
            return self.left_cutoff_s, self.left_cutoff_theta, self.right_cutoff_s, self.right_
        return None

    def applyModel(self, s):
        return self.apply_model(s)

    def getOrigin(self):
        return self.origin.copy()

    def getFitS(self):
        return self.s

    def getFitTheta(self):
        return self.theta

    def getOriginalFitS(self):
        return self.original_s

    def getLookupTable(self):
        return self.lookup_table

    # def getBevLineProxy(self):
    #     return BevLineArcLengthParamProxy(self)

    ### Private

    def _fitModel(self):
        self._checkEnoughPoints()

        t0 = time.time()
        if self.recenter_first_pt:
            self._recenterFirstPoint()
        t10 = time.time()

        t11 = time.time()
        self._computeSThetaPoints()
        t20 = time.time()

        t21 = time.time()
        model, model_fcn = self._fitModelInS() # return super(WorldLineArcLengthParam, self)._fitModelInS()
        self.apply_model = model_fcn
        t3 = time.time()

        return model

    def _computeSThetaPoints(self):
        self._fit_points_ordered = self._fit_points[np.argsort(self._fit_points[:,0]),:] # order points wrt their indep var
        # self._fit_points_ordered = self._fit_points # TODO test!! ripristinate previous line instead!!!
        self.origin = self._fit_points_ordered[0,:]
        pts_diff = np.diff(self._fit_points_ordered, axis=0)
        delta_s = np.sqrt(np.sum(np.square(pts_diff), axis=1)) # for each point i: sqrt( (x[i]-x[i-1])^2 + (y[i]-y[i-1])^2 )
        self.s = np.cumsum(delta_s)
        self.theta = np.unwrap(np.arctan2(pts_diff[:,1], pts_diff[:,0])) # unwrap removes unnecessary angle discontinuities

        # # TODO TEST!!! RESTORE PREVIOUS CODE AFTER TEST
        # pts_diff_mod = self._fit_points_ordered[1:,:] - self._fit_points_ordered[0,:]
        # self.theta = np.unwrap(np.arctan2(pts_diff_mod[:, 1], pts_diff_mod[:, 0]))  # unwrap removes unnecessary angle discontinuities
        # # /TODO TEST

        # TEST
        # plt.figure(10);
        # plt.plot(self._fit_points_ordered[:, 1], self._fit_points_ordered[:, 0], '.r');
        # # plt.plot(np.vstack((self._fit_points_ordered[:, 1], self._fit_points_ordered[:, 1] + np.insert(pts_diff[:, 1], (-1), [0]))), np.vstack((self._fit_points_ordered[:, 0], self._fit_points_ordered[:, 0] + np.insert(pts_diff[:, 1], (-1), [0]))), '-og', markersize=.5);
        # for i in range(self._fit_points_ordered.shape[0]):
        #     plt.plot([self._fit_points_ordered[i, 1], self._fit_points_ordered[i, 1] + np.append(pts_diff[:, 1], 0)[i]],
        #              [self._fit_points_ordered[i, 0], self._fit_points_ordered[i, 0] + np.append(pts_diff[:, 0], 0)[i]], '-og', markersize=.5);
        # plt.gca().invert_xaxis()
        # plt.gca().set_aspect('equal', adjustable='box')


        self.original_s = self.getFitS()
        if self.fixed_origin_x is not None:
            self._imposeFixedOrigin()

    def _imposeFixedOrigin(self):
        # mock model fit exactly like this apart from fixing the origin
        current_line = self._cloneWithoutFixedOrigin()

        translation_result = current_line._translatedPointsForNewOriginX(self.fixed_origin_x)
        if translation_result is None:
            return False

        new_s, new_origin = translation_result
        self.s = new_s
        self.origin = new_origin
        return True

    def _translatedPointsForNewOriginX(self, new_origin_x):
        # fix origin in measured_line
        ## find delta s to get to new_origin_x
        row_at_origin_x = self.computeRowGivenX(np.array([new_origin_x]))[0, :]
        measured_intercept = row_at_origin_x[WorldLineArcLengthParam.LOOKUP_TABLE_Y]
        measured_intercept_s = row_at_origin_x[WorldLineArcLengthParam.LOOKUP_TABLE_S]  # need to translate points in s,theta of this amount
        translation_delta_s = measured_intercept_s # (negative if new_origin_x is before current origin, positive if it's after)

        if np.isnan(translation_delta_s):
            # couldn't change origin
            return None

        new_s = self.getFitS() - translation_delta_s # translate the origin leaving points as they are
        new_origin = np.array([new_origin_x, measured_intercept, 0])
        return new_s, new_origin

    def _cloneWithoutFixedOrigin(self):
        return WorldLineArcLengthParam(self.bev_obj,
                                       self.getPoints(),
                                       model_type="arc_length_parametrization",
                                       model_order=self.getModelOrder(),
                                       id=self.id,
                                       min_n_pts_for_fitting=self.min_n_pts_for_fitting,
                                       min_m_range_for_fitting=self.min_m_range_for_fitting,
                                       recenter_first_pt=self.recenter_first_pt,
                                       fixed_origin_x=None)

    def _fitModelInS(self, fixed_intercept=None):
        if self.s.shape[0] <= self.model_order+1:
            raise Line.LineException("Not enough s points to build the model: given {} s points, needed >={} points"
                                     .format(self.s.shape[0], self.model_order + 1))
        # fit excluding first point
        # OLS
        # fit = np.polyfit(self.s, self.theta, self.model_order)

        # Robust estimator - Repeated median regression
        # s_features = np.vander(self.s, self.model_order + 1)[:,:-1] # -1: remove x^0, cause I fit intercept separately
        # fit = stats.siegelslopes(self.theta, self.s) # TODO doesn't work w features, only first degree

        # spline with 0-derivative at extremes
        # arc_length_model = spline_neumann(self.s, self.theta, k=self.model_order, s=5)

        # Robust estimator - Huber regression
        # huber = HuberRegressor(alpha=0, epsilon=1.10, max_iter=100, tol=0.001, fit_intercept=True) # (alpha is the L2 regulariz term) alpha=0 causes instabilities, needs to leave default (1e-4) for stability!s
        # order s,theta for increasing s
        s_sorted_i = np.argsort(self.getFitS())
        s_sorted = self.getFitS()[s_sorted_i]
        theta_sorted = self.getFitTheta()[s_sorted_i]
        s_features_sorted = np.vander(s_sorted, self.getModelOrder() + 1)[:, :-1]  # -1: remove x^0, cause I fit intercept separately/

        # huber = HuberRegressor(epsilon=1.10, max_iter=100, tol=1e-3, fit_intercept=True) # tol: 1 mm
        # huber.fit(s_features_sorted, theta_sorted)
        # fit = np.hstack((huber.coef_, huber.intercept_))

        # Repeated Huber with relaxed epsilon constraint (to solve bad convergence issue)
        epsilon_arr = [] # [1.1, 1.2, 1.35, 1.5]
        eps_i = 0
        huber_converged = False
        while not huber_converged and eps_i < len(epsilon_arr):
            epsilon = epsilon_arr[eps_i]
            try:
                huber = HuberRegressor(epsilon=epsilon, max_iter=100, tol=1e-3, fit_intercept=True)  # tol: 1 mm
                huber.fit(s_features_sorted, theta_sorted)
                fit = np.hstack((huber.coef_, huber.intercept_))
                huber_converged = True
            except ValueError as e:
                pass
            eps_i += 1
        if not huber_converged:
            if fixed_intercept is not None:
                # s_features_sorted are already without intercept term
                fit = np.linalg.lstsq(s_features_sorted, theta_sorted - fixed_intercept)[0]
                fit = np.append(fit, fixed_intercept)
            else:
                # reg = self.model_lsq_reg # could be None if no reg is required
                # fit = None
                # while fit is None:
                #     try:
                #         fit = np.polyfit(s_sorted, theta_sorted, self.getModelOrder())
                #     except np.linalg.LinAlgError:
                #         # if error, add regularization term and keep increasing it
                #         if reg is None:
                #             reg = 1e-16
                #         else:
                #             reg *= 10 # increase regularization term
                try:
                    if self.model_lsq_reg is None:
                        fit = np.polyfit(s_sorted, theta_sorted, self.getModelOrder())
                    else:
                        fit = polyfit_reg(s_sorted, theta_sorted, self.getModelOrder(), reg=self.model_lsq_reg)
                except np.linalg.LinAlgError:
                    raise Line.LineException("WorldLineArcLengthParam._fitModelInS(): LinAlgError while fitting model")
        arc_length_model = np.poly1d(fit, variable='s')
        arc_length_model_fcn = np.poly1d(np.copy(fit), variable='s')

        # piecewise model: clip extremes of fitted model to avoid extrapolation with weird results
        # min_s = np.min(self.s)
        # max_s = np.max(self.s)
        # self.apply_model = lambda s: np.piecewise(s,
        #                                               [np.asarray(s) <= min_s,
        #                                                np.bitwise_and(np.asarray(s) > min_s, np.asarray(s) <= max_s),
        #                                                np.asarray(s) > max_s],
        #                                               [arc_length_model(min_s),
        #                                                lambda s: arc_length_model(s),
        #                                                arc_length_model(min_s)])
        # self.apply_model = arc_length_model # moved to self._fitModel()

        if "cutoff" in self.getModelType():
            self.left_cutoff_s = s_sorted[0]
            self.left_cutoff_theta = arc_length_model(self.left_cutoff_s)  # theta_sorted[0]
            self.right_cutoff_s = s_sorted[-1]
            self.right_cutoff_theta = arc_length_model(self.right_cutoff_s)
            poly_model = np.poly1d(np.copy(arc_length_model.coeffs), variable='s')
            #arc_length_model_fcn = self._mollifyModel(poly_model, left_cutoff_s, left_cutoff_theta, right_cutoff_s, right_cutoff_theta)
            arc_length_model_fcn = lambda s: np.piecewise(s,
                                                          [np.asarray(s) <= self.left_cutoff_s,
                                                           np.bitwise_and(np.asarray(s) > self.left_cutoff_s, np.asarray(s) <= self.right_cutoff_s),
                                                           np.asarray(s) > self.right_cutoff_s],
                                                          [self.left_cutoff_theta,
                                                           lambda s: poly_model(s),
                                                           self.right_cutoff_theta])
        fign = 40 + self.id if self.id is not None else 45
        return arc_length_model, arc_length_model_fcn

    def _recenterFirstPoint(self):
        """
        Since first detected point is often shifted to a side, using it as origin makes the whole line shift.
        Instead, we first fit the line without the first point, with origin in the second, more reliable,
        and then project the line back to the same abscissa of the first point to obtain a corrected origin
        """
        first_pt = self.getPoints()[0, :]
        first_pt_x = first_pt[0]
        points_wo_first_pt = self.getPoints()[1:, :]
        # fit line from second point
        line_wo_first_pt = WorldLineArcLengthParam(self.bev_obj, points_wo_first_pt, model_order=self.getModelOrder(),
                                                   model_lsq_reg=self.model_lsq_reg,
                                                   min_n_pts_for_fitting=self.min_n_pts_for_fitting, min_m_range_for_fitting=self.min_m_range_for_fitting,
                                                   delta_pts=self.delta_pts,
                                                   id=self.id, recenter_first_pt=False)
        # extrapolate line to abscissa of old origin
        new_origin_y = line_wo_first_pt.computeYGivenX(np.array([first_pt_x]))[0]
        if not np.isnan(new_origin_y):
            # if point is reachable by model (model does not wrap on itself earlier)
            new_origin = [first_pt_x, new_origin_y, 0] # add coordinate z=0
            # reconstruct points s.t. new_origin substitutes old first point
            new_right_line_artificial_points = np.insert(points_wo_first_pt, 0, new_origin, axis=0)
            # substitute line points with newly created ones
            self._points = new_right_line_artificial_points.copy()
            self._setupFitPoints()
            return True
        # else
        return False

    def _initLookupTable(self, delta_pts=None, n_pts=None, min_n_pts=None):
        assert (delta_pts is None and n_pts is not None) or (delta_pts is not None and n_pts is None) # delta_pts and n_pts mutually exclusive
        # self.lookup_table
        s_min = - 2 * abs(self.bev_obj.outView[0] - self.getOrigin()[0]) # min limit is negative s, 2 times delta from bev bottom limit and s origin point
        s_max = 2 * self.bev_obj.outView[1] # 2 times x_max, arbitrarily, should be long enough to cover all cases
        if delta_pts is None:
            delta_pts = (s_max - s_min) / n_pts
        self.s_step = delta_pts
        self.lookup_table =self._computeLookupTablePoints(s_min, s_max, self.origin) # [[s,theta,x,y], ...]
        self._setup_lookup_table_indices()

        self._cleanupLookupTable()

        if not (self._checkLookupTableEnoughPoints(min_n_pts) and self._checkLookupTableSound()):
            raise Line.LineException(
                "Line model in (s,theta) is not sound. This often happens when fitting a model with the provided points "
                "generates a spiral or an infeasible road line configuration. Check the points provided.")

    def _setup_lookup_table_indices(self):
        self.lookup_table_i_sort_s = np.argsort(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_S])
        # self.lookup_table_i_sort_theta = np.argsort(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_THETA])
        self.lookup_table_i_sort_x = np.argsort(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_X])
        # self.lookup_table_i_sort_y = np.argsort(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_Y])
        self.lookup_table_i_origin = np.argwhere(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_S] == 0)[0][0]

    def _cleanupLookupTable(self):
        x_pos = self.lookup_table[self.lookup_table_i_sort_s, WorldLineArcLengthParam.LOOKUP_TABLE_X][self.lookup_table_i_origin:] # x coord in order of generation (increasing s)
        x_pos_diff = np.diff(x_pos)
        x_pos_diff_sign = np.insert(np.sign(x_pos_diff) < 0, 0, 0)
        unsound_i_pos = np.where(x_pos_diff_sign)[0]
        if unsound_i_pos.size > 0:
            last_sound_i_pos = unsound_i_pos[0] + self.lookup_table_i_origin # last el to keep
            self.lookup_table = self.lookup_table[self.lookup_table_i_sort_s, :][:last_sound_i_pos, :] # remove elements not sound
            self.reached_limit_max = True
            self._setup_lookup_table_indices()
        x_neg = self.lookup_table[self.lookup_table_i_sort_s, WorldLineArcLengthParam.LOOKUP_TABLE_X][:self.lookup_table_i_origin]  # x coord in order of generation (increasing s)
        x_neg_reversed = np.flip(x_neg)
        x_neg_reversed_diff = np.diff(x_neg_reversed)
        x_neg_reversed_diff_sign = np.insert(np.sign(x_neg_reversed_diff) > 0, 0, 0)
        unsound_i_neg = np.where(x_neg_reversed_diff_sign)[0]
        if unsound_i_neg.size > 0:
            first_sound_i_neg = self.lookup_table_i_origin - unsound_i_neg[0] + 1 # first el to keep
            self.lookup_table = self.lookup_table[self.lookup_table_i_sort_s, :][first_sound_i_neg:, :]  # remove elements not sound
            self.reached_limit_min = True
            self._setup_lookup_table_indices()

    def _checkLookupTableSound(self):
        # idea: if x coordinate grows but then decreases, means that line is 'coming back' --> fail
        x = self.lookup_table[self.lookup_table_i_sort_s, WorldLineArcLengthParam.LOOKUP_TABLE_X] # x coord in order of generation (increasing s)
        return np.all(np.sign(np.diff(x)) >= 0) # True if all x are ordered ascending

    def _checkLookupTableEnoughPoints(self, min_n_pts):
        # idea: if cleaned up too many points from lookup table, then probably model not sound
        if min_n_pts is None:
            return True
        return self.lookup_table.shape[0] >= min_n_pts

    def _computeLookupTablePoints(self, s_min, s_max, origin):
        """
        Compute lookup table from s_min to 0 and from 0 to s_max separately, then merge. This way I'm certain to include
        s=0 (origin) in table and each other point is evenly spaced from there of s_step
        """
        assert s_min <=0 and s_max >= 0, 'Lookup table must include 0 in [s_min,s_max] interval, instead [{},{}] passed'.format(s_min, s_min)
        # positive side
        n_pts_pos = ceil((s_max - 0) / self.s_step) + 1 # +1 to include also last point in range
        s_pos = np.linspace(0, s_max, n_pts_pos)
        theta_pos = self.applyModel(s_pos)
        delta_s_pos = np.diff(s_pos)
        delta_x_pos = np.multiply(delta_s_pos, np.cos(theta_pos[:-1]))
        delta_y_pos = np.multiply(delta_s_pos, np.sin(theta_pos[:-1]))
        x_pos = origin[0] + np.cumsum(delta_x_pos)
        y_pos = origin[1] + np.cumsum(delta_y_pos)
        rows_pos = np.vstack((s_pos[1:], theta_pos[1:], x_pos, y_pos)).T # [[s1, theta1, x1, y1], ...]
        # origin row
        rows_origin = np.array([[0, theta_pos[0], origin[0], origin[1]]]) # [[s0, theta0, x0, y0]]
        # negative side
        n_pts_neg = ceil((0 - s_min) / self.s_step) + 1  # +1 to include also last point in range
        s_neg = np.linspace(s_min, 0, n_pts_neg)[::-1] # reverse order: [0, ..., s_min] (s_min negative)
        theta_neg = self.applyModel(s_neg) # throw away last theta
        delta_s_neg = np.diff(s_neg)
        delta_x_neg = np.multiply(delta_s_neg, np.cos(theta_neg[:-1]))
        delta_y_neg = np.multiply(delta_s_neg, np.sin(theta_neg[:-1]))
        x_neg = origin[0] + np.cumsum(delta_x_neg)
        y_neg = origin[1] + np.cumsum(delta_y_neg)
        rows_neg_reversed = np.vstack((s_neg[1:], theta_neg[1:], x_neg, y_neg)).T # [[s_-1, theta_-1, x_-1, y_-1], ..., [s_-n, theta_-n, x_-n, y_-n]]
        rows_neg = np.flipud(rows_neg_reversed) # [[s_-n, theta_-n, x_-n, y_-n], ..., [s_-1, theta_-1, x_-1, y_-1]]
        # stack sections
        table = np.vstack((rows_neg, rows_origin, rows_pos))
        return table

    def _extendLookupTableInX(self, x_needed):
        table_x_min = np.min(self.lookup_table[:,WorldLineArcLengthParam.LOOKUP_TABLE_X])
        table_x_max = np.max(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_X])
        if x_needed > table_x_max:
            if self.reached_limit_max:
                return False
            # integrate forward until I *pass* x_needed
            start_row = self.lookup_table[self.lookup_table_i_sort_s, :][-1,:]
            line_shape_is_sound = True
            cur_state = np.copy(start_row)
            while line_shape_is_sound and cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X] < x_needed:
                new_s = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_S] + self.s_step
                new_x = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X] + self.s_step * cos(cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_THETA])
                new_y = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_Y] + self.s_step * sin(cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_THETA])
                new_theta = self.applyModel(new_s)

                # check if new state is sound
                if new_x <= cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X]:
                    line_shape_is_sound = False
                    self.reached_limit_max = True # FIXME why wasn't this here before?
                else:
                    cur_state = np.array([new_s, new_theta, new_x, new_y])
                    self.lookup_table = np.vstack((self.lookup_table, [cur_state]))
        else:
            if self.reached_limit_min:
                return False
            # integrate backward until I *pass* x_needed
            start_row = self.lookup_table[self.lookup_table_i_sort_s, :][0]
            line_shape_is_sound = True
            cur_state = np.copy(start_row)
            while line_shape_is_sound and cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X] > x_needed:
                new_s = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_S] - self.s_step
                new_theta = self.applyModel(new_s)
                new_x = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X] - self.s_step * cos(new_theta)
                new_y = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_Y] - self.s_step * sin(new_theta)

                # check if new state is sound
                if new_x >= cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X]:
                    line_shape_is_sound = False
                    self.reached_limit_min = True # FIXME why wasn't this here before?
                else:
                    cur_state = np.array([new_s, new_theta, new_x, new_y])
                    self.lookup_table = np.vstack(([cur_state], self.lookup_table))
        if line_shape_is_sound:
            self._setup_lookup_table_indices()
            return True
        return False

    def _extendLookupTableInS(self, s_needed):
        table_s_min = np.min(self.lookup_table[:,WorldLineArcLengthParam.LOOKUP_TABLE_S])
        table_s_max = np.max(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_S])
        if s_needed > table_s_max:
            if self.reached_limit_max:
                return False
            # integrate forward until I *pass* s_needed
            start_row = self.lookup_table[self.lookup_table_i_sort_s, :][-1,:]
            line_shape_is_sound = True
            cur_state = np.copy(start_row)
            while line_shape_is_sound and cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_S] < s_needed:
                new_s = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_S] + self.s_step
                new_theta = self.applyModel(new_s)
                new_x = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X] + self.s_step * cos(cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_THETA])
                new_y = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_Y] + self.s_step * sin(cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_THETA])

                # check if new state is sound
                if new_x <= cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X]:
                    line_shape_is_sound = False
                    self.reached_limit_max = True
                else:
                    cur_state = np.array([new_s, new_theta, new_x, new_y])
                    self.lookup_table = np.vstack((self.lookup_table, [cur_state]))
        elif s_needed < table_s_min:
            if self.reached_limit_min:
                return False
            # integrate backward until I *pass* x_needed
            start_row = self.lookup_table[self.lookup_table_i_sort_s, :][0]
            line_shape_is_sound = True
            cur_state = np.copy(start_row)
            while line_shape_is_sound and cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_S] > s_needed:
                new_s = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_S] - self.s_step
                new_theta = self.applyModel(new_s)
                new_x = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X] - self.s_step * cos(new_theta)
                new_y = cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_Y] - self.s_step * sin(new_theta)
                # TODO this while loop could be converted in all-numpy calls!

                # check if new state is sound
                if new_x >= cur_state[WorldLineArcLengthParam.LOOKUP_TABLE_X]:
                    line_shape_is_sound = False
                    self.reached_limit_min = True
                else:
                    cur_state = np.array([new_s, new_theta, new_x, new_y])
                    self.lookup_table = np.vstack(([cur_state], self.lookup_table))
        else:
            return True
        if line_shape_is_sound:
            self._setup_lookup_table_indices()
            return True
        return False

    def computeYGivenX(self, world_x_pts, tol=1e-3):
        # retrieve starting row from lookup table (lookup x <= query x) for each x point
        computed_rows = self.computeRowGivenX(world_x_pts, tol=tol)
        return computed_rows[:, WorldLineArcLengthParam.LOOKUP_TABLE_Y]

    # def computePointsGivenS(self, s):
    #     # retrieve starting row from lookup table (lookup x <= query x)
    #     first_i_arr = np.array([np.argmax(
    #         self.lookup_table[self.lookup_table_i_sort_s, WorldLineArcLengthParam.LOOKUP_TABLE_S] > s_i) - 1
    #                             for s_i in s]) # index just before I expect to find my s_i
    #     feasible_i_arr = np.logical_and(first_i_arr >= 0, first_i_arr < self.lookup_table.shape[0] - 1) # for each s point requested, bool says if it's within allowed range
    #
    #     #if np.any(~feasible_i_arr): # unfeasible s
    #         # TODO should integrate with little step, but looks like here it just takes one step,
    #         #  so if record not in lookup table it's bad
    #
    #     # FIXME first_i_arr == -1 where not feasible --> if used like this below, it takes last element of array without any sense!! To reproduce, try extrapolating any line for negative s below what is in lookup table: will take weird values
    #     lookup_table_row_prev = (self.lookup_table[self.lookup_table_i_sort_s, :])[first_i_arr, :]
    #
    #     s_prev = lookup_table_row_prev[:, WorldLineArcLengthParam.LOOKUP_TABLE_S]
    #     theta_prev = lookup_table_row_prev[:, WorldLineArcLengthParam.LOOKUP_TABLE_THETA]
    #     x_prev = lookup_table_row_prev[:, WorldLineArcLengthParam.LOOKUP_TABLE_X]
    #     y_prev = lookup_table_row_prev[:, WorldLineArcLengthParam.LOOKUP_TABLE_Y]
    #
    #     delta_s = s - s_prev
    #
    #     x_new = x_prev + np.multiply(delta_s, np.cos(theta_prev))
    #     y_new = y_prev + np.multiply(delta_s, np.sin(theta_prev))
    #
    #     return np.hstack((x_new[np.newaxis].T, y_new[np.newaxis].T))

    def computePointsGivenS(self, s):
        return self.computeRowGivenS(s)[:, [WorldLineArcLengthParam.LOOKUP_TABLE_X, WorldLineArcLengthParam.LOOKUP_TABLE_Y]]

    def _computeIncrementFromPoint(self, starting_xy, starting_s, delta_s):
        # theta = self.getModel()(starting_s + delta_s)
        # theta = np.clip(theta, -pi, pi)
        theta = self.applyModel(starting_s + delta_s)
        delta_x = delta_s * cos(theta)
        delta_y = delta_s * sin(theta)
        x = starting_xy[0] + delta_x
        y = starting_xy[1] + delta_y
        return np.array([x,y])

    # def computeRowGivenS(self, s):
    #     xy = self.computePointsGivenS(s)
    #     theta = self.applyModel(s)
    #     return np.hstack((s[np.newaxis].T, theta[np.newaxis].T, xy))


    def computeRowGivenS(self, s):
        s = np.asarray(s)

        # retrieve starting row from lookup table (lookup x <= query x)
        # first_i_arr = np.array([np.argmax(
        #     self.lookup_table[self.lookup_table_i_sort_s, WorldLineArcLengthParam.LOOKUP_TABLE_S] > s_i) - 1
        #                         for s_i in s])  # index just before I expect to find my s_i # FIXME am i sure?? shouldn't it be argmax(argwhere(...))??
        # feasible_i_arr = np.logical_and(first_i_arr >= 0, first_i_arr < self.lookup_table.shape[0] - 1)  # for each s point requested, bool says if it's within allowed range

        def findCloserSABelowValue(s_pt):
            # smaller_i_than_s_pt = np.argwhere(self.lookup_table[self.lookup_table_i_sort_s, WorldLineArcLengthParam.LOOKUP_TABLE_S] <= s_pt)
            # first_i = np.min(smaller_i_than_s_pt) if smaller_i_than_s_pt.size > 0 else -1
            larger_i_than_s_pt = np.where(self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_S] > s_pt)[0]
            if larger_i_than_s_pt.size > 0:
                first_i = np.min(larger_i_than_s_pt) - 1
            elif s_pt == (self.lookup_table[:, WorldLineArcLengthParam.LOOKUP_TABLE_S])[-1]:
                # particular case: if s_pt is exactly last element in lookup table, larger_i_than_s_pt is empty but the point can still be computed
                first_i = self.lookup_table.shape[0] - 1
            else:
                first_i = -1
            return first_i

        rows = np.empty((s.shape[0], 4))
        for id in range(s.shape[0]):
            s_pt = s[id]
            # first_i = np.argmax(self.lookup_table[self.lookup_table_i_sort_s, WorldLineArcLengthParam.LOOKUP_TABLE_S] > s_pt) - 1
            first_i = findCloserSABelowValue(s_pt)
            # feasible_i = (first_i >= 0 and first_i < self.lookup_table.shape[0] - 1)
            feasible_i = (first_i >= 0 and first_i < self.lookup_table.shape[0])
            if not feasible_i:
                extension_succeeded = self._extendLookupTableInS(s_pt)
                if not extension_succeeded:
                    # Point requested is not reachable integrating the model or it is too far to be reached
                    theta = np.nan
                    x = np.nan
                    y = np.nan
                    rows[id] = [s_pt, theta, x, y]
                    #     raise Line.LineException("Line model in (s,theta) is not sound when computing line value for x={}m. This often happens when fitting a model with the provided points "
                    # "generates a spiral or an infeasible road line configuration.".format(x_pt))
                else:
                    first_i = findCloserSABelowValue(s_pt)
                    lookup_table_row = self.lookup_table[first_i, :]
                    row = self._findPreciseRowGivenS(s_pt, lookup_table_row)
                    rows[id] = row
            else:
                lookup_table_row = self.lookup_table[first_i, :]
                row = self._findPreciseRowGivenS(s_pt, lookup_table_row)
                rows[id] = row

        return rows

    def computeRowGivenX(self, world_x_pts, tol=1e-3):
        # retrieve starting row from lookup table (lookup x <= query x) for each x point
        # first_i_arr = np.array([np.argmax(
        #     self.lookup_table[self.lookup_table_i_sort_x, WorldLineArcLengthParam.LOOKUP_TABLE_X] > world_x_pt) - 1
        #                         for world_x_pt in world_x_pts])  # index just before I expect to find my x
        # feasible_i_arr = np.logical_and(first_i_arr >= 0, first_i_arr < self.lookup_table.shape[
        #     0] - 1) # for each x point requested, bool says if it's within allowed range

        def findCloserXBelowAValue(x_pt):
            larger_i_than_s_pt = np.argwhere(self.lookup_table[self.lookup_table_i_sort_x, WorldLineArcLengthParam.LOOKUP_TABLE_X] > x_pt)  # FIXME lookup_table_i_sort_x?
            if larger_i_than_s_pt.size > 0:
                first_i = np.min(larger_i_than_s_pt) - 1
            elif x_pt == (self.lookup_table[self.lookup_table_i_sort_x, WorldLineArcLengthParam.LOOKUP_TABLE_X])[-1]:
                # particular case: if s_pt is exactly last element in lookup table, larger_i_than_s_pt is empty but the point can still be computed
                first_i = self.lookup_table.shape[0] - 1
            else:
                first_i = -1
            return first_i

        rows = np.empty((world_x_pts.shape[0], 4))
        for id in range(world_x_pts.shape[0]):
            x_pt = world_x_pts[id]
            # first_i = first_i_arr[id]
            # feasible_i = feasible_i_arr[id]

            first_i = findCloserXBelowAValue(x_pt)
            # feasible_i = (first_i >= 0 and first_i < self.lookup_table.shape[0] - 1)
            feasible_i = (first_i >= 0 and first_i < self.lookup_table.shape[0])

            if not feasible_i:
                extension_succeeded = self._extendLookupTableInX(x_pt)
                if not extension_succeeded:
                    # Point requested is not reachable integrating the model or it is too far to be reached
                    s = np.nan
                    theta = np.nan
                    y = np.nan
                    rows[id] = [s, theta, x_pt, y]
                    #     raise Line.LineException("Line model in (s,theta) is not sound when computing line value for x={}m. This often happens when fitting a model with the provided points "
                    # "generates a spiral or an infeasible road line configuration.".format(x_pt))
                else:
                    # first_i = np.argmax(self.lookup_table[self.lookup_table_i_sort_x, WorldLineArcLengthParam.LOOKUP_TABLE_X] > x_pt) - 1
                    first_i = findCloserXBelowAValue(x_pt)
                    lookup_table_row = (self.lookup_table[self.lookup_table_i_sort_x, :])[first_i, :]
                    row = self._findApproxRowGivenX(x_pt, lookup_table_row, tol=tol)
                    rows[id] = row
            else:
                lookup_table_row = (self.lookup_table[self.lookup_table_i_sort_x, :])[first_i, :]
                row = self._findApproxRowGivenX(x_pt, lookup_table_row, tol=tol)
                rows[id] = row

        return rows

    def _findPreciseRowGivenS(self, s_pt, lookup_table_row):
        """
        #Start from lookup table closest (lower) row, then converge up to wanted x with required precision
        """
        start_s = lookup_table_row[WorldLineArcLengthParam.LOOKUP_TABLE_S]
        start_x = lookup_table_row[WorldLineArcLengthParam.LOOKUP_TABLE_X]
        start_y = lookup_table_row[WorldLineArcLengthParam.LOOKUP_TABLE_Y]
        cur_xy = np.array([start_x, start_y])
        cur_s = start_s
        delta_s = s_pt  - cur_s
        cur_xy = self._computeIncrementFromPoint(cur_xy, cur_s, delta_s) # np.array([0, delta_s]), origin_pt_xy=cur_xy)
        cur_s += delta_s
        row = np.array([cur_s, self.applyModel(cur_s), cur_xy[0], cur_xy[1]])
        return row

    def _findApproxRowGivenX(self, x_pt, lookup_table_row, tol=1e-3):
        """
        Start from lookup table closest (lower) row, then converge up to wanted x with required precision
        """
        start_s = lookup_table_row[WorldLineArcLengthParam.LOOKUP_TABLE_S]
        start_x = lookup_table_row[WorldLineArcLengthParam.LOOKUP_TABLE_X]
        start_y = lookup_table_row[WorldLineArcLengthParam.LOOKUP_TABLE_Y]
        cur_xy = np.array([start_x, start_y])
        cur_s = start_s
        while x_pt - cur_xy[0] >= tol:
            prev_xy = cur_xy
            delta_s = x_pt - cur_xy[0]
            cur_xy = self._computeIncrementFromPoint(cur_xy, cur_s, delta_s) # np.array([0, delta_s]), origin_pt_xy=cur_xy)
            cur_s += delta_s
        row = np.array([cur_s, self.applyModel(cur_s), cur_xy[0], cur_xy[1]])
        return row

    def computeHeading(self, x_pt=None):
        if x_pt is None:
            s_heading = 0
        else:
            s_heading = self.computeRowGivenX(np.array([x_pt]))[0, WorldLineArcLengthParam.LOOKUP_TABLE_S]
        return self.getModel()([s_heading])[0]

    def computeCurvatureFunction(self):
        """
        Computes curvature model (derivative of line model theta=f(s))
        :return: poly1d representing the curvature function (in s)
        """
        return np.poly1d.deriv(self.getModel())

    def computeCurvatureAtXPoint(self, x_pt=None):
        if x_pt is None:
            s_pt = 0
        else:
            s_pt = self.computeRowGivenX(np.array([x_pt]))[0, WorldLineArcLengthParam.LOOKUP_TABLE_S]
        return self.computeCurvatureAtSPoint(s_pt)

    def computeCurvatureAtSPoint(self, s_pt=None):
        signed_curvature = self.computeCurvatureFunction()(s_pt)
        return np.abs(signed_curvature), np.sign(signed_curvature)

    def computeCurvature(self):
        return self.computeCurvatureAtXPoint(x_pt=None)

    def computeRadiusOfCurvatureAtXPoint(self, x_pt):
        """
        Computes radius of curvature and its direction
        :return: radius_of_curvature (in meters), curvature_side (+1 left, -1 right)
        """
        curvature, curvature_side = self.computeCurvatureAtXPoint(x_pt)
        radius_of_curvature = 1.0 / curvature
        return radius_of_curvature, curvature_side

    def computeRadiusOfCurvatureAtSPoint(self, s_pt):
        """
        Computes radius of curvature and its direction
        :return: radius_of_curvature (in meters), curvature_side (+1 left, -1 right)
        """
        curvature, curvature_side = self.computeCurvatureAtSPoint(s_pt)
        radius_of_curvature = 1.0 / curvature
        return radius_of_curvature, curvature_side

    def computeRadiusOfCurvature(self):
        return self.computeRadiusOfCurvatureAtXPoint(x_pt=None)

    def _computeOrdinateGivenAbscissae(self, abs_pts):
        raise NotImplementedError("Cannot use WorldLineArcLengthParam with x,y coordinate. Use s,theta instead.")

    def _computeThetaGivenS(self, s):
        return self._model(s)

    def computeConfidenceInterval(self, abs_pts, confidence=0.95):
        raise NotImplementedError("Confidence interval not implemented for WorldLineArcLengthParam")

class WorldLineArcLengthParamUpdatable(WorldLineArcLengthParam):

    def __init__(self, world_line_arc_length_param):
        self.apply_model = world_line_arc_length_param.apply_model
        self._model = world_line_arc_length_param.getModel()
        self.origin = world_line_arc_length_param.getOrigin()
        super(WorldLineArcLengthParamUpdatable, self).__init__(
            world_line_arc_length_param.bev_obj,
            world_line_arc_length_param.getPoints(),
            model_type=world_line_arc_length_param.model_type,
            model_order=world_line_arc_length_param.model_order,
            id=world_line_arc_length_param.id)
        self.last_measured_line = world_line_arc_length_param

    def _fitModel(self):
        # pass
        return self._model

    def updateModel(self, new_model, new_origin, last_measured_line=None, cutoff_bounds=None):
        self.origin = new_origin
        self._model = np.poly1d(new_model, variable='s')

        if "cutoff" in self.getModelType() and cutoff_bounds is not None:
            #assert cutoff_bounds is not None, "WorldLineArcLengthParamUpdatable.updateModel(): if model type requires cutoff, cutoff bounds must be set"
            left_cutoff_s = cutoff_bounds[0] #s_sorted[0]
            left_cutoff_theta = self._model(left_cutoff_s)  # theta_sorted[0]
            right_cutoff_s = cutoff_bounds[1]
            right_cutoff_theta = self._model(right_cutoff_s)
            poly_model = np.poly1d(np.copy(self._model.coeffs), variable='s')
            # self.apply_model = self._mollifyModel(poly_model, left_cutoff_s, left_cutoff_theta, right_cutoff_s, right_cutoff_theta)
            self.apply_model = arc_length_model_fcn = lambda s: np.piecewise(s,
                                                                            [np.asarray(s) <= left_cutoff_s,
                                                                             np.bitwise_and(np.asarray(s) > left_cutoff_s, np.asarray(s) <= right_cutoff_s),
                                                                             np.asarray(s) > right_cutoff_s],
                                                                            [left_cutoff_theta,
                                                                             lambda s: poly_model(s),
                                                                             right_cutoff_theta])
        else:
            self.apply_model = np.poly1d(new_model, variable='s')

        self._initLookupTable(delta_pts=self.delta_pts, min_n_pts=self.min_n_pts)
        self.last_measured_line = last_measured_line

    def getFitS(self):
        if self.last_measured_line is not None:
            return self.last_measured_line.getFitS()
        # else
        return None

    def getFitTheta(self):
        if self.last_measured_line is not None:
            return self.last_measured_line.getFitTheta()
        # else
        return None

class WorldLineArcLengthParamUpdatableMockup(WorldLineArcLengthParamUpdatable):
    def __init__(self, bev_obj, model_coeff, model_fcn, origin):
        self.apply_model = model_fcn
        self._model = model_coeff
        self.origin = origin
        super(WorldLineArcLengthParamUpdatable, self).__init__(
            bev_obj,
            [],
            model_order=len(model_coeff))

class WorldLineArcLengthParamCenterline(WorldLineArcLengthParam):

    LINES_ORIGIN_PROJECTION_MODE_NONE = 0
    LINES_ORIGIN_PROJECTION_MODE_SAME_X = 1
    LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_LINES = 2
    LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE = 3
    LINES_ORIGIN_PROJECTION_MODE_FROM_CURVATURE = 4
    LINES_ORIGIN_PROJECTION_MODE_FROM_CURVATURE_ORIGIN_IN_CENTER = 5
    LINES_ORIGIN_PROJECTION_MODE_FROM_EKF_CENTERLINE_PARAMS = 6

    CENTERLINE_ORIGIN_IN_VEHICLE_REF_PT = 100
    CENTERLINE_ORIGIN_IN_ROAD_CENTER = 101

    def __init__(self, lines, model_type="arc_length_parametrization", model_order=2, model_lsq_reg=None,
                 min_n_pts_for_fitting=None, min_m_range_for_fitting=None,
                 id=None,
                 fixed_origin_x=None, vehicle_reference_pt=None,
                 origin_in=CENTERLINE_ORIGIN_IN_VEHICLE_REF_PT,
                 lines_origin_projection_mode=LINES_ORIGIN_PROJECTION_MODE_NONE,
                 tracked_centerline_for_lines_origins_computation=None,
                 intercept_from_ekf = True,
                 # ekf_centerline_params_obj=None
                 ):
        """
        :param lines: lateral road lines used to find the centerline
        :param model_type:
        :param model_order:
        :param id:
        :param fixed_origin_x: if set, line origin is moved to match this x; used in origin tracking ot have a common reference for tracking
        :param vehicle_reference_pt: point from which heading and lateral offset is computed (usually center of mass)
        :param origin_in: whether the line should have origin in vehicle_reference_pt or in the center of the road
        :param lines_origin_projection_mode:
        :param tracked_centerline_for_lines_origins_computation: tracked centerline at t-1, if available
        :param ekf_centerline_params_state: state of the ekf optionally used for tracking heading and road center position
        """

        # assert origin is not None or fixed_origin_x is not None, "WorldLineArcLengthParamCenterline: either an origin must be passed or fixed_origin_x must be set."
        self.lines = lines
        self.lines_origin_projection_mode = lines_origin_projection_mode
        self.origin_in = origin_in
        self.vehicle_reference_pt = vehicle_reference_pt
        self.road_center = None
        self.tracked_centerline_for_lines_origins_computation = tracked_centerline_for_lines_origins_computation
        self.ekf_centerline_params_obj = \
            tracked_centerline_for_lines_origins_computation.ekf_centerline_params_obj \
                if self.tracked_centerline_for_lines_origins_computation is not None and self.lines_origin_projection_mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_EKF_CENTERLINE_PARAMS \
                else None
        self.intercept_from_ekf = intercept_from_ekf
        # self.origin = origin
        super(WorldLineArcLengthParamCenterline, self).__init__(lines[0].getBevObj(), [], model_type=model_type,
                                                                model_order=model_order, model_lsq_reg=model_lsq_reg,
                                                                min_n_pts_for_fitting=min_n_pts_for_fitting,
                                                                min_m_range_for_fitting=min_m_range_for_fitting,
                                                                id=id, fixed_origin_x=fixed_origin_x)
        # set line points limits as smaller x interval covered by both
        self.line_points_x_limit = (np.max([l.getPointsXLimits()[0] for l in self.lines]),
                                    np.min([l.getPointsXLimits()[1] for l in self.lines]))

    def _fitModel(self):
        t0 = time.time()
        self._computeSThetaPoints()
        t1 = time.time()

        # FIXME TEST!!!!!!!!!!!!! can remove if doesn't work
        if self.lines_origin_projection_mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_EKF_CENTERLINE_PARAMS \
                and self.ekf_centerline_params_obj is not None \
                and self.intercept_from_ekf:
            # correct heading (poly costant term): translate whole line up/down in s,theta to match ekf_theta as intercept
            ekf_rho, ekf_theta, ekf_w = tuple(self.ekf_centerline_params_obj.x.squeeze()) # predict not here

            # c = model.coeffs
            # c[-1] = ekf_theta
            # model = np.poly1d(c)

            model, model_fcn = self._fitModelInS(fixed_intercept=ekf_theta)

        else:
            model, model_fcn = self._fitModelInS()

        self.apply_model = model_fcn
        t2 = time.time()

        # self._computeNewOrigins()
        return model

    def _computeSThetaPoints(self):
        t0 = time.time()
        # align points to a common origin
        new_ss, new_origins = self._reprojectLinesOrigins(mode=self.lines_origin_projection_mode)

        t1 = time.time()

        self.lines_corresponding_s = new_ss
        self.lines_corresponding_origins_prev_t = new_origins

        self._fit_points_ordered = []
        # if self.model_type == "arc_length_parametrization":
        #     self.s = np.concatenate(tuple(new_ss))
        #     self.theta = np.concatenate(tuple([l.getFitTheta() for l in self.lines]))
        # else: # self.model_type == "arc_length_parametrization_avg":
        if self.model_type.startswith("arc_length_parametrization_avg"):
            # fit model to both new_ss of both lines --> could be any, even spline if easier (although shouldnt)
            new_ss_concat = np.concatenate(tuple(new_ss))
            new_ss_concat_sorted_i = np.argsort(new_ss_concat)
            new_ss_concat_sorted = new_ss_concat[new_ss_concat_sorted_i]

            projected_line_theta_in_new_ss_concat = []
            for i,l in enumerate(self.lines):
                new_s = new_ss[i]
                theta = l.getFitTheta()
                projected_line_s_fit = np.polyfit(new_s, theta, l.getModelOrder())
                # projected_line_s_fit = polyfit_reg(new_s, theta, l.getModelOrder(), reg=l.measured_line.lsq_reg)
                projected_line_s_model = np.poly1d(projected_line_s_fit)
                projected_line_theta_sorted = projected_line_s_model(new_ss_concat_sorted)
                projected_line_theta_in_new_ss_concat.append(projected_line_theta_sorted)

            projected_line_theta_sorted_in_new_ss_concat = np.array(projected_line_theta_in_new_ss_concat)
            new_ss_theta_sorted = np.mean(projected_line_theta_sorted_in_new_ss_concat, axis=0)
            self.s = new_ss_concat_sorted
            self.theta = new_ss_theta_sorted

        else:
            self.s = np.concatenate(tuple(new_ss))
            self.theta = np.concatenate(tuple([l.getFitTheta() for l in self.lines]))

        # find avg points wrt each fit points (new_ss)
            # use these new s,theta as s,theta to pass to _fitModelInS()

        # if self.origin_in == WorldLineArcLengthParamCenterline.CENTERLINE_ORIGIN_IN_ROAD_CENTER:
        #     # fit temporary line
        #     temp_model = self._fitModelInS()
        #     temp_origin = self.origin
        #     temp_line = WorldLineArcLengthParamUpdatableMockup(self.bev_obj, temp_model, temp_origin)
        #
        #     # find point on centerline for which its normal goes through the vehicle reference point
        #     centerline_intersection_pt_xy, centerline_intersection_pt_s = self._findLinePointOrthogonalToLineThroughCM(temp_line, self.vehicle_reference_pt, tol=1e-3)
        #
        #     # set road center
        #     self.road_center = centerline_intersection_pt_xy
        #
        #     # new centerline origin is intersection point
        #     self.origin = centerline_intersection_pt_xy
        #     self.s = self.s - centerline_intersection_pt_s # FIXME - NO GOOD! NEED TO TRANSLATE IN S TOO!! --> if translate in s here I need to fit again!.. can i do this before somehow? temp fit like i do in other occasions and move everything to computeSPoints before fit??
        #     # FIXME check if sign s -/+ delta_s is correct!
        #
        #     # TODO should compute new lateral lines origin - using "delta_s = R * delta_theta" trick, although they're not needed

        t2 = time.time()

        # fit temporary line
        temp_model, temp_model_fcn = self._fitModelInS()
        temp_origin = self.origin
        temp_line = WorldLineArcLengthParamUpdatableMockup(self.bev_obj, temp_model, temp_model_fcn, temp_origin)

        # if self.lines_origin_projection_mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_EKF_CENTERLINE_PARAMS \
        #         and self.ekf_centerline_params_obj is not None:
        #     ekf_rho, ekf_theta, ekf_w = tuple(self.ekf_centerline_params_obj.x.squeeze())
        #     ekf_road_center = np.array([
        #         self.vehicle_reference_pt[0] + ekf_w / 2 * ekf_rho * sin(ekf_theta),
        #         self.vehicle_reference_pt[1] - ekf_w / 2 * ekf_rho * cos(ekf_theta),
        #         0
        #     ])
        #
        #     # centerline predicted normal line
        #     centerline_normal_through_vehicle_ref_pt_points = np.vstack((self.vehicle_reference_pt, ekf_road_center))
        #
        #     # centerline new origin as intersection of predicted normal with temporary centerline
        #     search_bounds_delta = 10
        #     search_bounds = (0 - search_bounds_delta, 0 + search_bounds_delta)
        #     centerline_intersection_pt_xyz_tuple = self._findLineIntersectionWithStraightLine(temp_line, centerline_normal_through_vehicle_ref_pt_points[:, 0:2], search_bounds, tol=1e-3)
        #     if centerline_intersection_pt_xyz_tuple is not None:
        #         centerline_intersection_pt_xyz, centerline_intersection_pt_s = centerline_intersection_pt_xyz_tuple
        #     else:
        #         # raise Line.LineException("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of predicted normal line with temporary centerline.")
        #         print("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of predicted normal line with temporary centerline.")
        #         centerline_intersection_pt_xyz = temp_line.origin
        #         centerline_intersection_pt_s = 0
        # else:
        # find point on centerline for which its normal line goes through the vehicle reference point (+ find also normal line)
        centerline_intersection_pt_xyz, centerline_intersection_pt_s = self._findLinePointOrthogonalToLineThroughCM(temp_line, self.vehicle_reference_pt[0:2], tol=1e-3)
        centerline_normal_through_vehicle_ref_pt_points = np.vstack((self.vehicle_reference_pt, centerline_intersection_pt_xyz))

        # find intersection of normal lines with lateral lines
        search_bounds_delta = 10
        lateral_lines_new_origins = []
        lateral_lines_new_origins_s = []
        for i,l in enumerate(self.lines):
            search_bounds = (0-search_bounds_delta, 0+search_bounds_delta)

            # check where line origin is, plus maybe also where are pts

            lateral_lines_new_origin_tuple = self._findLineIntersectionWithStraightLine(l, centerline_normal_through_vehicle_ref_pt_points[:, 0:2], search_bounds, tol=1e-3)
            if lateral_lines_new_origin_tuple is not None:
                lateral_lines_new_origin, lateral_lines_new_origin_s = lateral_lines_new_origin_tuple
            else:
                # raise Line.LineException("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of normal line with line(id={}).".format(l.id))
                print("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of normal line with line(id={}).".format(l.id))
                lateral_lines_new_origin = self.lines_corresponding_origins_prev_t[i][0:2]
                lateral_lines_new_origin_s = 0
            lateral_lines_new_origins.append(lateral_lines_new_origin)
            lateral_lines_new_origins_s.append(lateral_lines_new_origin_s)
        lateral_lines_new_origins = np.array(lateral_lines_new_origins)
        lateral_lines_new_origins_s = np.array(lateral_lines_new_origins_s)

        if self.lines_origin_projection_mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_EKF_CENTERLINE_PARAMS \
                and self.ekf_centerline_params_obj is not None:
            if self.lines[0].isTrackingActive() and self.lines[1].isTrackingActive(): # if both tracked (and not one just estimated from the other and ekf_w)
                # update ekf with measurement Pl, Pr
                self.ekf_centerline_params_obj.upd(lateral_lines_new_origins[0,:], lateral_lines_new_origins[1,:])
                # predict new state
                self.ekf_centerline_params_obj.pred()
                pred_ekf_rho, pred_ekf_theta, pred_ekf_w = tuple(self.ekf_centerline_params_obj.x.squeeze())

                # set road center with filter estimate
                self.road_center = np.array([
                    self.vehicle_reference_pt[0] + pred_ekf_w / 2 * pred_ekf_rho * sin(pred_ekf_theta),
                    self.vehicle_reference_pt[1] - pred_ekf_w / 2 * pred_ekf_rho * cos(pred_ekf_theta),
                    0
                ])

                # recompute centerline normal line
                centerline_normal_through_vehicle_ref_pt_points = np.vstack((self.vehicle_reference_pt, self.road_center))

                # FIXME duplicated code?
                # recompute centerline origin s value (as intersection of new predicted normal with temporary centerline)
                search_bounds_delta = 10
                search_bounds = (0 - search_bounds_delta, 0 + search_bounds_delta)
                centerline_intersection_pt_xyz_tuple = self._findLineIntersectionWithStraightLine(temp_line, centerline_normal_through_vehicle_ref_pt_points[:, 0:2], search_bounds, tol=1e-3)
                if centerline_intersection_pt_xyz_tuple is not None:
                    _, centerline_intersection_pt_s = centerline_intersection_pt_xyz_tuple
                else:
                    # raise Line.LineException("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of predicted normal line with temporary centerline.")
                    print("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of predicted normal line with temporary centerline.")
                    centerline_intersection_pt_s = 0

                # FIXME duplicated code?
                # recompute intersection of normal line with lateral lines
                search_bounds_delta = 10
                measured_lateral_lines_new_origins = lateral_lines_new_origins # save measured value
                measured_lateral_lines_new_origins_s = lateral_lines_new_origins_s # save measured value
                lateral_lines_new_origins = []
                lateral_lines_new_origins_s = []
                for i, l in enumerate(self.lines):
                    search_bounds = (0 - search_bounds_delta, 0 + search_bounds_delta)
                    lateral_lines_new_origin_tuple = self._findLineIntersectionWithStraightLine(l, centerline_normal_through_vehicle_ref_pt_points[:, 0:2], search_bounds, tol=1e-3)
                    if lateral_lines_new_origin_tuple is not None:
                        lateral_lines_new_origin, lateral_lines_new_origin_s = lateral_lines_new_origin_tuple
                    else:
                        # raise Line.LineException("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of normal line with line(id={}).".format(l.id))
                        print("WorldLineArcLengthParamCenterline - lines origin final reprojection: couldn't find intersection of normal line with line(id={}).".format(l.id))
                        # use current measurement
                        lateral_lines_new_origin = measured_lateral_lines_new_origins[i]
                        lateral_lines_new_origin_s = measured_lateral_lines_new_origins_s[i]
                    lateral_lines_new_origins.append(lateral_lines_new_origin)
                    lateral_lines_new_origins_s.append(lateral_lines_new_origin_s)
                lateral_lines_new_origins = np.array(lateral_lines_new_origins)
                lateral_lines_new_origins_s = np.array(lateral_lines_new_origins_s)
            else:
                if not np.isnan(lateral_lines_new_origins).any():
                    self.road_center = np.insert(np.mean(lateral_lines_new_origins, axis=0), 2, (0))
        else:
            # set road center in middle of 2 new lines origins (intersections with normal line)
            # self.road_center = centerline_intersection_pt_xyz
            if not np.isnan(lateral_lines_new_origins).any():
                self.road_center = np.insert(np.mean(lateral_lines_new_origins, axis=0), 2, (0))
            # else:
            # leave road_center same as in previous frame
            # self.road_center(t) = self.road_center(t-1)

        # update new_origins as intersections with normal line: add z=0 coordinate
        new_origins = np.insert(lateral_lines_new_origins, 2, (0), axis=1)

        # translate new_ss to match new origin of lateral line
        for i,new_s in enumerate(new_ss):
            new_s = new_s - lateral_lines_new_origins_s[i]
            new_ss[i] = new_s # FIXME does it work like this? --> it should cause shape is the same, just need to change where origin is on that line

        if self.origin_in == WorldLineArcLengthParamCenterline.CENTERLINE_ORIGIN_IN_ROAD_CENTER and self.road_center is not None:
            # new centerline origin is intersection point
            self.origin = self.road_center
            # s should be translated to match new origin
            self.s = self.s - centerline_intersection_pt_s # translate the line along the normal line, so s origin is where the line intercepted normal line before, and origin is in center point of normal line
            # solved FIX-ME - NO GOOD! NEED TO TRANSLATE IN S TOO!! --> if translate in s here I need to fit again!.. can i do this before somehow? temp fit like i do in other occasions and move everything to computeSPoints before fit??
            # FIXME check if sign s -/+ delta_s is correct!
        else: # self.origin_in == WorldLineArcLengthParamCenterline.CENTERLINE_ORIGIN_IN_VEHICLE_REF_PT or self.road_center is None:
            self.origin = self.vehicle_reference_pt
            self.s = self.s - centerline_intersection_pt_s # FIXME right???

            # TODO should compute new lateral lines origin - using "delta_s = R * delta_theta" trick, although they're not needed
        self.lines_corresponding_origins = new_origins  # FIXME moved here since I removed the new origins fixed

        t3 = time.time()

        # if self.fixed_origin_x is not None:
        #     self._imposeFixedOrigin()

    # def _fitModelInS(self):
    #     if self.model_type == "arc_length_parametrization_avg":
    #         return self._fitModelInS_arc_length_parametrization_avg()
    #     elif self.model_type == "arc_length_parametrization":
    #         return super(WorldLineArcLengthParamCenterline, self)._fitModelInS()
    #     else:
    #         raise NotImplemented("WorldLineArcLengthParamCenterline: model type {} not implemented.".format(self.model_type))
    #
    # def _fitModelInS_arc_length_parametrization_avg(self):
    #

    # def _computeNewOrigins(self):
    #     self.lines_corresponding_origins = self._reprojectLinesNewOrigins(mode=self.lines_origin_projection_mode)

    def _reprojectLinesOrigins(self, mode=LINES_ORIGIN_PROJECTION_MODE_NONE):
        if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_NONE:
            self.origin = np.array([self.fixed_origin_x, 0, 0])
            return [l.getFitS() for l in self.lines], [l.getOrigin() for l in self.lines]
        if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_SAME_X:
            return self._reprojectLinesOriginsSameX()
        if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_LINES:
            # raise NotImplementedError("WorldLineArcLengthParamCenterline: lines origin projection mode ORTHOGONAL_FROM_LINES not implemented yet.")
            return self._reprojectLinesOriginsOrthogonalFromLines()
        if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE:
            # raise NotImplementedError("WorldLineArcLengthParamCenterline: lines origin projection mode ORTHOGONAL_FROM_CENTERLINE not implemented yet.")
            try:
                return self._reprojectLinesOriginsOrthogonalFromCenterline()
            except (NotImplementedError, Line.LineException) as e:
                print("WorldLineArcLengthParamCenterline._reprojectLinesOrigins caught exception: {}".format(e))
                print("Fallback on lines origin projection mode: SAME_X")
                return self._reprojectLinesOrigins(mode=WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_SAME_X)
        if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_CURVATURE:
            try:
                return self._reprojectLinesOriginsOrthogonalFromCurvature()
            except (NotImplementedError, Line.LineException) as e:
                print("WorldLineArcLengthParamCenterline._reprojectLinesOrigins caught exception: {}".format(e))
                print("Fallback on lines origin projection mode: ORTHOGONAL_FROM_CENTERLINE")
                return self._reprojectLinesOrigins(mode=WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE)
        if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_EKF_CENTERLINE_PARAMS:
            try:
                return self._reprojectLinesOriginsOrthogonalFromEkfCenterlineParams()
            except (NotImplementedError, Line.LineException) as e:
                print("WorldLineArcLengthParamCenterline._reprojectLinesOrigins caught exception: {}".format(e))
                print("Fallback on lines origin projection mode: ORTHOGONAL_FROM_CENTERLINE")
                return self._reprojectLinesOrigins(mode=WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE)
        # if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_CURVATURE_CHANGING:
        #     try:
        #         return self._reprojectLinesOriginsOrthogonalFromCurvatureChanging()
        #     except (NotImplementedError, Line.LineException) as e:
        #         print("WorldLineArcLengthParamCenterline._reprojectLinesOrigins caught exception: {}".format(e))
        #         print("Fallback on lines origin projection mode: ORTHOGONAL_FROM_CENTERLINE")
        #         # TODO check if this fallback is ok
        #         return self._reprojectLinesOrigins(mode=WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE)
        # else
        raise NotImplementedError("WorldLineArcLengthParamCenterline: invalid lines_origin_projection_mode: {}".format(self.lines_origin_projection_mode))

    # def _reprojectLinesNewOrigins(self, mode=LINES_ORIGIN_PROJECTION_MODE_NONE):
    #     if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_NONE:
    #         self.origin = np.array([self.fixed_origin_x, 0, 0])
    #         return self.lines_corresponding_origins_prev_t
    #     if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_SAME_X:
    #         return self.lines_corresponding_origins_prev_t
    #     if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_LINES:
    #         return self.lines_corresponding_origins_prev_t
    #     if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE:
    #         # raise NotImplementedError("WorldLineArcLengthParamCenterline: lines origin projection mode ORTHOGONAL_FROM_CENTERLINE not implemented yet.")
    #         # try:
    #         #     return self._reprojectLinesOriginsOrthogonalFromCenterline()
    #         # except (NotImplementedError, Line.LineException) as e:
    #         #     print("WorldLineArcLengthParamCenterline._reprojectLinesOrigins caught exception: {}".format(e))
    #         #     print("Fallback on lines origin projection mode: SAME_X")
    #         #     return self._reprojectLinesOrigins(mode=WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_SAME_X)
    #         # TODO
    #         pass
    #     if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_CURVATURE:
    #         try:
    #             return self._reprojectLinesNewOriginsOrthogonalFromCurvature()
    #         except (NotImplementedError, Line.LineException) as e:
    #             print("WorldLineArcLengthParamCenterline._reprojectLinesNewOrigins caught exception: {}".format(e))
    #             print("Fallback on lines origin projection mode: ORTHOGONAL_FROM_CENTERLINE")
    #             return self._reprojectLinesOrigins(mode=WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE)
    #     # if mode == WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_FROM_CURVATURE_CHANGING:
    #     #     try:
    #     #         return self._reprojectLinesOriginsOrthogonalFromCurvatureChanging()
    #     #     except (NotImplementedError, Line.LineException) as e:
    #     #         print("WorldLineArcLengthParamCenterline._reprojectLinesOrigins caught exception: {}".format(e))
    #     #         print("Fallback on lines origin projection mode: ORTHOGONAL_FROM_CENTERLINE")
    #     #         # TODO check if this fallback is ok
    #     #         return self._reprojectLinesOrigins(mode=WorldLineArcLengthParamCenterline.LINES_ORIGIN_PROJECTION_MODE_ORTHOGONAL_FROM_CENTERLINE)
    #     # else
    #     raise NotImplementedError("WorldLineArcLengthParamCenterline: invalid lines_origin_projection_mode: {}".format(self.lines_origin_projection_mode))

    def _reprojectLinesOriginsSameX(self):
        default_common_origin_x = np.min([l.getPointsXLimits()[0] for l in self.lines])  # minimum origin of lines
        if self.fixed_origin_x is not None:
            common_origin_x = self.fixed_origin_x
        else:
            common_origin_x = default_common_origin_x
        translation_results = [l._translatedPointsForNewOriginX(common_origin_x) for l in self.lines]
        if np.any([t is None for t in translation_results]): # if couldn't find common origin
            if self.fixed_origin_x is not None:
                # fallback on default
                common_origin_x = default_common_origin_x
                translation_results = [l._translatedPointsForNewOriginX(common_origin_x) for l in self.lines]
                if np.any([t is None for t in translation_results]): # if couldn't find common origin
                    raise Line.LineException("WorldLineArcLengthParamCenterline: couldn't find common origin.")
            else:
                raise Line.LineException("WorldLineArcLengthParamCenterline: couldn't find common origin.")
        if self.origin_in == WorldLineArcLengthParamCenterline.CENTERLINE_ORIGIN_IN_ROAD_CENTER:
            origin_y = np.mean([t[1][1] for t in translation_results])
        else:
            origin_y = 0
        self.origin = np.array([common_origin_x, origin_y, 0])
        new_ss, new_origins = [t[0] for t in translation_results], [t[1] for t in translation_results]
        return new_ss, new_origins

    def _reprojectLinesOriginsOrthogonalFromLines(self):
        assert self.fixed_origin_x is not None, "WorldLineArcLengthParamCenterline: fixed_origin_x param must be set in lines origin projection mode ORTHOGONAL_FROM_LINES."

        # self.origin = np.array([self.fixed_origin_x, 0, 0]) # workaround, to allow _findLinePointOrthogonalToLineThroughCM
        cm = np.array([self.fixed_origin_x, 0, 0])

        new_ss = []
        new_origins = []
        for l in self.lines:
            # compute best origin projection to each line
            best_origin_pts, best_origin_s = self._findLinePointOrthogonalToLineThroughCM(l, cm)
            # store new origin x,y
            new_origins.append(best_origin_pts)
            # translate points to s of new origin, then store
            translation_delta_s = best_origin_s
            new_s = l.getFitS() - translation_delta_s
            new_ss.append(new_s)

        if self.origin_in == WorldLineArcLengthParamCenterline.CENTERLINE_ORIGIN_IN_ROAD_CENTER:
            origin_y = np.mean(np.array(new_origins)[:,1])
        else:
            origin_y = 0
        self.origin = np.array([self.fixed_origin_x, origin_y, 0])

        return new_ss, new_origins

    def _findLinePointOrthogonalToLineThroughCM(self, l, origin, tol=1e-3):
        # TODO estimate interval for search in lookup table based on previous heading if available;
        #  this way I don't risk to have an orthogonal line far away coming by chance closer to origin than the actual
        #  wanted line point

        # coarse-grained location using lookup table
        lookup_table_pts_xy = l.getLookupTable()[:, WorldLineArcLengthParam.LOOKUP_TABLE_X:(WorldLineArcLengthParam.LOOKUP_TABLE_Y+1)]
        lookup_table_pts_s = l.getLookupTable()[:, WorldLineArcLengthParam.LOOKUP_TABLE_S]
        lookup_table_pts_theta = l.getLookupTable()[:, WorldLineArcLengthParam.LOOKUP_TABLE_THETA]
        lookup_table_pts_xy_h = np.insert(lookup_table_pts_xy, 2, (1), axis=1) # [[x,y,1], ...]
        lookup_table_pts_xy_h_plus_delta_orthog = lookup_table_pts_xy_h + np.vstack((np.cos(lookup_table_pts_theta + pi/2), np.sin(lookup_table_pts_theta + pi/2), np.zeros_like(lookup_table_pts_theta))).T
        # ort_lines_in_lookup_table_points = np.cross(lookup_table_pts_xy_h, lookup_table_pts_xy_h_plus_delta_orthog, axis=1)
        distance_lines_from_origin = np.abs(distance_point_from_line_through_points(origin[0:2], lookup_table_pts_xy_h[:,0:2], lookup_table_pts_xy_h_plus_delta_orthog[:,0:2]))
        ort_point_i = np.argmin(distance_lines_from_origin)
        ort_point = lookup_table_pts_xy[ort_point_i, :]
        ort_point_line_dist_from_origin = distance_lines_from_origin[ort_point_i]

        # fine-grained refinement
        ort_point_s = lookup_table_pts_s[ort_point_i]
        ort_point_theta = lookup_table_pts_theta[ort_point_i]
        # while ort_point_line_dist_from_origin > tol:
        #     pass
        delta_search_bounds = 1
        search_bounds = (ort_point_s-delta_search_bounds, ort_point_s+delta_search_bounds)
        ort_point_refined_tuple = self._findLinePointOrthogonalToLineThroughCM_refinement(l, origin, bounds=search_bounds, tol=tol)

        if ort_point_refined_tuple is not None:
            ret_ort_point, ret_ort_point_s = ort_point_refined_tuple
        else: # couldn't perform refinement
            ret_ort_point, ret_ort_point_s = ort_point, ort_point_s

        # add z=0 to ort_point, coordinate not included above for simplicity
        ret_ort_point = np.insert(ret_ort_point, 2, (0))
        return ret_ort_point, ret_ort_point_s

    def _findLinePointOrthogonalToLineThroughCM_refinement(self, l, origin, bounds, tol=1e-3):

        def obj(s):
            row = l.computeRowGivenS(np.array([s]))[0]
            # s = row[WorldLineArcLengthParam.LOOKUP_TABLE_S]
            theta = row[WorldLineArcLengthParam.LOOKUP_TABLE_THETA]
            line_pt1 = np.array([row[WorldLineArcLengthParam.LOOKUP_TABLE_X:(WorldLineArcLengthParam.LOOKUP_TABLE_Y+1)]])
            line_pt2 = line_pt1 + np.array([cos(theta - pi / 2), sin(theta - pi / 2)])
            dist = distance_point_from_line_through_points(origin[0:2], line_pt1, line_pt2)[0]
            return dist*dist # square distance, so more differentiable

        # ort_point_s = scopt.minimize_scalar(obj) #, bounds=(-3, -1), method='bounded')
        opt_res = scopt.minimize_scalar(obj, bracket=bounds, method='brent', tol=tol)
        if not opt_res.success:
            return None

        ort_point_s = opt_res.x
        ort_point = l.computePointsGivenS([ort_point_s])[0,:]

        return ort_point, ort_point_s

    def _findLineIntersectionWithStraightLine(self, l, straight_line_points, bounds, tol=1e-3):

        def obj(s):
            row = l.computeRowGivenS(np.array([s]))[0]
            # s = row[WorldLineArcLengthParam.LOOKUP_TABLE_S]
            point = row[WorldLineArcLengthParam.LOOKUP_TABLE_X:WorldLineArcLengthParam.LOOKUP_TABLE_Y+1] # row[x,y]
            if np.isnan(point).any():
                return +np.inf
            line_pt1 = straight_line_points[0,:]
            line_pt2 = straight_line_points[1,:]
            dist = distance_point_from_line_through_points(point, line_pt1, line_pt2)
            return dist*dist # square distance, so more differentiable

        # opt_res = scopt.minimize_scalar(obj, bracket=bounds, method='brent', tol=tol)
        opt_res = scopt.minimize_scalar(obj, bounds=bounds, method='bounded', tol=tol)

        # FIXME check opt_res for extra val indicating why it stops!!!!!!!!

        if not opt_res.success or np.isinf(opt_res.fun):
            return None

        ort_point_s = opt_res.x
        ort_point = l.computePointsGivenS([ort_point_s])[0,:]

        return ort_point, ort_point_s

    def _reprojectLinesOriginsOrthogonalFromCenterline(self):
        if self.tracked_centerline_for_lines_origins_computation is None or not self.tracked_centerline_for_lines_origins_computation.isTrackingActive():
            raise NotImplementedError("WorldLineArcLengthParamCenterline: lines origin projection mode "
                                  "ORTHOGONAL_FROM_CENTERLINE not available with tracking disabled.")

        assert self.fixed_origin_x is not None, "WorldLineArcLengthParamCenterline: fixed_origin_x param must be set in lines origin projection mode ORTHOGONAL_FROM_CENTERLINE."

        # self.origin = np.array([self.fixed_origin_x, 0, 0])
        cm = np.array([self.fixed_origin_x, 0, 0])

        # compute tracked heading at this time (without new measurements, so basically it's heading at previous step)
        centerline_heading = self.tracked_centerline_for_lines_origins_computation.computeHeading(cm[0])

        if np.isnan(centerline_heading):
            raise Line.LineException("WorldLineArcLengthParamCenterline - lines origin projection mode ORTHOGONAL_FROM_CENTERLINE: couldn't compute heading.")

        new_ss = []
        new_origins = []
        for l in self.lines:
            # compute best origin projection to each line
            best_origin_tuple = self._findLineIntersectionWithLineThroughCM(l, cm, centerline_heading)
            if best_origin_tuple is None:
                raise Line.LineException("WorldLineArcLengthParamCenterline - lines origin projection mode ORTHOGONAL_FROM_CENTERLINE: couldn't find intersection with line(id={}).".format(l.id))
            best_origin_pts, best_origin_s = best_origin_tuple
            # store new origin x,y
            new_origins.append(best_origin_pts)
            # translate points to s of new origin, then store
            translation_delta_s = best_origin_s
            new_s = l.getFitS() - translation_delta_s
            new_ss.append(new_s)

        if self.origin_in == WorldLineArcLengthParamCenterline.CENTERLINE_ORIGIN_IN_ROAD_CENTER:
            origin_y = np.mean(np.array(new_origins)[:,1])
        else:
            origin_y = 0
        self.origin = np.array([self.fixed_origin_x, origin_y, 0])

        return new_ss, new_origins

    def _findLineIntersectionWithLineThroughCM(self, l, origin, centerline_heading, tol=1e-3):
        normal_line_pt1 = np.asarray(origin)[0:2]
        normal_line_pt2 = normal_line_pt1 + np.array(
            [cos(centerline_heading - pi / 2), sin(centerline_heading - pi / 2)])

        def obj(s):
            """
            Squared distance from normal line through origin to point in l at arc length s
            """
            row = l.computeRowGivenS(np.array([s]))[0]
            # s = row[WorldLineArcLengthParam.LOOKUP_TABLE_S]
            theta = row[WorldLineArcLengthParam.LOOKUP_TABLE_THETA]
            l_pt = row[WorldLineArcLengthParam.LOOKUP_TABLE_X:(WorldLineArcLengthParam.LOOKUP_TABLE_Y + 1)]
            dist = distance_point_from_line_through_points(l_pt, normal_line_pt1, normal_line_pt2)
            return dist * dist  # square distance, so more differentiable

        bounds_x_width = 3
        bounds_x = np.array([origin[0] - bounds_x_width / 2.0, origin[0] + bounds_x_width / 2.0])
        bounds_s = l.computeRowGivenX(bounds_x, tol=tol)[:, WorldLineArcLengthParam.LOOKUP_TABLE_S]

        if np.isnan(bounds_s).any():
            # couldn't compute points at specified x
            return None

        # opt_res = scopt.minimize_scalar(obj, bracket=bounds, method='brent', tol=tol)
        opt_res = scopt.minimize_scalar(obj, bounds=bounds_s, method='bounded', tol=tol)
        if not opt_res.success or np.isnan(opt_res.x):
            return None

        ort_point_s = opt_res.x
        ort_point = l.computePointsGivenS([ort_point_s])[0, :]

        # add z=0 to ort_point, coordinate not included above for simplicity
        ort_point = np.insert(ort_point, 2, (0))

        return ort_point, ort_point_s

    def _reprojectLinesOriginsOrthogonalFromCurvature(self):
        if self.tracked_centerline_for_lines_origins_computation is None or not self.tracked_centerline_for_lines_origins_computation.isTrackingActive():
            raise NotImplementedError("WorldLineArcLengthParamCenterline: lines origin projection mode "
                                  "FROM_CURVATURE not available with tracking disabled.")

        # set centerline origin
        assert self.fixed_origin_x is not None, "WorldLineArcLengthParamCenterline: fixed_origin_x param must be set in lines origin projection mode FROM_CURVATURE."

        # if self.origin_in_center and self.tracked_centerline_for_lines_origins_computation._tracked_line.origin is not None:
        #     # approximate origin with previous one
        #     # self.origin = self.origin
        #     self.origin = self.tracked_centerline_for_lines_origins_computation._tracked_line.origin
        # else:
        #     self.origin = self.vehicle_reference_pt #np.array([self.fixed_origin_x, 0, 0])
        if self.tracked_centerline_for_lines_origins_computation._tracked_line is not None:
            # previous origin will be either previous road center or extractly vehicle_reference_pt, so both ways is good
            centerline_ref_pt = self.tracked_centerline_for_lines_origins_computation._tracked_line.origin
        else:
            # even if origin should be road center, if no tracking active I don't know where that is, must use cm for reprojection
            centerline_ref_pt = self.vehicle_reference_pt

        # FIXME no need to set self.origin, could use local var here, it's used only internally and then set again at the end
        #  should be actually not origin but sth else, like vehicle ref pt or sth similar, don't know yet

        origin_c = centerline_ref_pt[0:2]
        origin_c_h = np.insert(origin_c, 2, (1)) # homogeneous coordinates

        # compute Rc, theta_c
        origin_c_s = 0
        R_c, curvature_side_at_origin_c = self.tracked_centerline_for_lines_origins_computation.computeRadiusOfCurvature()
        # signed_R_at_origin_c = curvature_side_at_origin_c * R_c
        origin_c_theta = self.tracked_centerline_for_lines_origins_computation.getModel()(np.array([origin_c_s]))[0]

        # compute CIR from centerline
        # origin_c_normal_theta = origin_c_theta - curvature_side_at_origin_c * pi / 2  # theta +/- 90deg according to line side
        # CIR = origin_c + R_c * np.array([cos(origin_c_normal_theta), sin(origin_c_normal_theta)])
        origin_c_normal_theta = origin_c_theta + curvature_side_at_origin_c * pi / 2  # theta +/- 90deg according to line side, pointing towards CIR
        CIR = origin_c + R_c * np.array([cos(origin_c_normal_theta), sin(origin_c_normal_theta)])
        CIR_h = np.insert(CIR, 2, (1))

        new_ss = []
        new_origins = []
        lines_proj_pts = [] # for debugging purposes
        for l in self.lines:
            # compute delta_s ratio
            # x0 = l.getOrigin()
            # x1 = l.getOriginalFitS()[1,:] # FIXME ok???
            # FIXME should avoid also taking origin, cause that's sometimes changed with extrapolation! should instead take first two orig fit points!

            x0_s = l.getFitS()[0]
            x1_s = l.getFitS()[1]
            x0 = l.computePointsGivenS([x0_s])[0] #l.getOriginalFitS()[0]
            x1 = l.computePointsGivenS([x1_s])[0] # l.getOriginalFitS()[1]
            x0_h = np.insert(x0, 2, (1)) # homogeneous coordinates
            x1_h = np.insert(x1, 2, (1)) # homogeneous coordinates
            l_CIR_x0 = np.cross(CIR_h, x0_h) # line through 2 homogeneous points
            l_CIR_x1 = np.cross(CIR_h, x1_h)  # line through 2 homogeneous points
            d_CIR_x0 = np.array([l_CIR_x0[1], -l_CIR_x0[0]]) # direction l_CIR_x0 (intersection with l_inf)
            d_CIR_x1 = np.array([l_CIR_x1[1], -l_CIR_x1[0]]) # direction l_CIR_x0 (intersection with l_inf)
            # delta_theta_l_01 = np.arccos(np.dot(d_CIR_x0,d_CIR_x1) / (np.linalg.norm(d_CIR_x0)*np.linalg.norm(d_CIR_x1)) )
            R_l = np.linalg.norm(x0 - CIR)
            delta_theta_l_01 = np.arccos(np.dot(d_CIR_x0 / np.linalg.norm(d_CIR_x0), d_CIR_x1 / np.linalg.norm(d_CIR_x1)))
            if delta_theta_l_01 != 0.0:
                delta_s_l_01 = R_l * delta_theta_l_01
                delta_s_c_01 = R_c * delta_theta_l_01
                ratio_s_l_to_c = delta_s_l_01 / delta_s_c_01
            else:
                ratio_s_l_to_c = 1.0
            # compute s_l_origin: s of centerline origin projected on l line
            l_CIR_originc = np.cross(CIR_h, origin_c_h)  # line through 2 homogeneous points
            d_CIR_originc = np.array([l_CIR_originc[1], -l_CIR_originc[0]]) # direction l_CIR_originc (intersection with l_inf)
            # delta_theta_l_originc0 = np.arccos(np.dot(d_CIR_originc,d_CIR_x0) / (np.linalg.norm(d_CIR_originc)*np.linalg.norm(d_CIR_x0)) )
            delta_theta_l_originc0 = np.arccos(np.dot(d_CIR_originc / np.linalg.norm(d_CIR_originc), d_CIR_x0 / np.linalg.norm(d_CIR_x0)))
            s_l_originc = x0_s - R_l * delta_theta_l_originc0

            # translate and dilate s points
            new_origin_xy = l.computePointsGivenS(np.array([s_l_originc]))[0]
            new_origin = np.insert(new_origin_xy, 2, (0))
            # FIXME fixed bug changing sign: s_l_originc negative if origin moves backwards,
            #  then should use '-' so that new_s are actually further from new origin
            # new_s = ratio_s_l_to_c * (l.getFitS() + s_l_originc)
            new_s = ratio_s_l_to_c * (l.getFitS() - s_l_originc)

            # # store new origin x,y and projected s points
            new_origins.append(new_origin)
            new_ss.append(new_s)

            l_pts = l.computePointsGivenS(l.getFitS())
            l_proj_pts = l_pts + curvature_side_at_origin_c * (R_c - R_l) * np.vstack((np.sin(l.getModel()(l.getFitS())), -np.cos(l.getModel()(l.getFitS())))).T  # l.computePointsGivenS(new_s)
            lines_proj_pts.append(l_proj_pts)

        self.lines_proj_pts = np.array(lines_proj_pts) # for debugging purposes
        # if self.origin_in_center:
        #     # correct initial estimate and center it
        #     origin_y = np.mean(np.array(new_origins)[:,1])
        #     # FIXME why only y coordinate!?!?! Should be center wrt normal line!!
        #     self.origin = np.array([self.fixed_origin_x, origin_y, 0])
        # FIXME for now I set origin to previous origin, then will correct after fitting
        self.origin = centerline_ref_pt

        return new_ss, new_origins

    def _reprojectLinesOriginsOrthogonalFromEkfCenterlineParams(self):
        if self.tracked_centerline_for_lines_origins_computation is None \
                or not self.tracked_centerline_for_lines_origins_computation.isTrackingActive() \
                or self.ekf_centerline_params_obj is None:
            raise NotImplementedError("WorldLineArcLengthParamCenterline: lines origin projection mode "
                                      "FROM_EKF_CENTERLINE_PARAMS not available with tracking disabled.")

        # compute centerline ref pt from current ekf estimate
        ekf_rho, ekf_theta, ekf_w = tuple(self.ekf_centerline_params_obj.x.squeeze()) # predict not here

        if self.tracked_centerline_for_lines_origins_computation._tracked_line is not None:
            # previous origin will be either previous road center or extractly vehicle_reference_pt, so both ways is good
            centerline_ref_pt = np.array([
                    self.vehicle_reference_pt[0] + ekf_w/2 * ekf_rho * sin(ekf_theta),
                    self.vehicle_reference_pt[1] - ekf_w/2 * ekf_rho * cos(ekf_theta),
                    0
                ])
        else:
            # even if origin should be road center, if no tracking active I don't know where that is, must use cm for reprojection
            centerline_ref_pt = self.vehicle_reference_pt

        origin_c = centerline_ref_pt[0:2]
        origin_c_h = np.insert(origin_c, 2, (1)) # homogeneous coordinates

        # TODO NOTE: I also need curvature, where do i compute it? s=0?? doesn't match my theta technically...
        #  However, my filter is exactly estimating theta in road center, which is also what before was s=0, so if filter converged,
        #  theta and curvature at s=0 should be coherent

        # compute Rc, theta_c
        origin_c_s = 0
        R_c, curvature_side_at_origin_c = self.tracked_centerline_for_lines_origins_computation.computeRadiusOfCurvatureAtSPoint(s_pt=origin_c_s)
        # signed_R_at_origin_c = curvature_side_at_origin_c * R_c

        # FIXME which one??
        origin_c_theta = self.tracked_centerline_for_lines_origins_computation.getModel()(np.array([origin_c_s]))[0]
        # origin_c_theta = ekf_theta  # self.tracked_centerline_for_lines_origins_computation.getModel()(np.array([origin_c_s]))[0]

        # compute CIR from centerline
        # origin_c_normal_theta = origin_c_theta - curvature_side_at_origin_c * pi / 2  # theta +/- 90deg according to line side
        # CIR = origin_c + R_c * np.array([cos(origin_c_normal_theta), sin(origin_c_normal_theta)])
        origin_c_normal_theta = origin_c_theta + curvature_side_at_origin_c * pi / 2  # theta +/- 90deg according to line side, pointing towards CIR
        CIR = origin_c + R_c * np.array([cos(origin_c_normal_theta), sin(origin_c_normal_theta)])
        CIR_h = np.insert(CIR, 2, (1))

        new_ss = []
        new_origins = []
        lines_proj_pts = [] # for debugging purposes
        for l in self.lines:
            # compute delta_s ratio
            # x0 = l.getOrigin()
            # x1 = l.getOriginalFitS()[1,:] # FIXME ok???
            # FIXME should avoid also taking origin, cause that's sometimes changed with extrapolation! should instead take first two orig fit points!

            # FIXME bug: these are not always points closer to centerline! but they should, as i'm computing the ratio and it should be right especially there!

            l_s = l.getFitS()
            s_closer_to_origin_i = np.argmin(np.abs(l_s))
            if s_closer_to_origin_i == len(l_s)-1: # if last element is min # TODO why is it min tho? debug this: print points or so..., (they should be ordered?)
                s_closer_to_origin_i -= 1
            x0_s = l_s[s_closer_to_origin_i]
            x1_s = l_s[s_closer_to_origin_i+1]

            # x0_s = l.getFitS()[0]
            # x1_s = l.getFitS()[1]

            x0 = l.computePointsGivenS([x0_s])[0] #l.getOriginalFitS()[0]
            x1 = l.computePointsGivenS([x1_s])[0] # l.getOriginalFitS()[1]
            x0_h = np.insert(x0, 2, (1)) # homogeneous coordinates
            x1_h = np.insert(x1, 2, (1)) # homogeneous coordinates
            l_CIR_x0 = np.cross(CIR_h, x0_h) # line through 2 homogeneous points
            l_CIR_x1 = np.cross(CIR_h, x1_h)  # line through 2 homogeneous points
            d_CIR_x0 = np.array([l_CIR_x0[1], -l_CIR_x0[0]]) # direction l_CIR_x0 (intersection with l_inf)
            d_CIR_x1 = np.array([l_CIR_x1[1], -l_CIR_x1[0]]) # direction l_CIR_x0 (intersection with l_inf)
            # delta_theta_l_01 = np.arccos(np.dot(d_CIR_x0,d_CIR_x1) / (np.linalg.norm(d_CIR_x0)*np.linalg.norm(d_CIR_x1)) )
            R_l = np.linalg.norm(x0 - CIR)
            delta_theta_l_01 = np.arccos(np.dot(d_CIR_x0 / np.linalg.norm(d_CIR_x0), d_CIR_x1 / np.linalg.norm(d_CIR_x1)))
            if delta_theta_l_01 != 0.0:
                delta_s_l_01 = R_l * delta_theta_l_01
                delta_s_c_01 = R_c * delta_theta_l_01
                ratio_s_l_to_c = delta_s_l_01 / delta_s_c_01
            else:
                ratio_s_l_to_c = 1.0
            # compute s_l_origin: s of centerline origin projected on l line
            l_CIR_originc = np.cross(CIR_h, origin_c_h)  # line through 2 homogeneous points
            d_CIR_originc = np.array([l_CIR_originc[1], -l_CIR_originc[0]]) # direction l_CIR_originc (intersection with l_inf)
            # delta_theta_l_originc0 = np.arccos(np.dot(d_CIR_originc,d_CIR_x0) / (np.linalg.norm(d_CIR_originc)*np.linalg.norm(d_CIR_x0)) )
            delta_theta_l_originc0 = np.arccos(np.dot(d_CIR_originc / np.linalg.norm(d_CIR_originc), d_CIR_x0 / np.linalg.norm(d_CIR_x0)))
            s_l_originc = x0_s - R_l * delta_theta_l_originc0

            # translate and dilate s points
            new_origin_xy = l.computePointsGivenS(np.array([s_l_originc]))[0]
            new_origin = np.insert(new_origin_xy, 2, (0))
            # FIXME fixed bug changing sign: s_l_originc negative if origin moves backwards,
            #  then should use '-' so that new_s are actually further from new origin
            # new_s = ratio_s_l_to_c * (l.getFitS() + s_l_originc)
            new_s = ratio_s_l_to_c * (l.getFitS() - s_l_originc)

            # # store new origin x,y and projected s points
            new_origins.append(new_origin)
            new_ss.append(new_s)

        self.lines_proj_pts = np.array(lines_proj_pts) # for debugging purposes
        # if self.origin_in_center:
        #     # correct initial estimate and center it
        #     origin_y = np.mean(np.array(new_origins)[:,1])
        #     # FIXME why only y coordinate!?!?! Should be center wrt normal line!!
        #     self.origin = np.array([self.fixed_origin_x, origin_y, 0])
        # FIXME for now I set origin to previous origin, then will correct after fitting
        self.origin = centerline_ref_pt

        return new_ss, new_origins

    def _cloneWithoutFixedOrigin(self):
        return WorldLineArcLengthParamCenterline(self.lines,
                                                 self.origin,
                                                 self.getPoints(),
                                                 model_order=self.getModelOrder(),
                                                 model_lsq_reg=self.model_lsq_reg,
                                                 min_n_pts_for_fitting=self.min_n_pts_for_fitting,
                                                 min_m_range_for_fitting=self.min_m_range_for_fitting,
                                                 recenter_first_pt=self.recenter_first_pt,
                                                 id=self.id,
                                                 fixed_origin_x=None,
                                                 vehicle_reference_pt=self.vehicle_reference_pt,
                                                 origin_in=self.origin_in,
                                                 # origin_in_center=self.origin_in_center,
                                                 lines_origin_projection_mode=self.lines_origin_projection_mode)

    # FIXME incorrect if use origin in center option!
    def computeLateralOffset(self, x_pt=None, centerline_pt=None, lines_corresponding_pts=None):
        # assert: x_pt set => centerline_pt and lines_corresponding_pts are not set
        assert not (x_pt is not None) or (centerline_pt is None and lines_corresponding_pts is None), \
            "WorldLineArcLengthParamCenterline.computeLateralOffset(): if x_pt is set, centerline_pt and lines_corresponding_pts must be None."
        if x_pt is None:
            centerline_pt = self.origin
            lines_corresponding_pts = np.asarray(self.lines_corresponding_origins_prev_t)
        else:
            if centerline_pt is None or lines_corresponding_pts is None:
                raise NotImplementedError("WorldLineArcLengthParamCenterline.computeLateralOffset(): x_pt management not yet implemented.")
                # TODO compute points perpendicular to x_pt on centerline

        # FIXME notice, I'm assuming left and right are first and second respectively: is it ok?
        left_distance = np.linalg.norm( lines_corresponding_pts[0,:] - centerline_pt )
        right_distance = np.linalg.norm(lines_corresponding_pts[1, :] - centerline_pt)

        left_distance_percent = left_distance / (left_distance + right_distance)
        # right_distance_percent = right_distance / (left_distance + right_distance)
        right_distance_percent = 1 - left_distance_percent

        return [left_distance, right_distance], [left_distance_percent, right_distance_percent]

    def computeVehicleLateralOffsetFromRoadCenter(self):
        road_center = self.getRoadCenter()
        vehicle_ref_pt = self.getVehicleReferencePoint()
        offset_sign = np.sign(road_center[1] - vehicle_ref_pt[1])
        return offset_sign * np.linalg.norm(self.getRoadCenter() - self.getVehicleReferencePoint())

    def getRoadCenter(self):
        if self.road_center is not None:
            return self.road_center
        return np.array([np.nan, np.nan, np.nan])

    def getVehicleReferencePoint(self):
        return self.vehicle_reference_pt

# class BevLineProxy(ImageLine):
#     """
#     Represents an ImageLine on a BEV image derivated from a corresponding WorldLine.
#     """
#     def __init__(self, world_line):
#         self._world_line = world_line
#         points = self._computeArtificialBevPoints()
#         super(BevLineProxy, self).__init__(points=points,
#                                            model_type=self._world_line.model_type,
#                                            model_order=self._world_line.model_order,
#                                            id=self._world_line.id)
#
#     def _computeArtificialBevPoints(self, n_artificial_pts=300):
#         if self._world_line.getModelType() == 'arc_length_parametrization':
#             # generate many points to fit
#             s = np.linspace(0, np.max(self._world_line.getFitS()), n_artificial_pts)
#             xy_w = self._world_line.computePointsGivenS(s)
#             x_w = xy_w[:, 0]
#             y_w = xy_w[:, 1]
#         else:
#             # generate many points to fit
#             x_w = np.linspace(self._world_line.bev_obj.outView[0], self._world_line.bev_obj.outView[1], n_artificial_pts)
#             y_w = self._world_line.computeYGivenX(x_w)
#         pts_w_z0 = np.hstack((x_w[np.newaxis].T, y_w[np.newaxis].T))
#         pts_w = np.insert(pts_w_z0, 2, (0), axis=1)
#         pts_bev = self._world_line.bev_obj.projectWorldPointsToBevPoints(pts_w)
#         return pts_bev
#         # self._fit_points = self._world_line.bev_obj.projectWorldPointsToBevPoints(pts_w)
#         # self._model = self._fitModel()
class BevLineProxy(ImageLine):
    """
    Represents an ImageLine on a BEV image derivated from a corresponding WorldLine.
    """
    def __init__(self, world_line, allow_extrapolation=None):
        self._world_line = world_line
        if allow_extrapolation is not None:
            self.allow_extrapolation = allow_extrapolation
        else:
            # default values
            if self._world_line.model_type.startswith("arc_length_parametrization"):
                self.allow_extrapolation = False
            else:
                self.allow_extrapolation = True
        # points = self._computeArtificialBevPoints()
        points = self._world_line.bev_obj.projectWorldPointsToBevPoints(self._world_line._points)
        super(BevLineProxy, self).__init__(points=points,
                                           model_type=self._world_line.model_type,
                                           model_order=self._world_line.model_order,
                                           id=self._world_line.id)
        self._setup()

    def _fitModel(self):
        return None # no model is fit

    def _setup(self):
        # get world x axis projected in bev and save for _computeOrdinateGivenAbscissae
        bev_limits = self._world_line.bev_obj.outView[0:2]
        x_axis_extremes_w = np.hstack(( np.linspace(bev_limits[0], bev_limits[1], 2)[np.newaxis].T,
                              np.zeros((2,1)), np.zeros((2,1)) ))
        x_axis_extremes_bev = self._world_line.bev_obj.projectWorldPointsToBevPoints(x_axis_extremes_w)
        x_axis_extremes_bev_h = np.insert(x_axis_extremes_bev, 2, 1, axis=1)
        x_axis_line_bev_h_ord_abs_c = np.cross(x_axis_extremes_bev_h[0,:], x_axis_extremes_bev_h[1,:])
        x_axis_line_bev_h_abs_ord_c = np.array([x_axis_line_bev_h_ord_abs_c[1], x_axis_line_bev_h_ord_abs_c[0], x_axis_line_bev_h_ord_abs_c[2]])
        self.x_axis_line_bev_h = x_axis_line_bev_h_abs_ord_c

    def _computeOrdinateGivenAbscissae(self, abs_pts):
        abs_pts_arr = np.asarray(abs_pts)
        # project abs_pts on x axis projected in bev
        # https://math.stackexchange.com/questions/727743/homogeneous-coordinates
        proj_abs_pts_on_axis_bev = self._projectPointsOnLineSameAbscissae(abs_pts_arr, self.x_axis_line_bev_h) # [abs,ord] = [y,x]_bev
        proj_abs_pts_on_axis_xy_bev = np.fliplr(proj_abs_pts_on_axis_bev)
        # convert points to world points
        proj_abs_pts_on_axis_w = self._world_line.bev_obj.projectBevPointsToWorldGroundPlane(proj_abs_pts_on_axis_xy_bev)
        # get x world point
        x_pts_w = proj_abs_pts_on_axis_w[:,0]

        if not self.allow_extrapolation:
            is_pt_extrapolated_i = np.where(~self._isExtrapolating(x_pts_w))[0] # TODO this
            # remove extrapolation pts
            x_pts_w = x_pts_w[is_pt_extrapolated_i] # FIXME is this ok? I'm returning less points than I was requested...

        # compute line y world with world_line model
        y_pts_w = self._world_line.computeYGivenX(x_pts_w)
        fit_pts_w_z0 = np.hstack((x_pts_w[np.newaxis].T, y_pts_w[np.newaxis].T))
        fit_pts_w = np.insert(fit_pts_w_z0, 2, 0, axis=1)
        # reproject line x,y world in bev
        fit_pts_bev = self._world_line.bev_obj.projectWorldPointsToBevPoints(fit_pts_w)
        ord_pts_bev = fit_pts_bev[:,0]

        if np.isscalar(abs_pts):
            return ord_pts_bev.item()
        else:
            return np.asarray(ord_pts_bev)

    def _projectPointsOnLine(self, p, l):
        a, b, c = tuple(l)
        u, v, w = tuple([np.array(el) for el in zip(*p)])  # [x1,...,xn], [y1,...,yn], [z1,...,zn]
        proj_pts = np.vstack([b * (b * u - a * v) - a * c, -a * (b * u - a * v) - b * c, w * (a * a + b * b)]).T
        return proj_pts[0:2] / proj_pts[2]

    def _projectPointsOnLineSameAbscissae(self, pt_abs, l):
        a,b,c = tuple(l)
        x = pt_abs
        y = (-c - a*x)/b
        proj_pts = np.vstack((x,y)).T
        return proj_pts

    def _isExtrapolating(self, x_pts_w):
        line_limits = self._world_line.getPointsXLimits()
        return (x_pts_w < line_limits[0]) | ( x_pts_w > line_limits[1])

    # def _computeArtificialBevPoints(self, n_artificial_pts=300):
    #     if self._world_line.getModelType() == 'arc_length_parametrization':
    #         # generate many points to fit
    #         s = np.linspace(0, np.max(self._world_line.getFitS()), n_artificial_pts)
    #         xy_w = self._world_line.computePointsGivenS(s)
    #         x_w = xy_w[:, 0]
    #         y_w = xy_w[:, 1]
    #     else:
    #         # generate many points to fit
    #         x_w = np.linspace(self._world_line.bev_obj.outView[0], self._world_line.bev_obj.outView[1], n_artificial_pts)
    #         y_w = self._world_line.computeYGivenX(x_w)
    #     pts_w_z0 = np.hstack((x_w[np.newaxis].T, y_w[np.newaxis].T))
    #     pts_w = np.insert(pts_w_z0, 2, (0), axis=1)
    #     pts_bev = self._world_line.bev_obj.projectWorldPointsToBevPoints(pts_w)
    #     return pts_bev
    #     # self._fit_points = self._world_line.bev_obj.projectWorldPointsToBevPoints(pts_w)
    #     # self._model = self._fitModel()


# class BevLineArcLengthParamProxy(BevLineProxy):
#     """
#         Represents an ImageLine on a BEV image derivated from a corresponding WorldLineArcLengthParam.
#     """
#     def __init__(self, world_line):
#         self._world_line = world_line
#         points = self._computeArtificialBevPoints()
#         super(BevLineArcLengthParamProxy, self).__init__(points=points,
#                                                          model_type=self._world_line.model_type,
#                                                          model_order=self._world_line.model_order,
#                                                          id=self._world_line.id)
#
#     def _computeArtificialBevPoints(self, n_artificial_pts=300):
#         # generate many points to fit
#         s = np.linspace(0, np.max(self._world_line.getFitS()), n_artificial_pts)
#         xy_w = self._world_line.computePointsGivenS(s)
#         x_w = xy_w[:, 0]
#         y_w = xy_w[:, 1]
#         pts_w_z0 = np.hstack((x_w[np.newaxis].T, y_w[np.newaxis].T))
#         pts_w = np.insert(pts_w_z0, 2, (0), axis=1)
#         pts_bev = self._world_line.bev_obj.projectWorldPointsToBevPoints(pts_w)
#         return pts_bev
#         # self._fit_points = self._world_line.bev_obj.projectWorldPointsToBevPoints(pts_w)
#         # self._model = self._fitModel()
#
#     def _fitModel(self):
#         return self._fitModelInS()
#
#     def computePointsGivenS(self, s):

# class LineFactory(object):
#     @staticmethod
#     def createLine():


class Window(object):
    def __init__(self, center, size, dir=pi/2, mask_padding=0):
        self.center = center.copy() # [x,y]
        self.size = size.copy() # [width, height]
        self.dir = dir
        self.mask_padding = mask_padding
        self.mask = self._createWindowMask(self.size, self.dir)

    @staticmethod
    def createWindow(window_boundaries, dir=None):
        center = np.mean(window_boundaries, axis=0)
        size = np.array([window_boundaries[0, 1] - window_boundaries[0, 0], window_boundaries[1, 1] - window_boundaries[1, 0]]) # width, height # TODO check this!!
        if dir is not None:
            return Window(center, size, dir)
        return Window(center, size)

    def clone(self):
        return Window(self.center, self.size, self.dir, self.mask_padding)

    # def _setupWithNewBoundaries(self, window_boundaries, dir=None):
    #     center = np.mean(window_boundaries, axis=0)
    #     size = np.array([window_boundaries[0, 1] - window_boundaries[0, 0],
    #                      window_boundaries[1, 1] - window_boundaries[1, 0]])  # width, height # TODO check this!!
    #     self.__init__(center, size, dir)

    # def getBoundaries(self):
    #     # [[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]]
    #     return np.array([self.center - self.size/2, self.center + self.size/2 ]) # TODO check this

    def getWindowExtremities(self):
        if self.mask is not None:
            # FIX_ME is this needed? isn't mask just the right size needed to fit patch? - DONE
            # mask_pts = np.argwhere(self.mask) # FIXME
            # min_x = np.min(mask_pts[:, 1])
            # max_x = np.max(mask_pts[:, 1])
            # min_y = np.min(mask_pts[:, 0])
            # max_y = np.max(mask_pts[:, 0])
            # print([[min_x, min_y], [max_x, max_y]], (max_y-min_y+1, max_x-min_x+1), self.mask.shape )
            # TESTING with:
            # assert((max_y-min_y+1, max_x-min_x+1)==self.mask.shape)
            # NEW EFFICIENT SOLUTION: - assuming mask fits exactly the patch
            min_x = 0
            max_x = self.mask.shape[1]
            min_y = 0
            max_y = self.mask.shape[0]
        else:
            min_x = 0
            max_x = self.size[0]
            min_y = 0
            max_y = self.size[1]
        return np.array([[min_x, min_y], [max_x, max_y]])

    def getWindowExtremitiesInImage(self):
        patch_size_xy = self.mask.shape[0:2][::-1] if self.mask is not None else self.size
        return self.getWindowExtremities() - np.array(patch_size_xy) / 2.0 + self.center

    def getMaskBoundaries(self):
        # [[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]]
        # return np.array([self.center - self.mask.shape[1] / 2.0, self.center + self.mask.shape[0] / 2.0])  # TODO check this
        patch_size_xy = self.mask.shape[0:2][::-1] if self.mask is not None else self.size
        return np.array([self.center - np.array(patch_size_xy)/2.0,
                         self.center + np.array(patch_size_xy)/2.0])

    def moveWindow(self, new_center, new_dir=None, new_size=None):
        if new_dir is not None:
            self.dir = new_dir
        if new_size is not None:
            self.size = new_size.copy()
        self.center = new_center.copy()
        self.mask = self._createWindowMask(self.size, self.dir)

    def enlargeWindow(self, new_size=None, top=None, bottom=None, left=None, right=None):
        assert new_size is not None or (top is not None and bottom is not None and left is not None and right is not None), "Window.enlargeWindow: either new_size or top,bottom,left,right params must be passed."
        if new_size is not None:
            self.size = new_size.copy()
            self.mask = self._createWindowMask(self.size, self.dir)
        else:
            # re-center
            # FIX_ME BUG!! right,left are wrt mask, but center is in image --> depends also on direction!!! - DONE?
            # self.center[0] += (right - left) / 2.0
            # self.center[1] += (bottom - top) / 2.0
            delta_center_x = right - left
            delta_center_y = bottom - top
            self.center[0] += (delta_center_x / 2.0) * cos(self.dir - pi / 2) + (delta_center_y / 2.0) * sin(self.dir - pi / 2)
            self.center[1] += -(delta_center_x / 2.0) * sin(self.dir - pi / 2) + (delta_center_y / 2.0) * cos(self.dir - pi / 2)
            # re-size
            self.size[0] += right + left
            self.size[1] += bottom + top
            self.mask = self._createWindowMask(self.size, self.dir)

    def rotateWindow(self, new_dir):
        self.dir = new_dir
        self.mask = self._createWindowMask(self.size, self.dir)

    def extractImagePatch(self, image):
        if self.mask is not None:
            # Obs: approximate with floor and optional +1 s.t. distance is preserved (and it's same as mask size)
            top_left_x = int(floor(self.center[0] - self.mask.shape[1] / 2.0))
            top_left_y = int(floor(self.center[1] - self.mask.shape[0] / 2.0))
            bottom_right_x = top_left_x + self.mask.shape[1]
            bottom_right_y = top_left_y + self.mask.shape[0]
        else:
            # Obs: approximate with floor and optional +1 s.t. distance is preserved (and it's same as mask size)
            top_left_x = int(floor(self.center[0] - self.size[0] / 2.0))
            top_left_y = int(floor(self.center[1] - self.size[1] / 2.0))
            bottom_right_x = top_left_x + self.size[0]
            bottom_right_y = top_left_y + self.size[1]
        if top_left_x < 0:
            border_left = 0 - top_left_x
            top_left_x = 0
        else:
            border_left = 0
        if top_left_y < 0:
            border_top = 0 - top_left_y
            top_left_y = 0
        else:
            border_top = 0
        if bottom_right_x > image.shape[1]:
            border_right = bottom_right_x - image.shape[1]
            bottom_right_x = image.shape[1]
        else:
            border_right = 0
        if bottom_right_y > image.shape[0]:
            border_bottom = bottom_right_y - image.shape[0]
            bottom_right_y = image.shape[0]
        else:
            border_bottom = 0
        patch = image[top_left_y:bottom_right_y,
                      top_left_x:bottom_right_x]
        patch = cv2.copyMakeBorder(patch, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, (0))
        if self.mask is not None:
            patch = cv2.bitwise_and(patch, patch, mask=self.mask)
        return patch

    def bound(self, bounds_im):
        if self.mask is not None:
            bound_mask = np.zeros_like(self.mask)
            # bounds in window (mask) coordinate
            mask_size = np.array(self.mask.shape[0:2][::-1])
            bounds_w = np.ceil(bounds_im + mask_size/2.0 - self.center).astype(np.int) # [[top_left_x,top_left_y],[bottom_right_x,bottom_right_y]]
            # clip to window size
            bounds_w[:, 0] = np.clip(bounds_w[:, 0], 0, mask_size[0])
            bounds_w[:, 1] = np.clip(bounds_w[:, 1], 0, mask_size[1])
            bound_mask[bounds_w[0,1]:bounds_w[1,1], bounds_w[0,0]:bounds_w[1,0]] = 1
            self.mask = cv2.bitwise_and(self.mask, self.mask, mask=bound_mask)

    def _createWindowMask(self, size, dir):
        if dir == pi/2 or dir == -pi/2:
            return None # no mask if dir is straight up --> need only a ROI, no rotations involved
        mask = np.ones((size[1], size[0]), dtype=np.uint8)
        rotated_mask, _ = rotate_image_fit(mask, degrees(dir-pi/2))
        padded_rotated_mask = cv2.copyMakeBorder(rotated_mask,
                                          top=self.mask_padding, bottom=self.mask_padding,
                                          left=self.mask_padding, right=self.mask_padding,
                                          borderType=cv2.BORDER_CONSTANT, value=0)
        return padded_rotated_mask

class LineBuilder:
    """
    Helper class designed to be used by MovingWindowFeatureSelection to facilitate the constructions of line objects
    within the algorithm it implements.
    """

    LINE_SIDE_LEFT = 0
    LINE_SIDE_RIGHT = 1

    def __init__(self, init_window_boundaries, init_dir, window_size, im_bounds, init_point=None, init_window_size=None,
                 dir_allowed_range=None, max_resize=3, line_side=None, line_id=None):
        """
        Initializes the LineBuilder.
        :param init_window_boundaries: [[top_left_x,top_left_y],[bottom_right_x,bottom_right_y]] or None, initial window
        :param init_dir: initial search direction, in rad
        :param window_size: [width,height]
        :param im_bounds: [image_width, image_height]
        :param init_point: alternative to init_window, if set, the initial window is constructed around this point
        :param dir_allowed_range: allowed range of search direction at any step, in rad
        :param max_resize: maximum number of times the window can be resized before being labeled as completed
        :param line_side: (LINE_SIDE_LEFT|LINE_SIDE_RIGHT) or None, side of the road where the line lies, can be used by the external algorithm
        :param line_id: number or None, id that can be used outside (e.g. tracking)
        """
        self.points = np.empty((0,2))
        self.window_size = window_size
        self.init_window_size = init_window_size if init_window_size is not None else self.window_size
        self.im_bounds = im_bounds # [x,y]
        self.forwardDir = init_dir
        self.dir_allowed_range = dir_allowed_range
        # completed labeling: manage when window can be considered completed and further search should be stopped
        self.completing = False # window is touching image border --> next point collected, line completed
        self.completed = False # last point added after window touched border (or no point found in last window)
        self.lost = False # algorithm lost the line (usually after window is enlarged unsuccessfully too many times)
        if init_point is not None:
            #self.centerWindowOnPoint(init_point, window_size=init_window_size, is_init=True)
            self.window = Window(init_point, self.init_window_size, dir=self.forwardDir)
        else:
            self.window = Window.createWindow(init_window_boundaries, dir=self.forwardDir)
        self.resize_count = 0
        self.max_resize = max_resize
        self.line_side = line_side # can be used externally to resolve split conflicts
        self.line_id = line_id  # can be used in tracking
        self.just_init = True

    def addPoint(self, point):
        self.points = np.append(self.points, np.array([point]), axis=0)
        self._computeNewForwardDir()
        self.checkCompleted()
        self.resize_count = 0
        self.setNotJustInit()

    def _computeNewForwardDir(self):
        if self.points.shape[0] >= 2:
            # self.forwardDir = LineBuilder.computeDir(self.points[-2, :], self.points[-1, :])  # compute dir between 2 last points
            # forward dir is mean of angle between 5 last points detected
            n_pts_mean = min(5, self.points.shape[0]) # use max 5 previous pts, or less if no more available
            self.forwardDir = np.mean([LineBuilder.computeDir(self.points[i - 1, :], self.points[i, :]) for i in range(-1, -n_pts_mean, -1)])
            if self.dir_allowed_range is not None:
                # check direction in allowed range
                # # fallback 1: clip to range
                # self.forwardDir = np.clip(self.forwardDir, self.dir_allowed_range[0], self.dir_allowed_range[1])
                # fallback 2: set line as completed if tries to go out of range
                if self.forwardDir < self.dir_allowed_range[0] or self.forwardDir > self.dir_allowed_range[1]:
                    self.setCompleted()
        # else:
        #    pass # use previous self.forwardDir

    @classmethod
    def computeDir(cls, pt1, pt2):
        return atan2( -(pt2[1]-pt1[1]), pt2[0]-pt1[0]) # takes into account that y points down, dir 0deg points right, counterclockwise

    def moveWindowForward(self, distance):
        return self.moveWindow(self.forwardDir, distance)

    def moveWindow(self, direction, distance, window_size=None):
        starting_point = self.points[-1,:]
        new_point = starting_point + distance * np.array([cos(direction), - sin(direction)])
        if window_size is not None:
            self.centerWindowOnPoint(new_point, window_size=window_size)
        else:
            self.centerWindowOnPoint(new_point)
        return new_point # for debugging

    def centerWindowOnLastPoint(self):
        self.centerWindowOnPoint(self.points[-1,:])

    def centerWindowOnPoint(self, center_point, window_size=None, window_dir=None, is_init=False):
        if window_size is None:
            window_size = self.window.size # maintain current window size
        if window_dir is None:
            window_dir = self.forwardDir
        self.window = Window(center_point, window_size, window_dir)
        self.checkCompleted() # if was completing before, now completed (I was waiting for last point in previous window, if I move window I'm done regardless)
        self.boundWindow(checkCompleting=(not is_init))
        self.setNotJustInit()

    def getWindowCenterPoint(self):
        # return np.mean(self.window, axis=0) # TODO check this
        return self.window.center

    def resizeWindow(self, deltaSize, fixSide=np.array([None,None]), checkCompleting=True, countResize=True):
        # deltaSize = [delta_x, delta_y]
        # fixSide = ["left"|"right"|None,"top"|"bottom"|None], None doesn't fix it
        fixSide_x = fixSide[0]
        fixSide_y = fixSide[1]
        # resize x
        if fixSide_x == "left":
            right = deltaSize[0]
            left = 0
        elif fixSide_x == "right":
            right = 0
            left = deltaSize[0]
        elif fixSide_x is None:
            right = deltaSize[0] / 2.0
            left = deltaSize[0] / 2.0
        # resize y
        if fixSide_y == "top":
            bottom = deltaSize[1]
            top = 0
        elif fixSide_y == "bottom":
            bottom = 0
            top = deltaSize[1]
        elif fixSide_y is None:
            bottom = deltaSize[1] / 2.0
            top = deltaSize[1] / 2.0
        self.window.enlargeWindow(new_size=None, top=top, bottom=bottom, left=left, right=right)
        self.checkCompleted() # if before was already completing, now completed
        self.boundWindow(checkCompleting)
        # if max resize reached, completed
        if countResize:
            self.resize_count += 1
            if self.resize_count >= self.max_resize:
                # self.completed = True
                self.lost = True

    def resetWindowSize(self):
        self.window = Window(self.window.center, self.window_size, self.window.dir, self.window.mask_padding)

    def boundWindow(self, checkCompleting=True):
        if checkCompleting and not self.isJustInit():
            self.checkCompleting()
        # self.window[:, 0] = np.clip(self.window[:, 0], 0, self.im_bounds[1]) # TODO change
        # self.window[:, 1] = np.clip(self.window[:, 1], 0, self.im_bounds[0])
        bounds = np.array([[0, 0],[self.im_bounds[1], self.im_bounds[0]]]) # [[x_min,y_min], [x_max, y_max]]
        self.window.bound(bounds)

    def checkCompleting(self):
        if not self.completing:
            window_mask_extremities_in_mask = self.window.getWindowExtremities()
            window_mask_extremities_in_image = self.window.getWindowExtremitiesInImage() #window_mask_extremities_in_mask - np.array(self.window.mask.shape[0:2][::-1]) / 2.0 + self.window.center
            # set completing flag if part of window is out of image bounds
            x_min = window_mask_extremities_in_image[0, 0]
            y_min = window_mask_extremities_in_image[0, 1]
            x_max = window_mask_extremities_in_image[1, 0]
            y_max = window_mask_extremities_in_image[1, 1]
            self.completing = x_min < 0 or \
                              y_min < 0 or \
                              x_max > self.im_bounds[1] or \
                              y_max > self.im_bounds[0]

            # ADDITION: directly set as completed if whole window is out of image bounds
            if (x_min < 0 and x_max < 0) or \
                (y_min < 0 and y_max < 0) or \
                (x_min > self.im_bounds[1] and x_max > self.im_bounds[1]) or \
                (y_min > self.im_bounds[0] and y_max > self.im_bounds[0]):
                self.completed = True

    def checkCompleted(self):
        if self.completing: # if was completing, adding point is completed
            self.completed = True

    def isCompleted(self):
        return self.completed

    def setCompleted(self):
        self.completed = True

    def isLost(self):
        return self.lost

    def resetLost(self):
        self.lost = False

    def setNotJustInit(self):
        self.just_init = False

    def isJustInit(self):
        return self.just_init

    def buildLine(self, model_type=None, model_order=None):
        try:
            if model_type is not None and model_order is not None:
                return ImageLine(self.points, model_type=model_type, model_order=model_order, id=self.line_id)
            elif model_type is not None:
                return ImageLine(self.points, model_type=model_type, id=self.line_id)
            elif model_order is not None:
                return ImageLine(self.points, model_order=model_order, id=self.line_id)
            else:
                return ImageLine(self.points, id=self.line_id)
        except Line.LineException as e:
            print(e)
            return None

    def getPoints(self):
        return self.points

class LineBuilderCollection:
    """
    Collection of LineBuilders, directly used by MovingWindowFeatureSelection to manage the collection and have fast
    access to specific groups of them
    """

    def __init__(self):
        self.line_builders = np.array([])

    def addLineBuilder(self, line_builder):
        self.line_builders = np.append(self.line_builders, line_builder)

    def getLineBuilders(self, indices=None):
        if indices is None: # or len(indices)==0: # FIXME I don't know why I added this OR but python version doesn't work with it (because it returns all builders when should return none!)
            return self.line_builders
        return self.line_builders[indices.astype(np.int)]

    def isCompleted(self, index):
        return self.line_builders[index].isCompleted()

    def getNotCompleted(self):
        indices_not_completed = np.array([ i for i in range(self.getLineBuilders().size) if not self.isCompleted(i)]).astype(np.int)
        # indices_not_completed = np.argwhere([not self.isCompleted(i) for i in range(self.getRoadBuilders().size)])
        return indices_not_completed, self.getLineBuilders(indices_not_completed)

    def getCompleted(self):
        indices_completed = np.array([i for i in range(self.getLineBuilders().size) if self.isCompleted(i)])
        return indices_completed, self.getLineBuilders(indices_completed)

    def getCompletedLinePoints(self):
        _, completed_line_builders = self.getCompleted()
        lines_points = []
        for rb in completed_line_builders:
            line_points = rb.getPoints()
            lines_points.append(line_points)
        return lines_points

    def getCompletedLines(self, model_type=None, model_order=None, bev_obj=None,
                          world_model_type=None, world_model_order=None,
                          world_line_spline_mae_allowed=None, world_line_model_alp_lsq_reg=None,
                          world_line_min_n_pts_for_fitting=None, world_line_min_m_range_for_fitting=None,
                          world_line_postprocess_points=False,
                          world_line_postprocess_points_model_type=None, world_line_postprocess_points_model_order=None,
                          world_line_postprocess_points_model_spline_mae_allowed=None, world_line_postprocess_points_model_extrapolate=True,
                          fixed_origin_x=2,
                          displaced_prev_fitting_points=None):
        """
        Create Line for each completed line built.
        If a Bev object is passed, assumes the lines have been built in BEV image, and creates corresponding Line
        objects for their representation in both the front view image and the world coordinate frame
        :param model_type: param to init each Line
        :param model_order: param to init each Line
        :param bev_obj: instance of Bev
        :return: if bev_obj is passed, returns world_lines, front_lines, bev_lines; else returns only bev_lines
        """
        bev_lines = np.array([])
        if bev_obj is not None:
            front_lines = np.array([])
            world_lines = np.array([])
        if world_model_type is None or world_model_order is None:
            world_model_type = model_type
            world_model_order = model_order
        _, completed_line_builders = self.getCompleted()

        for rb in completed_line_builders:
            rb_line_side_i = 0 if rb.line_side == LineBuilder.LINE_SIDE_LEFT else 1
            try:
                bev_line = rb.buildLine(model_type=model_type, model_order=model_order)
                bev_line_points = bev_line.getPoints()
            except:
                bev_line = None
                bev_line_points = rb.getPoints()

            if bev_line is not None or (bev_line_points is not None and len(bev_line_points)>0):
                if bev_line is not None:
                    # line in bev image
                    bev_lines = np.append(bev_lines, bev_line)
                if bev_obj is not None:
                    # line in world coordinates
                    try:
                        world_pts = bev_obj.projectBevPointsToWorldGroundPlane(bev_line_points)
                        if world_model_type.startswith("arc_length_parametrization"):
                            if world_line_postprocess_points:
                                new_points = world_pts.copy()

                                # todo this is param
                                line_distance_threshold = 5  # m

                                # if has previous points and I detected new points
                                if displaced_prev_fitting_points is not None and len(new_points) > 0 \
                                        and rb_line_side_i < len(displaced_prev_fitting_points):
                                    # add previous displaced points to fitting points
                                    detected_world_pts = np.concatenate((new_points, displaced_prev_fitting_points[rb_line_side_i]), axis=0)

                                    preprocessing_line_distance_filter = WorldLine(bev_obj,
                                                                                   detected_world_pts,
                                                                                   model_type="poly_robust",  # "poly",
                                                                                   model_order=3,
                                                                                   min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                                                                   min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                                                                   id=rb.line_id
                                                                                   )
                                    filter_x_limits = (np.min(detected_world_pts[:, 0]), np.max(detected_world_pts[:, 0]))
                                    n_filter_pts = 500
                                    filter_x_pts = np.linspace(filter_x_limits[0], filter_x_limits[1], n_filter_pts)
                                    filter_y_pts = preprocessing_line_distance_filter.computeYGivenX(filter_x_pts)
                                    preprocessing_line_distance_filter_pts = np.vstack((filter_x_pts, filter_y_pts)).T
                                    new_points_filter_distance = np.array([np.min(np.linalg.norm(pt[0:2] - preprocessing_line_distance_filter_pts, axis=1)) for pt in new_points])
                                    #prev_points_filter_distance = np.array([np.min(np.linalg.norm(pt - preprocessing_line_distance_filter_pts)) for pt in displaced_prev_fitting_points[rb_line_side_i]])

                                    if (new_points_filter_distance <= line_distance_threshold).any():
                                        points_filter_distance = np.array([np.min(np.linalg.norm(pt[0:2] - preprocessing_line_distance_filter_pts, axis=1)) for pt in detected_world_pts])

                                        # filter based on distance
                                        detected_world_pts = detected_world_pts[points_filter_distance <= line_distance_threshold, :]

                                    else:
                                        # keep only new points
                                        detected_world_pts = None
                                else:
                                    detected_world_pts = None

                                if detected_world_pts is None:
                                    detected_world_pts = new_points.copy()

                                    preprocessing_line_distance_filter = WorldLine(bev_obj,
                                                                                   detected_world_pts,
                                                                                   model_type="poly_robust",  # "poly",
                                                                                   model_order=3,
                                                                                   min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                                                                   min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                                                                   id=rb.line_id
                                                                                   )
                                    filter_x_limits = (np.min(detected_world_pts[:, 0]), np.max(detected_world_pts[:, 0]))
                                    n_filter_pts = 500
                                    filter_x_pts = np.linspace(filter_x_limits[0], filter_x_limits[1], n_filter_pts)
                                    filter_y_pts = preprocessing_line_distance_filter.computeYGivenX(filter_x_pts)
                                    preprocessing_line_distance_filter_pts = np.vstack((filter_x_pts, filter_y_pts)).T
                                    points_filter_distance = np.array([np.min(np.linalg.norm(pt[0:2] - preprocessing_line_distance_filter_pts, axis=1)) for pt in detected_world_pts])
                                    # filter based on distance
                                    detected_world_pts = detected_world_pts[points_filter_distance <= line_distance_threshold, :]


                                # TODO try spline st. x0'==x1' and xn'==xn-1' (extrapolation with a line) - nah


                                ### TEST

                                # TODO see test results, cause
                                #  s should be connected to number of points! cause its compared with a sum over all pts!!
                                #  s is kinda of the mean squared error allowed!
                                #  world_line_postprocess_points_model_spline_s -> world_line_postprocess_points_model_spline_mae_allowed = 3 m
                                #  spline_s = detected_world_pts.shape[0] * world_line_postprocess_points_model_spline_mae_allowed ** 2
                                #  +
                                #  not enough, cause if points lot divergent, then spline will still start oscillating
                                #  --> need to filter out outliers --> how?
                                #  no robust estimators for splines around, so:
                                #  based on previous spline, remove points not close enough to prev spline if there are at least some points close to it


                                # fign=61+rb_line_side_i if rb_line_side_i is not None else 66
                                # if configs.debug and fign in configs.debug_only:
                                #     plt.figure(fign);
                                #     plt.clf();
                                #     ax = plt.subplot(111)
                                #     ax.plot(-detected_world_pts[:, 1], detected_world_pts[:, 0], 'r.')
                                #     for s in np.round(np.logspace(0,4,10)).astype(np.int):
                                #         preprocessing_line = WorldLine(bev_obj, detected_world_pts, model_type=world_line_postprocess_points_model_type,  # "poly # "spline" # "poly_robust"
                                #                                        model_order=world_line_postprocess_points_model_order,
                                #                                        min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                #                                        min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                #                                        spline_s=s,
                                #                                        id=rb.line_id
                                #                                        )
                                #         world_pts_x_limits = (np.min(detected_world_pts[:, 0]), min(bev_obj.outView[1], np.max(detected_world_pts[:, 0])))
                                #         n_postprocess_pts = 100
                                #         postprocessed_x_pts = np.linspace(world_pts_x_limits[0], world_pts_x_limits[1], n_postprocess_pts)
                                #         postprocessed_y_pts = preprocessing_line.computeYGivenX(postprocessed_x_pts)
                                #         world_pts = np.insert(np.vstack((postprocessed_x_pts, postprocessed_y_pts)).T, 2, (0), axis=1)
                                #
                                #         ax.plot(-world_pts[:, 1], world_pts[:, 0], '-', label='s={}'.format(s))
                                #         ax.set_xlim([-15, 15])
                                #         ax.set_ylim([-2, bev_obj.outView[1] + 5])
                                #         ax.set_aspect('equal', adjustable='box')
                                #         ax.set_title("N_pts: {}".format(detected_world_pts.shape[0]))
                                #     plt.figure(fign);
                                #     plt.legend();

                                # /TEST

                                world_line_postprocess_points_model_spline_s = detected_world_pts.shape[0] * (world_line_postprocess_points_model_spline_mae_allowed ** 2)

                                preprocessing_line = WorldLine(bev_obj, detected_world_pts,
                                                               model_type=world_line_postprocess_points_model_type,  # "poly # "spline" # "poly_robust"
                                                               model_order=world_line_postprocess_points_model_order,
                                                               min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                                               min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                                               spline_s=world_line_postprocess_points_model_spline_s,
                                                               id=rb.line_id
                                                               )

                                # if world_line_postprocess_points_model_type == "spline":
                                #     if displaced_prev_fitting_points is not None:
                                #         weights = 1.0 / (1.0 + np.square([np.min(pt - displaced_prev_fitting_points[rb_line_side_i]) for pt in new_points]))
                                #         weights = np.concatenate((weights, np.ones(displaced_prev_fitting_points[rb_line_side_i].shape[0])))
                                #     else:
                                #         weights = np.ones(detected_world_pts.shape[0])
                                #     preprocessing_line = WeightedSplineLine(bev_obj, detected_world_pts, weights,
                                #                                             model_order=world_line_postprocess_points_model_order,
                                #                                             min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                #                                             min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                #                                             spline_s=world_line_postprocess_points_model_spline_mae_allowed,
                                #                                             id=rb.line_id
                                #                                             )
                                # else:
                                #     preprocessing_line = WorldLine(bev_obj, detected_world_pts, model_type=world_line_postprocess_points_model_type,  # "poly # "spline" # "poly_robust"
                                #                                    model_order=world_line_postprocess_points_model_order,
                                #                                    min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                #                                    min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                #                                    spline_s=world_line_postprocess_points_model_spline_mae_allowed,
                                #                                    id=rb_line_side_i
                                #                                    )

                                if world_line_postprocess_points_model_extrapolate:
                                    world_pts_x_limits = (0, bev_obj.outView[1])
                                else:
                                    # world_pts_x_limits = (max(0,np.min(detected_world_pts[:,0])), min(bev_obj.outView[1],np.max(detected_world_pts[:,0]))) # FIXME why min not below 0?
                                    world_pts_x_limits = (np.min(detected_world_pts[:, 0]), min(bev_obj.outView[1], np.max(detected_world_pts[:, 0])))
                                # world_pts_x_limits = tuple(bev_obj.outView[0:2]) # extrapolate whole line (doesn't work well)
                                # world_pts_x_limits = (0, bev_obj.outView[1]) # extrapolate whole line (doesn't work well)
                                n_postprocess_pts = 100 #20
                                postprocessed_x_pts = np.linspace(world_pts_x_limits[0], world_pts_x_limits[1], n_postprocess_pts)
                                postprocessed_y_pts = preprocessing_line.computeYGivenX(postprocessed_x_pts)
                                world_pts = np.insert(np.vstack((postprocessed_x_pts, postprocessed_y_pts)).T, 2, (0), axis=1)

                                # print(np.min(world_pts[:,0]))
                            world_line = WorldLineArcLengthParam(bev_obj, world_pts, model_type=world_model_type,
                                                                 model_order=world_model_order, model_lsq_reg=world_line_model_alp_lsq_reg,
                                                                 min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                                                 min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                                                 id=rb.line_id,
                                                                 recenter_first_pt=True,
                                                                 fixed_origin_x=fixed_origin_x)
                        else:
                            world_line_spline_s = world_pts.shape[0] * (world_line_spline_mae_allowed ** 2) if world_model_type == "spline" else None
                            world_line = WorldLine(bev_obj, world_pts, model_type=world_model_type,
                                                   model_order=world_model_order,
                                                   min_n_pts_for_fitting=world_line_min_n_pts_for_fitting,
                                                   min_m_range_for_fitting=world_line_min_m_range_for_fitting,
                                                   spline_s=world_line_spline_s,
                                                   id=rb.line_id)
                        world_lines = np.append(world_lines, world_line)
                    except Line.LineException as e:
                        print(e)

                    # line in front image
                    try:
                        front_pts = bev_obj.projectBevPointsToImagePoints(bev_line_points)
                        front_line = ImageLine(front_pts, model_type=model_type, model_order=model_order, id=rb.line_id)
                        front_lines = np.append(front_lines, front_line)
                    except Line.LineException as e:
                        print(e)
        if bev_obj is not None:
            return world_lines, front_lines, bev_lines
        # else:
        return bev_lines
