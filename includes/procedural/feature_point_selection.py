#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
from math import pi
import cv2
from includes.MovingWindowLineFollower import MovingWindowLineFollower
from includes.procedural.fitting import Fitting

class FeaturePointSelection:
    line_distance_threshold = 5
    min_m_range_for_fitting = 2

    class FeaturePointSelectionException(Exception):
        pass

    @classmethod
    def feature_point_selection(cls, bev_feature_mask, bev_obj,
                                window_init_point_w, window_size, window_enlarge_delta, window_enlarge_type,
                                window_max_resize, window_init_size, window_init_enlarge_delta, previous_world_lines=None):
        windows_init_points = bev_obj.projectWorldPointsToBevPoints(window_init_point_w)
        wlf = MovingWindowLineFollower(
                windows_init_points,
                window_init_size=window_init_size,
                window_init_enlarge_delta=window_init_enlarge_delta,
                window_size=window_size,
                window_enlarge_delta=window_enlarge_delta,
                window_enlarge_type=window_enlarge_type,
                window_max_resize=window_max_resize,
                window_init_dir=pi/2, # upwards
                window_dir_allowed_range=np.array([0+np.finfo(np.float64).eps,pi-np.finfo(np.float64).eps]), # (0,pi), extremes excluded
                # centroid_type=CENTROID_CLOSEST_WITH_MIN_AREA,
                # centroid_min_area_percentage=0.01,
                line_model_type=None, line_model_order=None,
                line_sides=None,
                use_ids = False, refine_with_prev_line=False)
        # # if previous world lines passed, convert them into bev lines
        if previous_world_lines is not None:
            previous_bev_lines = np.array([l.getBevLineProxy(allow_extrapolation=True) for l in previous_world_lines])
        else:
            previous_bev_lines = None
        wlf.newFrame(bev_feature_mask,
                     new_windows_init_points=windows_init_points,
                     line_sides=None,
                     previous_lines=previous_bev_lines)
        line_builders = wlf._follow_line()
        line_points_bev = []
        line_points_w = []
        for rb in line_builders.line_builders:
            bev_pts = rb.getPoints()
            line_points_bev.append(bev_pts)
            line_points_w.append(bev_obj.projectBevPointsToWorldGroundPlane(bev_pts))
        return line_points_w, line_points_bev

    @classmethod
    def smooth_subsampling(cls, line_points_w, spline_order, spline_mae_allowed, s_lim, n_pts):
        # if world_line_postprocess_points_model_extrapolate:
        #     world_pts_x_limits = (0, bev_obj.outView[1])
        # else:
        #     world_pts_x_limits = (np.min(detected_world_pts[:, 0]), min(bev_obj.outView[1], np.max(detected_world_pts[:, 0])))
        smooth_line_points = []
        for i, pts in enumerate(line_points_w):
            try:
                cls._check_enough_meters_for_fitting(pts)
                spline_smoothing_factor = pts.shape[0] * (spline_mae_allowed ** 2)
                smooth_fit = Fitting.fit_spline(pts[:,0], pts[:,1], spline_order, spline_smoothing_factor)
                subsampled_x_pts = np.linspace(s_lim[0], s_lim[1], n_pts)
                subsampled_y_pts = smooth_fit(subsampled_x_pts)
                subsampled_pts = np.insert(np.vstack((subsampled_x_pts, subsampled_y_pts)).T, 2, (0), axis=1)
            except:
                subsampled_pts = None
            smooth_line_points.append(subsampled_pts)
        return smooth_line_points

    @classmethod
    def accumulate_prev_points(cls, line_points_w, displaced_prev_points):
        # TODO call before smooth_subsampling
        acc_line_points = []
        for i, pts in enumerate(line_points_w):
            pts_to_add = displaced_prev_points[i]
            pts_to_add_filtered = cls._discard_outliers(pts_to_add)
            acc_pts = np.concatenate((pts, pts_to_add_filtered), axis=0)
            acc_line_points.append(acc_pts)
        return acc_line_points

    @classmethod
    def _discard_outliers(cls, line_points_w):
        try:
            cls._check_enough_meters_for_fitting(line_points_w)
            smooth_fit = Fitting.fit_poly_robust(line_points_w[:,0], line_points_w[:,1], 3)
            filter_x_limits = (np.min(line_points_w[:, 0]), np.max(line_points_w[:, 0]))
            n_filter_pts = 500
            filter_x_pts = np.linspace(filter_x_limits[0], filter_x_limits[1], n_filter_pts)
            filter_y_pts = smooth_fit(filter_x_pts)
            preprocessing_line_distance_filter_pts = np.vstack((filter_x_pts, filter_y_pts)).T
            points_filter_distance = np.array([np.min(np.linalg.norm(pt[0:2] - preprocessing_line_distance_filter_pts, axis=1)) for pt in line_points_w])
            # filter based on distance
            filtered_line_points_w = line_points_w[points_filter_distance <= cls.line_distance_threshold, :]
            return filtered_line_points_w
        except:
            return np.array([])

    @classmethod
    def _check_enough_meters_for_fitting(cls, points):
    # def _has_enough_meters_for_fitting(cls, points):
        # check that range of points is larger than min_m_range_for_fitting
        pts_x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        # return pts_x_range >= cls.min_m_range_for_fitting
        if pts_x_range < cls.min_m_range_for_fitting:
            raise cls.FeaturePointSelectionException(
                    "FeaturePointSelection: x points range not large enough: given range {} m, needed range {} m"
                    .format(pts_x_range, cls.min_m_range_for_fitting))
