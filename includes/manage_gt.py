#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import csv
import includes.geo as geo
from scipy import interpolate

class _SmoothingFactors(object):
    _gt_track_smoothing_factor_default = 0  # m
    _gt_pose_gps_smoothing_factor_default = 10  # gps deg
    _gt_pose_enu_smoothing_mae_allowed_default =  0.03 # m
    _gt_centerline_smoothing_mae_allowed_default = 0  # m
    _odom_smoothing_mae_allowed_default = 0  # m

    def __init__(self, track_s=None, pose_gps_s=None, pose_enu_mae=None, centerline_enu_mae=None, odom_s=None):
        self.gt_track_smoothing_factor = track_s if track_s is not None else self._gt_track_smoothing_factor_default
        self.gt_pose_gps_smoothing_factor = pose_gps_s if pose_gps_s is not None else self._gt_pose_gps_smoothing_factor_default
        self.gt_pose_enu_smoothing_mae_allowed = pose_enu_mae if pose_enu_mae is not None else self._gt_pose_enu_smoothing_mae_allowed_default
        self.gt_centerline_smoothing_mae_allowed = centerline_enu_mae if centerline_enu_mae is not None else self._gt_centerline_smoothing_mae_allowed_default
        self.odom_smoothing_mae_allowed = odom_s if odom_s is not None else self._odom_smoothing_mae_allowed_default

def load_gt(gt_folder, rec, cap_fps, smoothing_factors=None, traj_only=False):
    if smoothing_factors is None:
        smoothing_factors = _SmoothingFactors()
    if not traj_only:
        gt_lines_gps_coords_m, gt_coords_m_ref, gt_lines_gps_coords_m_interp_fcn, gt_lines_abs_orientation_interp_fcn = load_track_gt(gt_folder, rec, smoothing_factors) # TODO
    else:
        gt_lines_gps_coords_m, gt_coords_m_ref, gt_lines_gps_coords_m_interp_fcn, gt_lines_abs_orientation_interp_fcn = None, None, None, None
    gt_traj_timestamps, gt_traj_gps_coords, gt_traj_gps_coords_m, gt_traj_gps_coords_m_interp_fcn, gt_traj_heading_interp_fcn, gt_traj_lateral_offset_interp_fcn, gps_data_abs_orientation_interp_fcn, gt_coords_m_ref = load_trajectory_gt(gt_folder, rec, cap_fps, gt_coords_m_ref, smoothing_factors, traj_only=traj_only) # TODO

    return gt_lines_gps_coords_m, gt_coords_m_ref, gt_lines_gps_coords_m_interp_fcn, gt_lines_abs_orientation_interp_fcn, gt_traj_timestamps, gt_traj_gps_coords, gt_traj_gps_coords_m, gt_traj_gps_coords_m_interp_fcn, gps_data_abs_orientation_interp_fcn, gt_traj_heading_interp_fcn, gt_traj_lateral_offset_interp_fcn

#################################################################################
#################################### Private ####################################
#################################################################################

def load_track_gt(gt_folder, rec, smoothing_factors):
    gt_lines_csv_filenames = np.array([gt_folder + 'circuits_lines/'+ rec['id'] +'/inner_line.csv',
                                       gt_folder + 'circuits_lines/'+ rec['id'] +'/outer_line.csv'])
    start_inner = 0
    end_inner = -1
    start_outer = 0
    end_outer = -1
    bad_segments_inner = []  # [ [start1, end1], [start2, end2], ....]
    bad_segments_outer = []  # [ [start1, end1], [start2, end2], ....]
    ### ground truth track
    data_files = []
    for csv_filename in gt_lines_csv_filenames:
        data = []
        with open(csv_filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile)  # csv.reader(csvFile)
            for i, row in enumerate(reader):
                row_dict = dict(row)
                row_dict['i'] = i
                data.append(row_dict)
                # print(i, dict(row))
        data_files.append(data)
        csvFile.close()
    gt_gps_coords = []
    for i, data in enumerate(data_files):
        gps_coords_m = np.array([[d['latitude(degrees)'], d['longitude(degrees)']] for d in data]).astype(np.float64)
        # clean gps coords (remove 0,0)
        # gps_coords = gps_coords[ ~(gps_coords[:,0]==0.0 & gps_coords[:,0]==0.0), :]
        non_null_island_coords = [i for i in range(gps_coords_m.shape[0]) if not (gps_coords_m[i, 0] == 0.0 and gps_coords_m[i, 1] == 0.0)]
        gps_coords_cleaned = gps_coords_m[non_null_island_coords, :]
        gt_gps_coords.append(gps_coords_cleaned)
    # GPS coords to ENU (x,y in meters)
    gt_gps_coords_m = []
    gt_gps_coords_m_interp_fcn = []
    gt_gps_coords_abs_orientation_interp_fcn = []
    gt_gps_coords_m_ref = [np.min([np.min(gps_coords_m[:, 0]) for gps_coords_m in gt_gps_coords]) - 0.00001,
                           np.min([np.min(gps_coords_m[:, 1]) for gps_coords_m in gt_gps_coords]) - 0.00001]
    for j,gps_coords_m in enumerate(gt_gps_coords):
        enu_coords = np.array([list(geo.geodetic_to_enu(coord[0], coord[1], 0, gt_gps_coords_m_ref[0], gt_gps_coords_m_ref[1], 0))[0:2] for coord in gps_coords_m])
        gt_gps_coords_m.append(enu_coords)
        if j==0: # inner
            start = start_inner
            end = end_inner
            bad_segments = bad_segments_inner
        elif j==1: # outer
            start = start_outer
            end = end_outer
            bad_segments = bad_segments_outer
        # # correct noise/outliers
        enu_coords_filtered = np.array([]).reshape((-1, 2))
        prev_start = start
        pt_density = enu_coords.shape[0] / np.sum(np.sqrt(np.sum(np.square(np.diff(enu_coords, axis=0)), axis=1)))
        for bs_i in range(len(bad_segments)):
            # from prev_start to beginning of next bad segment
            enu_coords_filtered = np.vstack((enu_coords_filtered, enu_coords[prev_start:bad_segments[bs_i][0], :]))
            # add straight points in bad segment: x = (1-lambda) * x1 + lambda * x2 = x1 + lambda * (x2-x1), lambda in 0,1
            xy1 = enu_coords[bad_segments[bs_i][0], :]
            xy2 = enu_coords[bad_segments[bs_i][1], :]
            ld = np.linspace(0, 1, int(np.ceil(pt_density * np.sqrt(np.sum(np.square(np.subtract(xy1, xy2)))))))
            new_pts = xy1 + ld[:, np.newaxis] * (xy2 - xy1).T  # convex combinations
            enu_coords_filtered = np.vstack((enu_coords_filtered, new_pts))
            # end of bad segment, where to start in next iter
            prev_start = bad_segments[bs_i][1]
        # from last segment to the end (Obs: if no bad segments, prev_start=start)
        enu_coords_filtered = np.vstack((enu_coords_filtered, enu_coords[prev_start:end, :]))
        # interpolate lines in ENU ref frame
        x = enu_coords_filtered[:, 0]
        y = enu_coords_filtered[:, 1]
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        # fit spline for parametric curve, parameter: timestamp
        gt_gps_coords_m_interp_i, _ = interpolate.splprep([x, y], u=np.linspace(0,1, x.shape[0]), s=smoothing_factors.gt_track_smoothing_factor, per=True)  # s=0 no smoothing (1e-10 seems to be good already)
        gt_gps_coords_m_interp_fcn_i = (lambda spl: (lambda i: np.array(interpolate.splev(np.asarray(i), spl)).T))(gt_gps_coords_m_interp_i)  # evaluate spline in timestamps
        gt_gps_coords_m_deriv_interp_fcn_i = (lambda spl: (lambda i: np.array(interpolate.splev(np.asarray(i), spl, der=1)).T))(gt_gps_coords_m_interp_i)  # evaluate spline in timestamps
        gt_gps_coords_abs_orientation_interp_fcn_i = (lambda deriv_fcn: (lambda i: np.arctan2(*deriv_fcn(i)[:,::-1].T) ))(gt_gps_coords_m_deriv_interp_fcn_i)
        gt_gps_coords_m_interp_fcn.append(gt_gps_coords_m_interp_fcn_i)
        gt_gps_coords_abs_orientation_interp_fcn.append(gt_gps_coords_abs_orientation_interp_fcn_i)
    return gt_gps_coords_m, gt_gps_coords_m_ref, gt_gps_coords_m_interp_fcn, gt_gps_coords_abs_orientation_interp_fcn

def load_trajectory_gt(gt_folder, rec, cap_fps, gt_gps_coords_m_ref, smoothing_factors, traj_only=False):
    gt_bag_file_name = rec['id']
    gt_folder = gt_folder + 'vehicle_pose/'
    ### ground truth vehicle
    with open(gt_folder + gt_bag_file_name + '.csv', mode='r') as gps_csv_file:
        gps_csv_reader = csv.reader(gps_csv_file)
        next(gps_csv_reader, None)  # skip the headers
        gps_data = []
        gps_data_timestamps = []
        for row in gps_csv_reader:
            timestamp = np.uint64(row[0])
            gps_lat = row[1]
            gps_long = row[2]
            if not traj_only:
                heading = row[3]
                lateral_offset = row[4]
            else:
                heading = 0
                lateral_offset = 0
            row = [gps_lat, gps_long, timestamp, heading, lateral_offset]
            gps_data.append(row)
            gps_data_timestamps.append(timestamp)
    gps_data = np.array(gps_data).astype(np.float)
    gps_data_timestamps = np.array(gps_data_timestamps).astype(np.uint64)
    gps_data[:, 3] = np.degrees(gps_data[:, 3])
    # remove double points on same timestamp (keep first)
    gps_data_duplicate_timestamps_i = np.where(np.diff(gps_data_timestamps) == 0)[0] + 1
    gps_data_timestamps = np.delete(gps_data_timestamps, gps_data_duplicate_timestamps_i)
    gps_data = np.delete(gps_data, gps_data_duplicate_timestamps_i, axis=0)
    # remove points in the past (issue for bag 12-13)
    gps_data_past_timestamps_i = np.where(gps_data_timestamps < gps_data_timestamps[0])[0]
    gps_data_timestamps = np.delete(gps_data_timestamps, gps_data_past_timestamps_i)
    gps_data = np.delete(gps_data, gps_data_past_timestamps_i, axis=0)
    # interpolate gt
    # clean data out of range
    gps_data[ np.abs(gps_data[:,3]) > 90, 3] = np.nan
    gps_data[ np.abs(gps_data[:,4]) > 15, 4] = np.nan
    gps_data_heading_interp_fcn = interpolate.interp1d(gps_data_timestamps, -gps_data[:, 3], bounds_error=True, kind='linear')  # kind='cubic')
    gps_data_lateral_offset_interp_fcn = interpolate.interp1d(gps_data_timestamps, -gps_data[:, 4], bounds_error=True, kind='linear')  # kind='cubic')
    # interpolate trajectory gps coordinates
    # interpolate GPS coordinates append the starting x,y coordinates
    x = gps_data[:, 0]
    y = gps_data[:, 1]
    u = gps_data_timestamps #- gps_data_timestamps[0]
    # GSP trajectory to ENU ref frame (x,y in meters)
    if traj_only and gt_gps_coords_m_ref is None:
        gt_gps_coords_m_ref = np.median(gps_data[:, 0:2], axis=0)
    gps_data_latlong_m = np.array([list(geo.geodetic_to_enu(coord[0], coord[1], 0, gt_gps_coords_m_ref[0], gt_gps_coords_m_ref[1], 0))[0:2] for coord in gps_data[:,0:2]])
    # interpolate trajectory in ENU ref frame
    x = gps_data_latlong_m[:, 0]
    y = gps_data_latlong_m[:, 1]
    u0 = gps_data_timestamps[0]
    u = (gps_data_timestamps - u0) / 1e11  #- gps_data_timestamps[0]
    gps_data_latlong_m_interp, _ = interpolate.splprep([x, y], u=u, s=smoothing_factors.gt_pose_enu_smoothing_mae_allowed * len(x))#, per=True)  # s=0, per=True) # s=0 no smoothing (1e-10 seems to be good already)
    gps_data_latlong_m_interp_fcn = lambda timestamps: np.array(interpolate.splev((np.asarray(timestamps)-u0)/1e11, gps_data_latlong_m_interp)).T # evaluate spline in timestamps
    gps_data_latlong_m_deriv_interp_fcn = lambda timestamps: np.array(interpolate.splev((np.asarray(timestamps)-u0)/1e11, gps_data_latlong_m_interp, der=1)).T # evaluate spline in timestamps
    gps_data_abs_orientation_interp_fcn = lambda timestamps: np.arctan2(*gps_data_latlong_m_deriv_interp_fcn((np.asarray(timestamps)).reshape((-1,1))[:,0])[:,::-1].T)

    return gps_data_timestamps, gps_data[:,0:2], gps_data_latlong_m[:,0:2], gps_data_latlong_m_interp_fcn, gps_data_heading_interp_fcn, gps_data_lateral_offset_interp_fcn, gps_data_abs_orientation_interp_fcn, gt_gps_coords_m_ref

def load_odom(odom_folder, rec, smoothing_factors):
    gt_bag_file_name = rec['id']
    csv_filename = odom_folder + gt_bag_file_name
    odom_timestamps = []
    odom_data = []
    with open(csv_filename + '.csv', mode='r') as csvFile:
        gps_csv_reader = csv.reader(csvFile)
        for row in gps_csv_reader:
            timestamp = np.uint64(row[0])
            if timestamp > 0 and (len(odom_timestamps) == 0 or timestamp > odom_timestamps[-1]):
                delta_pos_x = row[1]
                delta_pos_y = row[2]
                delta_heading = row[3]
                save_row = [delta_pos_x, delta_pos_y, delta_heading]
                odom_timestamps.append(timestamp)
                odom_data.append(save_row)
    odom_data = np.array(odom_data).astype(np.float)
    odom_timestamps = np.array(odom_timestamps).astype(np.uint64)
    # create evaluation function: given 2 instant in time, returns displacements over the time occurred between them
    ## cumsum delta_x relative to vehicle to have sampled x rel to vehicle
    odom_data_cumsum = np.cumsum(odom_data, axis=0)
    ## interpolate cumsum to have interpolated x rel to vehicle
    odom_data_cumsum_interp, u_interp = interpolate.splprep([odom_data_cumsum[:,0], odom_data_cumsum[:,1], odom_data_cumsum[:,2]], u=odom_timestamps, s=smoothing_factors.odom_smoothing_mae_allowed * odom_data_cumsum.shape[0], per=0) # s=0 no smoothing (1e-10 seems to be good already)
    odom_data_cumsum_interp_fcn = lambda t: np.array(interpolate.splev(np.asarray(t), odom_data_cumsum_interp)).T  # evaluate spline in timestamps
    ## evaluation fcn
    odom_data_displacement_fcn = lambda tb, ta: np.multiply([-1,+1,+1], np.subtract(odom_data_cumsum_interp_fcn(tb), odom_data_cumsum_interp_fcn(ta)))
    return odom_timestamps, odom_data, odom_data_displacement_fcn

