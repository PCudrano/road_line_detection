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

def load_gt(gt_folder, bag, cap_fps, smoothing_factors=None):
    if smoothing_factors is None:
        smoothing_factors = _SmoothingFactors()
    # gt_lines_csv_folder = configs.in_data_path
    # gt_bag_file_name = 'BAG' + configs.bag
    # if configs.bag == 8:
    #     gt_lines_csv_filenames = np.array([gt_lines_csv_folder + 'bordi_int_dritti.csv', gt_lines_csv_folder + 'bordi_ext_dritti.csv'])  # BAG 8 only
    # else:
    #     gt_lines_csv_filenames = np.array([gt_lines_csv_folder + 'bordi_int.csv', gt_lines_csv_folder + 'bordi_ext.csv'])

    gt_lines_gps_coords_m, gt_coords_m_ref, gt_lines_gps_coords_m_interp_fcn, gt_lines_abs_orientation_interp_fcn = load_track_gt(gt_folder, bag, smoothing_factors) # TODO
    gt_traj_timestamps, gt_traj_gps_coords, gt_traj_gps_coords_m, gt_traj_gps_coords_m_interp_fcn, gt_traj_heading_interp_fcn, gt_traj_lateral_offset_interp_fcn, gps_data_abs_orientation_interp_fcn = load_trajectory_gt(gt_folder, bag, cap_fps, gt_coords_m_ref, smoothing_factors) # TODO

    return gt_lines_gps_coords_m, gt_coords_m_ref, gt_lines_gps_coords_m_interp_fcn, gt_lines_abs_orientation_interp_fcn, gt_traj_timestamps, gt_traj_gps_coords, gt_traj_gps_coords_m, gt_traj_gps_coords_m_interp_fcn, gps_data_abs_orientation_interp_fcn, gt_traj_heading_interp_fcn, gt_traj_lateral_offset_interp_fcn

#################################################################################
#################################### Private ####################################
#################################################################################

def load_track_gt(gt_folder, bag, smoothing_factors):
    # load gt files and start-end limits (to avoid issues on loop closure)
    gt_folder = gt_folder + 'lines/'
    if bag in [4, 8]:
        gt_lines_csv_filenames = np.array([gt_folder + 'bordi_int_dritti.csv', gt_folder + 'bordi_ext_dritti.csv'])  # BAG 8 only
    elif bag in [5]:
        gt_lines_csv_filenames = np.array([gt_folder + 'bordi_int_dritti_half.csv', gt_folder + 'bordi_ext_dritti_half.csv'])  # BAG 8 only
    elif bag <= 8: # else rec4
        gt_lines_csv_filenames = np.array([gt_folder + 'bordi_int.csv', gt_folder + 'bordi_ext.csv'])
    elif bag in [9,10,11]:
        gt_lines_csv_filenames = np.array([gt_folder + 'position_log_20191121-145553.csv', gt_folder + 'position_log_20191121-145556.csv'])
    else:
        raise NotImplementedError("Bag GT not found or not implemented")

    # Monza gt points to be removed from gt (and replaced by straight lines)
    if bag in [9,10,11]:
        start_inner = 400
        end_inner = -1895
        start_outer = 400
        end_outer = -1922
        bad_segments_inner = [[21665, 21680], [26555,26690], [28255, 28355]]  # [ [start1, end1], [start2, end2], ....]
        bad_segments_outer = [[21610, 21640], [26480, 26610], [28195, 28205]]  # [ [start1, end1], [start2, end2], ....]
    else:
        start_inner = 0
        end_inner = -1
        start_outer = 0
        end_outer = -1
        bad_segments_inner = []  # [ [start1, end1], [start2, end2], ....]
        bad_segments_outer = []  # [ [start1, end1], [start2, end2], ....]

    print(gt_lines_csv_filenames)

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
        # remove initial bad lines
        # if i == 0:
        #     gps_coords_cleaned = gps_coords_cleaned[1100:, :]
        # else:
        #     gps_coords_cleaned = gps_coords_cleaned[650:-450, :]

        gt_gps_coords.append(gps_coords_cleaned)

        # plt.figure(1);
        # plt.plot(gps_coords_cleaned[:,1], gps_coords_cleaned[:,0], '.k', markersize=.5)
        # plt.gca().set_aspect('equal', adjustable='box')

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
        # gt_gps_coords_m_interp_fcn_i = lambda i, spl=gt_gps_coords_m_interp_i: np.array(interpolate.splev(np.asarray(i), spl)).T  # evaluate spline in timestamps
        # gt_gps_coords_m_deriv_interp_fcn_i = lambda i, spl=gt_gps_coords_m_interp_i: np.array(interpolate.splev(np.asarray(i), spl, der=1)).T  # evaluate spline in timestamps
        # gt_gps_coords_abs_orientation_interp_fcn_i = lambda i, deriv_fcn=gt_gps_coords_m_deriv_interp_fcn_i: np.arctan2(*deriv_fcn(i)[:,::-1].T)
        # double lambda to fix value of some arguments (otherwise Python keeps last value in for loop)
        gt_gps_coords_m_interp_fcn_i = (lambda spl: (lambda i: np.array(interpolate.splev(np.asarray(i), spl)).T))(gt_gps_coords_m_interp_i)  # evaluate spline in timestamps
        gt_gps_coords_m_deriv_interp_fcn_i = (lambda spl: (lambda i: np.array(interpolate.splev(np.asarray(i), spl, der=1)).T))(gt_gps_coords_m_interp_i)  # evaluate spline in timestamps
        gt_gps_coords_abs_orientation_interp_fcn_i = (lambda deriv_fcn: (lambda i: np.arctan2(*deriv_fcn(i)[:,::-1].T) ))(gt_gps_coords_m_deriv_interp_fcn_i)

        gt_gps_coords_m_interp_fcn.append(gt_gps_coords_m_interp_fcn_i)
        gt_gps_coords_abs_orientation_interp_fcn.append(gt_gps_coords_abs_orientation_interp_fcn_i)

    return gt_gps_coords_m, gt_gps_coords_m_ref, gt_gps_coords_m_interp_fcn, gt_gps_coords_abs_orientation_interp_fcn

def load_trajectory_gt(gt_folder, bag, cap_fps, gt_gps_coords_m_ref, smoothing_factors):
    gt_bag_file_name = 'BAG' + str(bag)
    gt_folder = gt_folder + 'trajectory/'

    ### ground truth vehicle
    with open(gt_folder + gt_bag_file_name + '.csv', mode='r') as gps_csv_file:
        gps_csv_reader = csv.reader(gps_csv_file)
        gps_data = []
        gps_data_timestamps = []
        for row in gps_csv_reader:
            if bag <= 8:
                timestamp_part1 = str(int(row[2]))
                timestamp_part2 = str(int(row[3]))
                n_padding_zeros_timestamp_part2 = 9 - len(timestamp_part2)
                timestamp_part2 = '0' * n_padding_zeros_timestamp_part2 + timestamp_part2
                # timestamp = float(timestamp_part1 + timestamp_part2)
                # timestamp= decimal.Decimal(timestamp_part1 + timestamp_part2)
                timestamp = np.uint64(timestamp_part1 + timestamp_part2)
                # print("{:030d}".format(timestamp))
                gps_lat = row[0]
                gps_long = row[1]
                heading = row[4]
                lateral_offset = row[5]
            else:
                timestamp_part1 = str(int(float(row[0])))
                timestamp_part2 = str(int(float(row[1])))
                n_padding_zeros_timestamp_part2 = 9 - len(timestamp_part2)
                timestamp_part2 = '0' * n_padding_zeros_timestamp_part2 + timestamp_part2
                # timestamp = float(timestamp_part1 + timestamp_part2)
                # timestamp= decimal.Decimal(timestamp_part1 + timestamp_part2)
                timestamp = np.uint64(timestamp_part1 + timestamp_part2)
                # print("{:030d}".format(timestamp))
                gps_lat = row[2]
                gps_long = row[3]
                heading = row[4]
                lateral_offset = row[5]
            row = [gps_lat, gps_long, timestamp_part1, timestamp_part2, heading, lateral_offset]
            gps_data.append(row)
            gps_data_timestamps.append(timestamp)
    gps_data = np.array(gps_data).astype(np.float)
    gps_data_timestamps = np.array(gps_data_timestamps).astype(np.uint64)
    gps_data[:, 4] = np.degrees(gps_data[:, 4])

    # remove double points on same timestamp (keep first)
    gps_data_duplicate_timestamps_i = np.where(np.diff(gps_data_timestamps) == 0)[0] + 1
    gps_data_timestamps = np.delete(gps_data_timestamps, gps_data_duplicate_timestamps_i)
    gps_data = np.delete(gps_data, gps_data_duplicate_timestamps_i, axis=0)

    # interpolate gt
    # clean data out of range
    gps_data[ np.abs(gps_data[:,4]) > 90, 4] = np.nan
    gps_data[ np.abs(gps_data[:,5]) > 15, 5] = np.nan

    gps_data_heading_interp_fcn = interpolate.interp1d(gps_data_timestamps, -gps_data[:, 4], bounds_error=True, kind='linear')  # kind='cubic')
    gps_data_lateral_offset_interp_fcn = interpolate.interp1d(gps_data_timestamps, -gps_data[:, 5], bounds_error=True, kind='linear')  # kind='cubic')

    # interpolate trajectory gps coordinates
    # interpolate GPS coordinates append the starting x,y coordinates
    x = gps_data[:, 0]
    y = gps_data[:, 1]
    #x = np.r_[x, x[0]]
    #y = np.r_[y, y[0]]
    u = gps_data_timestamps #- gps_data_timestamps[0]
    #u = np.r_[u, u[-1] + cap_fps * 1e6] # simulate path closure
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    # fit spline for parametric curve, parameter: timestamp
    gps_data_latlong_interp, u_interp = interpolate.splprep([x, y], u=u, s=smoothing_factors.gt_pose_gps_smoothing_factor)#, per=True) #s=0, per=True) # s=0 no smoothing (1e-10 seems to be good already)
    gps_data_latlong_interp_fcn = lambda timestamps: np.array(interpolate.splev(np.asarray(timestamps), gps_data_latlong_interp)).T # evaluate spline in timestamps
    # xi, yi = interpolate.splev(gps_data_timestamps, gps_data_latlong_interp) # evaluate spline in timestamps
    # xiyi = gps_data_latlong_interp_fcn(gps_data_timestamps)
    # xi, yi = xiyi[:,0], xiyi[:,1]
    # debug plot the result
    # plt.figure(10)
    # plt.plot(gps_data[:,1], gps_data[:,0], '.', markersize=1)
    # plt.plot(yi, xi, '-')
    # # set same axis scale -> FIXME I'm assuming degrees lat and long have same distance!
    # plt.gca().set_aspect('equal', adjustable='box')

    # # plot ground truth heading
    # plt.figure(0)
    # plt.subplot(2,1,1); plt.plot((gps_data_timestamps_t_interval-t0)/ 1e9, gps_data_t_interval[:,4], '.', markersize=1); plt.title("GT Heading");
    # plt.subplot(2,1,2); plt.plot((gps_data_timestamps_t_interval-t0)/ 1e9, gps_data_t_interval[:,5], '.', markersize=1); plt.title("GT Lateral offset");

    # plot ground truth ego-pose
    # plt.figure(1);
    # plt.plot(gps_data_t_interval[:,1], gps_data_t_interval[:,0], '.r', markersize=1)
    # plt.plot(gps_data_t_interval[0,1], gps_data_t_interval[0,0], '^y', markersize=3)
    # plt.plot(gps_data_t_interval[-1,1], gps_data_t_interval[-1,0], '^g', markersize=3)

    # GSP trajectory to ENU ref frame (x,y in meters)
    gps_data_latlong_m = np.array([list(geo.geodetic_to_enu(coord[0], coord[1], 0, gt_gps_coords_m_ref[0], gt_gps_coords_m_ref[1], 0))[0:2] for coord in gps_data[:,0:2]])
    # interpolate trajectory in ENU ref frame
    x = gps_data_latlong_m[:, 0]
    y = gps_data_latlong_m[:, 1]
    #x = np.r_[x, x[0]]
    #y = np.r_[y, y[0]]

    # smooth_kernel = np.ones(5) / 3
    # x = np.hstack([x[0], np.convolve(x, np.ones(3) / 3, 'valid'), x[-1]])
    # y = np.hstack([y[0], np.convolve(y, np.ones(3) / 3, 'valid'), y[-1]])
    # OR from scipy.ndimage import gaussian_filter1d; gaussian_filter1d(x, 3)

    u0 = gps_data_timestamps[0]
    u = (gps_data_timestamps - u0) / 1e11  #- gps_data_timestamps[0]
    #u = np.r_[u, u[-1] + cap_fps * 1e6] # simulate path closure
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    # fit spline for parametric curve, parameter: timestamp
    # gps_data_latlong_m_interp, _ = interpolate.splprep([x, y], u=u, s=0.15*len(x), per=True) # s=0, per=True) # s=0 no smoothing (1e-10 seems to be good already)
    gps_data_latlong_m_interp, _ = interpolate.splprep([x, y], u=u, s=smoothing_factors.gt_pose_enu_smoothing_mae_allowed * len(x))#, per=True)  # s=0, per=True) # s=0 no smoothing (1e-10 seems to be good already)
    gps_data_latlong_m_interp_fcn = lambda timestamps: np.array(interpolate.splev((np.asarray(timestamps)-u0)/1e11, gps_data_latlong_m_interp)).T # evaluate spline in timestamps
    gps_data_latlong_m_deriv_interp_fcn = lambda timestamps: np.array(interpolate.splev((np.asarray(timestamps)-u0)/1e11, gps_data_latlong_m_interp, der=1)).T # evaluate spline in timestamps
    gps_data_abs_orientation_interp_fcn = lambda timestamps: np.arctan2(*gps_data_latlong_m_deriv_interp_fcn((np.asarray(timestamps)).reshape((-1,1))[:,0])[:,::-1].T)

    return gps_data_timestamps, gps_data[:,0:2], gps_data_latlong_m[:,0:2], gps_data_latlong_m_interp_fcn, gps_data_heading_interp_fcn, gps_data_lateral_offset_interp_fcn, gps_data_abs_orientation_interp_fcn

def load_centerline_gt(gt_folder, bag, gt_gps_coords_m_ref, smoothing_factors):
    gt_folder = gt_folder + 'centerline/'
    gt_bag_file_name = 'BAG' + str(bag)
    gt_lines_csv_filename = gt_folder + gt_bag_file_name + '.csv'

    ### ground truth centerline
    data = []
    with open(gt_lines_csv_filename, 'r') as csvFile:
        reader = csv.reader(csvFile)  # csv.reader(csvFile)
        for i, row in enumerate(reader):
            gps_lat = row[0]
            gps_long = row[1]
            row_dict = {'latitude(degrees)': gps_lat,
                        'longitude(degrees)': gps_long,
                        'i': i}
            data.append(row_dict)
            # print(i, dict(row))
    csvFile.close()

    gps_coords = np.array([[d['latitude(degrees)'], d['longitude(degrees)']] for d in data]).astype(np.float64)

    # clean gps coords (remove 0,0)
    # gps_coords = gps_coords[ ~(gps_coords[:,0]==0.0 & gps_coords[:,0]==0.0), :]
    non_null_island_coords = [i for i in range(gps_coords.shape[0]) if not (gps_coords[i, 0] == 0.0 and gps_coords[i, 1] == 0.0)]
    gps_coords_cleaned = gps_coords[non_null_island_coords, :]
    # remove initial bad lines
    # if i == 0:
    #     gps_coords_cleaned = gps_coords_cleaned[1100:, :]
    # else:
    #     gps_coords_cleaned = gps_coords_cleaned[650:-450, :]

    # gt_gps_coords = gps_coords_cleaned

    # GPS coords to ENU (x,y in meters)
    gt_gps_coords_m = []
    gt_gps_coords_m_interp_fcn = []
    gt_gps_coords_abs_orientation_interp_fcn = []

    gt_gps_coords_m = np.array([list(geo.geodetic_to_enu(coord[0], coord[1], 0, gt_gps_coords_m_ref[0], gt_gps_coords_m_ref[1], 0))[0:2] for coord in gps_coords_cleaned])

    # interpolate lines in ENU ref frame
    x = gt_gps_coords_m[:, 0]
    y = gt_gps_coords_m[:, 1]
    #x = np.r_[x, x[0]]
    #y = np.r_[y, y[0]]
    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    # fit spline for parametric curve, parameter: timestamp
    gt_gps_coords_m_interp, _ = interpolate.splprep([x, y], u=np.linspace(0,1, x.shape[0]), s=smoothing_factors.gt_centerline_smoothing_mae_allowed * len(x))#, per=True)  # s=0 no smoothing (1e-10 seems to be good already)
    gt_gps_coords_m_interp_fcn = lambda i: np.array(interpolate.splev(np.asarray(i), gt_gps_coords_m_interp)).T  # evaluate spline in timestamps
    gt_gps_coords_m_deriv_interp_fcn = lambda i: np.array(interpolate.splev(np.asarray(i), gt_gps_coords_m_interp_fcn, der=1)).T  # evaluate spline in timestamps
    gt_gps_coords_abs_orientation_interp_fcn = lambda i: np.arctan2(*gt_gps_coords_m_deriv_interp_fcn(i)[:,::-1].T)

    return gt_gps_coords_m, gt_gps_coords_m_interp_fcn, gt_gps_coords_abs_orientation_interp_fcn

# def load_odom(odom_folder, bag, timestamps):
#     odom_file_name = 'BAG' + str(bag)
#     csv_filename = odom_folder + odom_file_name
#
#     odom_timestamps = []
#     odom_data = []
#     with open(csv_filename + '.csv', mode='r') as csvFile:
#         gps_csv_reader = csv.reader(csvFile)
#         for row in gps_csv_reader:
#             timestamp = np.uint64(row[3])
#             if timestamp > 0 and (len(odom_timestamps) == 0 or timestamp > odom_timestamps[-1]):
#                 pos_x = row[0]
#                 pos_y = row[1]
#                 heading = row[2]
#                 save_row = [pos_x, pos_y, heading]
#                 odom_timestamps.append(timestamp)
#                 odom_data.append(save_row)
#     odom_data = np.array(odom_data).astype(np.float)
#     odom_timestamps = np.array(odom_timestamps).astype(np.uint64)
#     # odom_data[:, 2] = np.degrees(odom_data[:, 2]) # NO, I need radians
#
#     # interpolate pose
#     x = odom_data[:, 0]
#     y = odom_data[:, 1]
#     z = odom_data[:, 2]
#     u = odom_timestamps
#     odom_data_pose_interp_orig, u_interp = interpolate.splprep([x, y, z], u=u, s=0, per=0) # s=0 no smoothing (1e-10 seems to be good already)
#     odom_data_pose_interp_fcn_orig = lambda t: np.array(interpolate.splev(np.asarray(t), odom_data_pose_interp_orig)).T # evaluate spline in timestamps
#
#     odom_displacements = []
#     # init
#     # odom_displacements.append([0,0,0])
#     # odom_data_pose_interp_fcn = odom_data_pose_interp_fcn_orig
#     odom_data_pose_interp = np.copy(odom_data_pose_interp_orig)
#     odom_data_pose_interp_fcn = lambda t: np.array(interpolate.splev(np.asarray(t), odom_data_pose_interp)).T  # evaluate spline in timestamps
#     prev_odom_data_pose_interp_fcn = odom_data_pose_interp_fcn
#     for i,t in enumerate(timestamps):
#         cur_odom_interp_values = odom_data_pose_interp_fcn(t)
#         if i > 0:
#             prev_odom_interp_values = prev_odom_data_pose_interp_fcn(timestamps[i-1])
#         else:
#             prev_odom_interp_values = np.array([0,0,0])
#
#         odometry_displacement_xy = rot2d(-cur_odom_interp_values[2]).dot((prev_odom_interp_values[0:2] - cur_odom_interp_values[0:2]).T).T
#         odometry_displacement_x, odometry_displacement_y = odometry_displacement_xy[0:2]
#         odometry_displacement_orient = np.unwrap([cur_odom_interp_values[2] - prev_odom_interp_values[2]])[0]
#
#         odom_displacements.append([odometry_displacement_x, odometry_displacement_y, odometry_displacement_orient])
#
#         if i % 20 == 0:
#             remaining_odom_data = odom_data_pose_interp_fcn(timestamps[i:])
#             rotated_xy = rot2d(-odometry_displacement_orient).dot(remaining_odom_data[:,0:2].T).T
#             x = rotated_xy[:, 0]
#             y = rotated_xy[:, 1]
#             z = remaining_odom_data[:,2] - odometry_displacement_orient
#             u = timestamps[i:]
#             odom_data_pose_interp, u_interp = interpolate.splprep([x, y, z], u=u, s=0, per=0)  # s=0 no smoothing (1e-10 seems to be good already)
#             odom_data_pose_interp_fcn = lambda timestamps: np.array(interpolate.splev(np.asarray(timestamps), odom_data_pose_interp)).T  # evaluate spline in timestamps
#
#     odom_displacements = np.asarray(odom_displacements)
#
#     return odom_timestamps, odom_data, odom_data_pose_interp_fcn_orig, odom_displacements

# def load_odom_old(odom_folder, bag, timestamps):
#     odom_file_name = 'BAG' + str(bag)
#     csv_filename = odom_folder + 'old/' + odom_file_name
#
#     odom_timestamps = []
#     odom_data = []
#     with open(csv_filename + '.csv', mode='r') as csvFile:
#         gps_csv_reader = csv.reader(csvFile)
#         for row in gps_csv_reader:
#             timestamp = np.uint64(row[3])
#             if timestamp > 0 and (len(odom_timestamps) == 0 or timestamp > odom_timestamps[-1]):
#                 pos_x = row[0]
#                 pos_y = row[1]
#                 heading = row[2]
#                 save_row = [pos_x, pos_y, heading]
#                 odom_timestamps.append(timestamp)
#                 odom_data.append(save_row)
#     odom_data = np.array(odom_data).astype(np.float)
#     odom_timestamps = np.array(odom_timestamps).astype(np.uint64)
#     # odom_data[:, 2] = np.degrees(odom_data[:, 2]) # NO, I need radians
#
#     # interpolate pose
#     x = odom_data[:, 0]
#     y = odom_data[:, 1]
#     z = odom_data[:, 2]
#     u = odom_timestamps
#     odom_data_pose_interp_orig, u_interp = interpolate.splprep([x, y, z], u=u, s=odom_smoothing_mae_allowed * len(x), per=0) # s=0 no smoothing (1e-10 seems to be good already)
#     odom_data_pose_interp_fcn_orig = lambda t: np.array(interpolate.splev(np.asarray(t), odom_data_pose_interp_orig)).T # evaluate spline in timestamps
#
#     odom_displacements = []
#     # init
#     # odom_displacements.append([0,0,0])
#     # odom_data_pose_interp_fcn = odom_data_pose_interp_fcn_orig
#     # odom_data_pose_interp = np.copy(odom_data_pose_interp_orig)
#     # odom_data_pose_interp_fcn = lambda t: np.array(interpolate.splev(np.asarray(t), odom_data_pose_interp)).T  # evaluate spline in timestamps
#     # prev_odom_data_pose_interp_fcn = odom_data_pose_interp_fcn
#     odom_pose_cur = odom_data_pose_interp_fcn_orig(timestamps)
#     # odom_displacements_prev = np.copy(odom_displacements_cur)
#     for i,t in enumerate(timestamps):
#         cur_odom_interp_values = odom_pose_cur[0]
#         if i > 0:
#             prev_odom_interp_values = odom_pose_prev_1 # odom_displacements_prev[0]
#         else:
#             prev_odom_interp_values = np.array([0,0,0])
#
#         odometry_displacement_xy = rot2d(-cur_odom_interp_values[2]).dot((prev_odom_interp_values[0:2] - cur_odom_interp_values[0:2]).T).T
#         odometry_displacement_x, odometry_displacement_y = odometry_displacement_xy[0:2]
#         odometry_displacement_orient = np.unwrap([cur_odom_interp_values[2] - prev_odom_interp_values[2]])[0]
#
#         odom_displacements.append([odometry_displacement_x, odometry_displacement_y, odometry_displacement_orient])
#
#         # odom_displacements_prev = np.copy(odom_displacements_cur)
#         odom_pose_prev_1 = odom_pose_cur[0,:]
#         odom_pose_cur = odom_pose_cur[1:, :]
#         if odom_pose_cur.shape[0] > 0:
#             # odom_pose_cur[:,0:2] =rot2d(-odometry_displacement_orient).dot((odom_pose_cur[:,0:2] - odom_pose_prev_1[0:2]).T).T
#             # odom_pose_cur[:,0:2] = odom_pose_prev_1[0:2] - rot2d(-odom_pose_prev_1[2]).dot((odom_pose_cur[:,0:2] - odom_pose_prev_1[0:2]).T).T
#             odom_pose_cur[:, 0:2] = rot2d(odom_pose_prev_1[2]-odom_pose_cur[0,2]).dot((odom_pose_cur[:, 0:2] - odom_pose_prev_1[0:2]).T).T
#             odom_pose_cur[:,2] = odom_pose_cur[:,2] - odom_pose_prev_1[2]
#
#     odom_displacements = np.asarray(odom_displacements)
#
#     return odom_timestamps, odom_data, odom_data_pose_interp_fcn_orig, odom_displacements

# WRONG!! This interpolates on the displacements, but this way if I sample with double freq I get double displacement, cause displacement depends on delta time, not on absolute time
# def load_odom(odom_folder, bag):
#     odom_file_name = 'BAG' + str(bag)
#     csv_filename = odom_folder + odom_file_name
#
#     odom_timestamps = []
#     odom_data = []
#     with open(csv_filename + '.csv', mode='r') as csvFile:
#         gps_csv_reader = csv.reader(csvFile)
#         for row in gps_csv_reader:
#             timestamp = np.uint64(row[3])
#             if timestamp > 0 and (len(odom_timestamps) == 0 or timestamp > odom_timestamps[-1]):
#                 delta_pos_x = row[0]
#                 delta_pos_y = row[1]
#                 delta_heading = row[2]
#                 save_row = [delta_pos_x, delta_pos_y, delta_heading]
#                 odom_timestamps.append(timestamp)
#                 odom_data.append(save_row)
#     odom_data = np.array(odom_data).astype(np.float)
#     odom_timestamps = np.array(odom_timestamps).astype(np.uint64)
#     # odom_data[:, 2] = np.degrees(odom_data[:, 2]) # NO, I need radians
#
#     # interpolate pose
#     x = odom_data[:, 0]
#     y = odom_data[:, 1]
#     z = odom_data[:, 2]
#     u = odom_timestamps
#     odom_data_pose_interp_orig, u_interp = interpolate.splprep([x, y, z], u=u, s=0, per=0) # s=0 no smoothing (1e-10 seems to be good already)
#     odom_data_pose_interp_fcn_orig = lambda t: np.array(interpolate.splev(np.asarray(t), odom_data_pose_interp_orig)).T # evaluate spline in timestamps
#
#     return odom_timestamps, odom_data, odom_data_pose_interp_fcn_orig

def load_odom(odom_folder, bag, smoothing_factors):
    odom_file_name = 'BAG' + str(bag)
    csv_filename = odom_folder + odom_file_name

    odom_timestamps = []
    odom_data = []
    with open(csv_filename + '.csv', mode='r') as csvFile:
        gps_csv_reader = csv.reader(csvFile)
        for row in gps_csv_reader:
            timestamp = np.uint64(row[3])
            if timestamp > 0 and (len(odom_timestamps) == 0 or timestamp > odom_timestamps[-1]):
                delta_pos_x = row[0]
                delta_pos_y = row[1]
                delta_heading = row[2]
                save_row = [delta_pos_x, delta_pos_y, delta_heading]
                odom_timestamps.append(timestamp)
                odom_data.append(save_row)
    odom_data = np.array(odom_data).astype(np.float)
    odom_timestamps = np.array(odom_timestamps).astype(np.uint64)
    # odom_data[:, 2] = np.degrees(odom_data[:, 2]) # NO, I need radians

    # create evaluation function: given 2 instant in time, returns displacements over the time occurred between them
    ## cumsum delta_x relative to vehicle to have sampled x rel to vehicle
    odom_data_cumsum = np.cumsum(odom_data, axis=0)
    ## interpolate cumsum to have interpolated x rel to vehicle
    odom_data_cumsum_interp, u_interp = interpolate.splprep([odom_data_cumsum[:,0], odom_data_cumsum[:,1], odom_data_cumsum[:,2]], u=odom_timestamps, s=smoothing_factors.odom_smoothing_mae_allowed * odom_data_cumsum.shape[0], per=0) # s=0 no smoothing (1e-10 seems to be good already)
    odom_data_cumsum_interp_fcn = lambda t: np.array(interpolate.splev(np.asarray(t), odom_data_cumsum_interp)).T  # evaluate spline in timestamps
    ## evaluation fcn
    odom_data_displacement_fcn = lambda tb, ta: np.multiply([-1,+1,+1], np.subtract(odom_data_cumsum_interp_fcn(tb), odom_data_cumsum_interp_fcn(ta)))

    # def evaluate_fcn(t_before, t_after):
    #     odom_sampled_t_before_i = np.argmin(np.where(odom_timestamps <= t_before))
    #     odom_sampled_t_after_i = np.argmin(np.where(odom_timestamps <= t_after))
    #     if odom_sampled_t_before_i == odom_sampled_t_after_i:
    #         if odom_sampled_t_before_i > 0:
    #             odom_sampled_t_before_i -= 1
    #         else:
    #             odom_sampled_t_after_i += 1
    #     estim_v = odom_data[odom_sampled_t_after_i]

    return odom_timestamps, odom_data, odom_data_displacement_fcn

