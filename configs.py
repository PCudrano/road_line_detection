#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
from configparser import ConfigParser
import codecs

from includes.utils import get_project_root
from includes.procedural.feature_extraction import FeatureExtraction
from includes.MovingWindowLineFollower import MovingWindowLineFollower

################################################################################
################################ Configurations ################################
################################################################################

# General
verbose = True

# Input data
bag = 9
rec = 4 if bag <= 8 else 5
in_data_path = get_project_root() + 'data/rec{}/'.format(rec)
in_video_path = in_data_path + '9_2019-11-22-14-57-10/' # frame start: 90
gt_data_path = get_project_root() + 'data/rec{}/ground_truth/'.format(rec)
odom_data_path = get_project_root() + 'data/rec{}/odom/'.format(rec)
read_from_frame = 1000 #10434#13081#10434 #750 #13906#2073 # 2068 #2005#0#1380#0#420#200#1950 #1150#1215#3340#5983#15000 #13081#100#2983#7960 #100 # 1465 #9600#7960#13000#13900#7960#8100#9600#7960#13900#11800 #13900 #13568# 13568 #7960 # 8610 # 13561 # 7960 # 13561 #13501 # 11980 #7960+700 #16431 #13834 #4300 #4056 # 3150 # 3200 #3150 # 0 # frame number2
read_to_frame = -1 #10435#13082#10435#423#-1#9958 #-1#-1 # 8610+2 #-1 # 15350+1#-1 # 13565#13564 #13501+2 #11985+1 #1491 # 640 # -1 # frame number | -1 (go to the end)
read_frame_step = 1
expected_fps = float(100 if bag <= 8 else 15) / read_frame_step

#### Camera calibration
camera_intrinsics_config_file = get_project_root() + 'data/calibration/SN18837.conf'
camera_extrinsics_config_file = get_project_root() + 'data/calibration/extrinsics3.conf'
camera_resolution = 'VGA'
camera_side = 'LEFT'
config_intrinsics_section = camera_side + "_CAM_" + camera_resolution
config_extrinsics_section = 'EXTRINSICS'
# load intrinsics
camera_config = ConfigParser()
with codecs.open(camera_intrinsics_config_file, 'r', encoding="utf-8-sig") as f:
    camera_config.readfp(f)
# load extrinsics
with codecs.open(camera_extrinsics_config_file, 'r', encoding="utf-8-sig") as f:
    camera_config.readfp(f)
camera_data = {
    'intrinsic': {
        'fx': camera_config.getfloat(config_intrinsics_section, 'fx'),
        'fy': camera_config.getfloat(config_intrinsics_section, 'fy'),
        'cx': camera_config.getfloat(config_intrinsics_section, 'cx'),
        'cy': camera_config.getfloat(config_intrinsics_section, 'cy')
    },
    'extrinsic': {
        'x': camera_config.getfloat(config_extrinsics_section, 'x'),
        'y': camera_config.getfloat(config_extrinsics_section, 'y'),
        'z': camera_config.getfloat(config_extrinsics_section, 'z'),
        'yaw': camera_config.getfloat(config_extrinsics_section, 'yaw'),
        'pitch': camera_config.getfloat(config_extrinsics_section, 'pitch'),
        'roll': camera_config.getfloat(config_extrinsics_section, 'roll')
    },
    'center_of_mass': {
        # center of mass of the car
        'x': camera_config.getfloat(config_extrinsics_section, 'cm_x'),
        'y': camera_config.getfloat(config_extrinsics_section, 'cm_y'),
        'z': camera_config.getfloat(config_extrinsics_section, 'cm_z')
    }
}

# BEV parameters
bev_out_view = [5, 30, -20, 20] # [x_min, x_max, y_min, y_max] in world coordinates
bev_out_image_size = [-1, 1000] # height, width; -1 maintains aspect ratio

# CNN params
cnn_id = 1 # 2 #
line_mask_th = 65 #80 # 65 # 30 #
line_interp_th = 250
line_mask_size_comp = 0.015
postprocess_cnn_features_mode = FeatureExtraction.CNN_FEATURES_POSTPROCESS_MORPH

# Line-following algorithm
window_size_ratio = np.array([0.03,0.05])
window_init_size_ratio = np.array([0.15,0.07])
if rec == 4:
    window_init_point_w = np.array([[1, 5, 0], [1, -5, 0]])
    red_search_window_init_point_bound_w = np.array([[5, 4.5], [4.5, 4]])
else:
    window_init_point_w = np.array([[1, 6, 0], [1, -6, 0]])
    red_search_window_init_point_bound_w = np.array([[7, 5.5], [5.5, 7]])
window_enlarge_delta_ratio = np.array([0.005, 0.01])
window_init_enlarge_delta_ratio = np.array([0.020, 0.020])
window_enlarge_type = MovingWindowLineFollower.WINDOW_ENLARGE_ADAPTIVE
window_max_resize = 6
window_size = np.round(window_size_ratio * bev_out_image_size[1]).astype(np.int)
window_init_point_w = window_init_point_w
window_init_size = np.round(window_init_size_ratio * bev_out_image_size[1]).astype(np.int)
red_search_window_init_point_bound_w = red_search_window_init_point_bound_w
window_enlarge_delta = np.round(window_enlarge_delta_ratio * bev_out_image_size[1]).astype(np.int)
window_init_enlarge_delta = np.round(window_init_enlarge_delta_ratio * bev_out_image_size[1]).astype(np.int)

# Subsampled points
world_line_model_order = 3
world_line_model_extrapolate = False
world_line_spline_mae_allowed = 2 # m

# Outputs
out_image_size = (1280, 720)
