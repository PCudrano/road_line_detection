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
rec = {
    'letter': 'B',  # 'A': Arese; 'B': Monza
    'run': 1  # 1,2
}
rec['id'] = rec['letter'] + str(rec['run'])
in_data_path = get_project_root() + 'data/'
in_video_path = in_data_path + 'imgs/' + rec['id'] + '/'
gt_data_path = get_project_root() + 'data/gt/'
odom_data_path = get_project_root() + 'data/odom/'
read_from_frame = 0
read_to_frame = -1
read_frame_step = 1
expected_fps = float(100 if rec['letter'] == 'A' else 15) / read_frame_step

#### Camera calibration
camera_config_file = get_project_root() + 'data/calibration/camera.conf'
config_intrinsics_section = "INTRINSICS"
config_extrinsics_section = 'EXTRINSICS'
camera_config = ConfigParser()
with codecs.open(camera_config_file, 'r', encoding="utf-8-sig") as f:
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
        'x': 0,
        'y': 0,
        'z': 0,
    }
}

# BEV parameters
bev_out_view = [0, 30, -20, 20] # [x_min, x_max, y_min, y_max] in world coordinates
bev_out_image_size = [-1, 1000] # height, width; -1 maintains aspect ratio

# CNN params
cnn_id = 1 # 1,2
line_mask_th = 65
line_interp_th = 250
line_mask_size_comp = 0.015
postprocess_cnn_features_mode = FeatureExtraction.CNN_FEATURES_POSTPROCESS_MORPH

# Line-following algorithm
window_size_ratio = np.array([0.03,0.05])
if rec['letter'] == 'A':
    window_init_point_w = np.array([[1, 5, 0], [1, -5, 0]])
    red_search_window_init_point_bound_w = np.array([[5, 4.5], [4.5, 4]])
    window_init_size_ratio = np.array([0.15, 0.07])
else:
    window_init_point_w = np.array([[1, 6, 0], [1, -6, 0]])
    red_search_window_init_point_bound_w = np.array([[7, 5.5], [5.5, 7]])
    window_init_size_ratio = np.array([0.15, 0.07])
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
