#!/usr/bin/env python
#
# Author: Paolo Cudrano (archnnj)
#

################################################################################
################################### Imports ###################################
################################################################################

import numpy as np
import traceback
import cv2
import os
import csv
import time

import configs
from includes.Camera import Camera
from includes.Bev import Bev
from includes.utils import resize_image_fit_nocrop, get_project_root
from includes.plot_utils import plot_points_on_image, plot_gt_lines
from includes.input_manager import ImageInputManager, GroundTruthManager,  OdometryManager
from includes.output_manager import Display, VideoSaver, OutputSpecsCollection, CsvSaver
from includes.procedural.feature_extraction import FeatureExtraction
from includes.procedural.feature_point_selection import FeaturePointSelection


################################################################################
################################ Output configs ################################
################################################################################

out_timestamp = time.strftime("%Y%m%d_%H%M%S")

# csv
export_csv = True
export_path = get_project_root() + 'outputs/{:}/'.format(out_timestamp)

# video
save_video_out = True
video_out_max_n_frames = 500 # -1: no bound set

# Create output folder if not existing
if export_csv and not os.path.exists(export_path):
    os.makedirs(export_path)

################################################################################
##################################### Main #####################################
################################################################################

## Init Camera and Bev objects
camera = Camera(configs.camera_data)
bev_obj = Bev(camera, configs.bev_out_view, configs.bev_out_image_size)

## Setup input data
input_mgr = ImageInputManager(
        configs.in_video_path,
        from_frame=configs.read_from_frame,
        to_frame=configs.read_to_frame,
        step_frame=configs.read_frame_step)
gt_mgr = GroundTruthManager(configs.gt_data_path, configs.bag, input_mgr.get_fps())
vehicle_cm = np.array([configs.camera_data['center_of_mass']['x'],
                       configs.camera_data['center_of_mass']['y'],
                       configs.camera_data['center_of_mass']['z']])

## Setup output
# Output video setup
out_mgr = VideoSaver(
    OutputSpecsCollection({
        'collage_points': {"size": configs.out_image_size, "name": "collage_points"},
    }),
    fps=input_mgr.get_fps(),
    split_after_frames=video_out_max_n_frames,
    output_path=export_path,
    input_id=input_mgr.get_input_id(),
    enabled=save_video_out)
# Output csv setup
csv_out_mgr = CsvSaver(
    fields_names=['timestamp', 'frame_i', 'enu_pose', 'enu_ref', 'left_x', 'left_y', 'right_x', 'right_y', 'left_x_subsampled', 'left_y_subsampled', 'right_x_subsampled', 'right_y_subsampled'],
    output_path=export_path,
    input_id=input_mgr.get_input_id(),
    enabled=export_csv)

## Setup pipeline
FeatureExtraction.init(configs.line_mask_th, configs.line_mask_size_comp, cnn_id=configs.cnn_id)

while(input_mgr.has_frames()):
    try:
        # Read frame
        frame = input_mgr.next_frame()
        frame_read = (frame is not None)

        if frame_read:
            print("Frame {}".format(input_mgr.frame_i))

            # Compute BEV
            bev = bev_obj.computeBev(frame)

            # Feature Extraction with CNN
            front_feature_mask, front_feature_predictions, bev_feature_mask, bev_feature_predictions = \
                FeatureExtraction.extraction(frame, bev_obj, postprocess_mode=configs.postprocess_cnn_features_mode)
            # Feature points selection using WLF algorithm
            lines_pts_w, lines_pts_bev = FeaturePointSelection.feature_point_selection(
                    bev_feature_mask, bev_obj,
                    configs.window_init_point_w, configs.window_size,
                    configs.window_enlarge_delta, configs.window_enlarge_type,
                    configs.window_max_resize, configs.window_init_size,
                    configs.window_init_enlarge_delta, previous_world_lines=None)
            # line_pts_w = FeaturePointSelection.accumulate_prev_points(line_pts_w, ...)

            # Extracting subsampled points
            smoothed_lines_pts_w = []
            for line_pts_w in lines_pts_w:
                if line_pts_w.shape[0] > 0:
                    if configs.world_line_model_extrapolate:
                        smooth_s_lim = (0, bev_obj.outView[1])
                    else:
                        smooth_s_lim = (np.min(line_pts_w[:, 0]), min(bev_obj.outView[1], np.max(line_pts_w[:, 0])))
                    smoothed_lines_pts_w.extend(FeaturePointSelection.smooth_subsampling([line_pts_w],
                            spline_order=configs.world_line_model_order,
                            spline_mae_allowed=configs.world_line_spline_mae_allowed,
                            s_lim=smooth_s_lim, n_pts=100))
                else:
                    smoothed_lines_pts_w.append(np.array([]).reshape(-1,3))

            enu_pose = gt_mgr.get_pose(input_mgr.get_prev_timestamp())

            # Define variables to store
            timestamp = input_mgr.get_prev_timestamp()
            frame_i = input_mgr.frame_i - 1
            enu_pose = np.hstack(list([v.tolist() for v in gt_mgr.get_pose(input_mgr.get_prev_timestamp())]))
            enu_ref = np.array([v.tolist() for v in gt_mgr.get_enu_reference()])
            left_x = lines_pts_w[0][:, 0] if lines_pts_w is not None and lines_pts_w[0] is not None else None
            left_y = lines_pts_w[0][:, 1] if lines_pts_w is not None and lines_pts_w[0] is not None else None
            right_x = lines_pts_w[1][:, 0] if lines_pts_w is not None and lines_pts_w[1] is not None else None
            right_y = lines_pts_w[1][:, 1] if lines_pts_w is not None and lines_pts_w[1] is not None else None
            left_x_subsampled = smoothed_lines_pts_w[0][:, 0] \
                if smoothed_lines_pts_w is not None and smoothed_lines_pts_w[0] is not None else None
            left_y_subsampled = smoothed_lines_pts_w[0][:, 1] \
                if smoothed_lines_pts_w is not None and smoothed_lines_pts_w[0] is not None else None
            right_x_subsampled = smoothed_lines_pts_w[1][:, 0] \
                if smoothed_lines_pts_w is not None and smoothed_lines_pts_w[1] is not None else None
            right_y_subsampled = smoothed_lines_pts_w[1][:, 1] \
                if smoothed_lines_pts_w is not None and smoothed_lines_pts_w[1] is not None else None

            ## Compose output images (overlay points and compose mosaics of front+BEV)
            if configs.verbose:
                front_points = frame.copy()
                bev_points = bev.copy()
                for pts_w in lines_pts_w:
                    if pts_w is not None:
                        pts_bev = bev_obj.projectWorldPointsToBevPoints(pts_w)
                        pts_front = bev_obj.projectBevPointsToImagePoints(pts_bev)
                        # plot_points_on_image(front_points, pts_front, color=(0, 0, 255), thickness=5)
                        plot_points_on_image(front_points, pts_front, color=(254, 198, 47), thickness=5)
                        plot_points_on_image(bev_points, pts_bev, color=(254, 198, 47), thickness=5)
                # Composing result mosaic image
                bev_points_resized = cv2.resize(
                        bev_points, dsize=None,
                        fx=float(front_points.shape[0]) / bev_points.shape[0],
                        fy=float(front_points.shape[0]) / bev_points.shape[0])
                collage = np.hstack((
                    front_points,
                    bev_points_resized
                ))
                collage_resized = resize_image_fit_nocrop(collage, dsize=configs.out_image_size)

                front_smoothed_points = frame.copy()
                bev_smoothed_points = bev.copy()
                for pts_w in smoothed_lines_pts_w:
                    if pts_w is not None:
                        pts_bev = bev_obj.projectWorldPointsToBevPoints(pts_w)
                        pts_front = bev_obj.projectBevPointsToImagePoints(pts_bev)
                        plot_points_on_image(front_smoothed_points, pts_front, color=(254, 198, 47), thickness=5)
                        plot_points_on_image(bev_smoothed_points, pts_bev, color=(254, 198, 47), thickness=5)
                bev_smoothed_points_resized = cv2.resize(
                    bev_smoothed_points, dsize=None,
                    fx=float(front_points.shape[0]) / bev_smoothed_points.shape[0],
                    fy=float(front_points.shape[0]) / bev_smoothed_points.shape[0])
                collage_smoothed = np.hstack((
                    front_smoothed_points,
                    bev_smoothed_points_resized
                ))
                collage_smoothed_resized = resize_image_fit_nocrop(collage_smoothed, dsize=configs.out_image_size)

                # Show output
                Display._show_image(collage_resized, window_name="Output",
                                    window_size=configs.out_image_size, wait_sec=5)
                Display._show_image(collage_smoothed_resized, window_name="Output smoothed",
                                    window_size=configs.out_image_size, wait_sec=5)
            # Save points in csv
            if export_csv:
                csv_out_mgr.write({
                    'timestamp': timestamp,
                    'frame_i': frame_i,
                    'enu_pose': enu_pose.tolist(),
                    'enu_ref': enu_ref.tolist(),
                    'left_x': left_x.tolist() if left_x is not None else '[]',
                    'left_y': left_y.tolist() if left_y is not None else '[]',
                    'right_x': right_x.tolist() if right_x is not None else '[]',
                    'right_y': right_y.tolist() if right_y is not None else '[]',
                    'left_x_subsampled': left_x_subsampled.tolist() if left_x_subsampled is not None else '[]',
                    'left_y_subsampled': left_y_subsampled.tolist() if left_y_subsampled is not None else '[]',
                    'right_x_subsampled': right_x_subsampled.tolist() if right_x_subsampled is not None else '[]',
                    'right_y_subsampled': right_y_subsampled.tolist() if right_y_subsampled is not None else '[]'
                })

            # Save output
            if save_video_out:
                out_mgr.write('collage_points', collage_resized)
        else:
            raise Exception("Missed frame")
    except:
        traceback.print_exc()
        print("Exception occurred.\nTerminating...")
        break

# Close video objects
input_mgr.close()
out_mgr.close_all()
csv_out_mgr.close()
