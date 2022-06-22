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
from matplotlib import pyplot as plt
from tqdm import tqdm

import configs
from includes.Camera import Camera
from includes.Bev import Bev
from includes.utils import resize_image_fit_nocrop, get_project_root, vehicle2enu, vehicle2image
from includes.plot_utils import plot_points_on_image, plot_gt_lines
from includes.input_manager import ImageInputManager, GroundTruthManager,  OdometryManager
from includes.output_manager import Display, VideoSaver, OutputSpecsCollection, CsvSaver
from includes.procedural.feature_extraction import FeatureExtraction
from includes.procedural.feature_point_selection import FeaturePointSelection

################################################################################
############################## Input CSV configs ###############################
################################################################################

plot_on_img = False
plot_final = True

export_path = get_project_root() + 'outputs/csv/'
csvFilePath = export_path + '/12_first_run_points.csv'
# Notice: bag number and paths must match what specified in configs.py

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
gt_mgr = GroundTruthManager(configs.gt_data_path, configs.bag, input_mgr.get_fps(), traj_only=True)
vehicle_cm = np.array([configs.camera_data['center_of_mass']['x'],
                       configs.camera_data['center_of_mass']['y'],
                       configs.camera_data['center_of_mass']['z']])

## Setup output
# Output video setup
# out_mgr = VideoSaver(
#     OutputSpecsCollection({
#         'collage_points': {"size": configs.out_image_size, "name": "collage_points"},
#     }),
#     fps=input_mgr.get_fps(),
#     split_after_frames=video_out_max_n_frames,
#     output_path=export_path,
#     input_id=input_mgr.get_input_id(),
#     enabled=save_video_out)

## Read csv

left_points_enu_cum = np.array([]).reshape((-1,2))
right_points_enu_cum = np.array([]).reshape((-1,2))
left_points_subsampled_enu_cum = np.array([]).reshape((-1,2))
right_points_subsampled_enu_cum = np.array([]).reshape((-1,2))
vehicle_trajectory = np.array([]).reshape((-1,2))
with open(csvFilePath, 'r') as csvf:
    csvReader = csv.DictReader(csvf)
    with tqdm(total=100) as pbar:
        for row in csvReader:
            try:
                timestamp = row['timestamp']
                frame_i = row['frame_i']
                enu_pose = eval(row['enu_pose']) if row['enu_pose'] else []
                enu_ref = eval(row['enu_ref']) if row['enu_ref'] else []
                left_x = eval(row['left_x']) if row['left_x'] else []
                left_y= eval(row['left_y']) if row['left_y'] else []
                right_x = eval(row['right_x']) if row['right_x'] else []
                right_y = eval(row['right_y']) if row['right_y'] else []
                left_x_subsampled = eval(row['left_x_subsampled']) if row['left_x_subsampled'] else []
                left_y_subsampled = eval(row['left_y_subsampled']) if row['left_y_subsampled'] else []
                right_x_subsampled = eval(row['right_x_subsampled']) if row['right_x_subsampled'] else []
                right_y_subsampled = eval(row['right_y_subsampled']) if row['right_y_subsampled'] else []

                vehicle_pose = gt_mgr.get_pose(np.uint(timestamp)) # np.array(enu_pose)
                enu_ref = gt_mgr.get_enu_reference() #np.array(enu_ref)

                left_points_camera_w = np.vstack((left_x, left_y)).T
                right_points_camera_w = np.vstack((right_x,right_y)).T
                left_points_enu = vehicle2enu(left_points_camera_w, vehicle_pose, vehicle_cm, enu_ref)
                right_points_enu = vehicle2enu(right_points_camera_w, vehicle_pose, vehicle_cm, enu_ref)
                left_points_enu_cum = np.vstack((left_points_enu_cum, left_points_enu))
                right_points_enu_cum = np.vstack((right_points_enu_cum, right_points_enu))
                vehicle_trajectory = np.vstack((vehicle_trajectory, vehicle_pose[0:2]))

                # Overlay points on frame image
                if plot_on_img:
                    left_x_front = vehicle2image(left_points_camera_w, bev_obj)
                    right_x_front = vehicle2image(right_points_camera_w, bev_obj)

                    closest_timestamp, frame = input_mgr.get_frame(timestamp)
                    plt.figure(4)
                    plt.clf()
                    plt.imshow(frame)
                    plt.plot(left_x_front[:, 0], left_x_front[:, 1], 'ro')
                    plt.plot(right_x_front[:, 0], right_x_front[:, 1], 'ro')
                    plt.show()
                    plt.pause(0.05)
            except:
                print("Error with frame {}".format(row['frame_i']))

            pbar.update(1)

# Plot all points
if plot_final:
    plt.figure(2)
    # plot_gt_lines(gt_mgr, 'k.', markersize=.5)
    plt.plot(-vehicle_trajectory[:, 1], vehicle_trajectory[:, 0], 'g.', markersize=.5)
    plt.plot(-left_points_enu_cum[:, 1], left_points_enu_cum[:, 0], 'r.', markersize=.5)
    plt.plot(-right_points_enu_cum[:, 1], right_points_enu_cum[:, 0], 'b.', markersize=.5)
    plt.gca().axis('equal')
    plt.show()
