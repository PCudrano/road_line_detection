#
# Author: Paolo Cudrano (archnnj)
#

import warnings
import logging
import glob
import numpy as np
import os
import cv2
from includes.manage_gt import load_odom, load_gt
from includes.manage_gt import _SmoothingFactors
from includes.utils import rot2d

class SmoothingFactors(_SmoothingFactors):
    def __init__(self, track_s=None, pose_gps_s=None, pose_enu_mae=None, centerline_enu_mae=None, odom_s=None):
        super().__init__(track_s, pose_gps_s, pose_enu_mae, centerline_enu_mae, odom_s)


class InputManager(object):
    _DEFAULT_FPS = 33

    def __init__(self, from_frame=0, to_frame=-1, step_frame=1):
        self.from_frame = from_frame
        self.to_frame = to_frame
        self.step_frame = step_frame
        self.original_fps = self._DEFAULT_FPS
        self.fps = self.original_fps

        self.frame_i = from_frame

    def next_frame(self):
        raise NotImplementedError("InputManager.next_frame(): Virtual method. InputManager is an abstract class.")

    def has_frames(self):
        return self.frame_i < self.to_frame

    def get_fps(self):
        return self.fps

    def get_original_fps(self):
        return self.original_fps

    def close(self):
        raise NotImplementedError("InputManager.close(): Virtual method. InputManager is an abstract class.")

    def get_input_id(self):
        raise NotImplementedError("InputManager.get_input_id(): Virtual method. InputManager is an abstract class.")

    def __del__(self):
        self.close()

    def __iter__(self):
        return InputManager._InputIterator(self)

    class _InputIterator():
        def __init__(self, input_manager):
            self.input_manager = input_manager

        def __next__(self):
            if self.input_manager.has_frames():
                return self.input_manager.next_frame()
            else:
                raise StopIteration

class VideoInputManager(InputManager):
    def __init__(self, video_path, *args, **kwargs):
        self.video_path = video_path
        super(VideoInputManager, self).__init__(*args, **kwargs)

        self.cap = cv2.VideoCapture(video_path)
        if (not self.cap.isOpened()):
            raise Exception("VideoInputManager: cannot open video file")
        if self.cap.get(cv2.CAP_PROP_FPS) > 0:
            self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.original_fps = self._DEFAULT_FPS
            warnings.warn("VideoInputManager: cannot read video fps, using default value: {} fps".format(self._DEFAULT_FPS))
        self.fps = round(self.original_fps / self.step_frame)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # set starting frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.from_frame)
        self.frame_i = self.from_frame
        if self.to_frame < 0:
            self.to_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            logging.info("VideoInputManager: no end frame specified, using end of video")
        self.timestamps = [] # list of timestamps for which the frame was read
        self.timestamps.append(self._get_cur_timestamp_from_video())  # append next timestamp (yet to be read)

    def next_frame(self):
        for i in range(self.step_frame):
            success, frame = self.cap.read()
        self.frame_i += self.step_frame
        self.timestamps.append(self._get_cur_timestamp_from_video())  # append next timestamp (yet to be read)
        return frame if success else None

    def has_frames(self):
        return self.cap.isOpened() and super(VideoInputManager, self).has_frames()

    def close(self):
        self.cap.release()

    def get_input_id(self):
        video_filename, _ = os.path.splitext(os.path.basename(self.video_path))
        return video_filename

    def get_next_timestamp(self):
        return self.timestamps[-1]

    def get_prev_timestamp(self):
        if self.frame_i - 1 < self.from_frame:
            raise IndexError("ImageInputManager.get_prev_timestamp(): no previous timestamp available")
        return self.timestamps[-2]

    def _get_cur_timestamp_from_video(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6


class ImageInputManager(InputManager):
    def __init__(self, images_folder_path, *args, **kwargs):
        self.images_folder_path = images_folder_path
        super(ImageInputManager, self).__init__(*args, **kwargs)

        self.images_folder_name = self.images_folder_path.split('/')[-2]
        frames_names_unordered = glob.glob1(self.images_folder_path, '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].png')
        if len(frames_names_unordered) == 0:
            raise ImportError("ImageInputManager: cannot find any frame in specified path: {}".format(self.images_folder_path))
        frames_names_timestamps_unordered = np.array([fn.replace('.png','') for fn in frames_names_unordered]).astype(np.int)
        i_sort_frames_ = np.argsort(frames_names_timestamps_unordered) # in nanoseconds
        self.frames_names = np.array(frames_names_unordered)[i_sort_frames_]
        self.frames_names_timestamps = frames_names_timestamps_unordered[i_sort_frames_].astype(np.uint)
        self.original_fps = round(1 / np.mean(np.diff(np.vectorize(self._timestamp_full2sec)(self.frames_names_timestamps)))) ## round!!!cap_fps = round(1 / np.mean(np.diff(frames_timestamps)))
        self.fps = round(self.original_fps / self.step_frame)
        self.from_timestamp = self.frames_names_timestamps[self.from_frame]
        self.frame_i = self.from_frame
        if self.to_frame < 0:
            self.to_frame = len(self.frames_names_timestamps)
            logging.info("VideoInputManager: no end frame specified, using end of video")

    def next_frame(self):
        frames_name = self.frames_names[self.frame_i]
        frame = cv2.imread("{}{}".format(self.images_folder_path, frames_name))
        self.frame_i += self.step_frame
        return frame # None if not found

    def has_frames(self):
        return self.frame_i < len(self.frames_names) and super(ImageInputManager, self).has_frames()

    def close(self):
        pass # nothing to close

    def get_input_id(self):
        recording_id = os.path.basename(os.path.normpath(self.images_folder_path))
        return recording_id

    def get_next_timestamp(self):
        return self.frames_names_timestamps[self.frame_i]

    def get_prev_timestamp(self):
        if self.frame_i - 1 < self.from_frame:
            raise IndexError("ImageInputManager.get_prev_timestamp(): no previous timestamp available")
        return self.frames_names_timestamps[self.frame_i-1]

    def get_frame(self, timestamp):
        closest_timestamp_i = np.argmin(np.abs(self.frames_names_timestamps - int(timestamp)))
        closest_timestamp = self.frames_names[closest_timestamp_i]
        frame_name = closest_timestamp
        frame = cv2.imread("{}{}".format(self.images_folder_path, frame_name))
        return closest_timestamp, frame  # None,None if not found

    @ staticmethod
    def _timestamp_full2millisec(timestamp_full):
        return float(timestamp_full) / 1e6  # [ms]

    @staticmethod
    def _timestamp_full2sec(timestamp_full):
        return float(timestamp_full) / 1e9  # [s]

    def _timestamp_abs2rel(self, timestamp_abs):
        return timestamp_abs - self.from_timestamp

    def _timestamp_rel2abs(self, timestamp_rel):
        return timestamp_rel + self.from_timestamp

    def _framei2timestamp(self, frame_i):
        return self.frames_names_timestamps[frame_i]

    def _timestamp2framei(self, timestamp):
        frame_i = np.argmin(np.abs(self.frames_names_timestamps - timestamp))
        return frame_i, self.frames_names_timestamps[frame_i] # return closest frame i and its actual timestamp


class InputManagerBuilder(object):
    def __init__(self):
        raise NotImplementedError("InputManagerBuilder cannot be instantiated.")

    @staticmethod
    def build_input_manager(configs):
        if configs.recording_n < 4:
            return VideoInputManager(configs.in_video_path,
                                     from_frame=configs.read_from_frame,
                                     to_frame=configs.read_to_frame,
                                     step_frame=configs.read_frame_step)
        else:
            return ImageInputManager(configs.in_video_path,
                                     from_frame=configs.read_from_frame,
                                     to_frame=configs.read_to_frame,
                                     step_frame=configs.read_frame_step)

class OdometryManager:
    def __init__(self, odom_data_path, bag_n, smoothing_factors):
        self.smoothing_factors = smoothing_factors if smoothing_factors is not None else SmoothingFactors()
        self.odom_timestamps, \
                self.odom_data_displacements, \
                self.odom_data_displacements_pose_interp_fcn = load_odom(odom_data_path, bag_n, smoothing_factors)

    def get_displacement(self, cur_timestamp, prev_timestamp):
        odometry_displacement_x, odometry_displacement_y, odometry_displacement_orient = self.odom_data_displacements_pose_interp_fcn(cur_timestamp, prev_timestamp)
        odometry_displacement_x = - odometry_displacement_x
        odometry_displacement_y = - odometry_displacement_y
        odometry_displacement = np.array([odometry_displacement_x, odometry_displacement_y, odometry_displacement_orient])
        return odometry_displacement

# if configs.odometry_from == "gps_rtk":
#     odometry_displacement_xy = rot2d(-gt_abs_orientation[-1]).dot((gt_traj_gps_coords_m[-2, 0:2] - gt_traj_gps_coords_m[-1, 0:2]).T).T
#     odometry_displacement_x, odometry_displacement_y = odometry_displacement_xy[0:2]
#     odometry_displacement_orient = np.unwrap([gt_abs_orientation[-1] - gt_abs_orientation[-2]])[0]
# elif configs.odometry_from == 'dead_reckoning':
#     timestamp_full_prev = timestamps_full_uptonow[-2]
#     odometry_displacement_x, odometry_displacement_y, odometry_displacement_orient = odom_data_displacements_pose_interp_fcn(timestamp_full, timestamp_full_prev)
#     # odometry_displacement_x = - odometry_displacement_x
#     # odometry_displacement_y = - odometry_displacement_y
# else:
#     raise NotImplementedError("Main: not implemented mode odometry_from={}".format(configs.odometry_from))
#
# else:
# odometry_displacement_x, odometry_displacement_y, odometry_displacement_orient = 0, 0, 0
#
#
# odometry_displacement = np.array([odometry_displacement_x, odometry_displacement_y, odometry_displacement_orient])

class GroundTruthManager():
    def __init__(self, gt_data_path, bag_n, input_fps, smoothing_factors=None, traj_only=False):
        self.smoothing_factors = smoothing_factors if smoothing_factors is not None else SmoothingFactors()
        self.traj_only = traj_only
        self.gt_lines_gps_coords_m, self.gt_coords_m_ref, \
                self.gt_lines_gps_coords_m_interp_fcns, self.gt_lines_abs_orientation_interp_fcns, \
                self.gt_traj_timestamps, self.gt_traj_gps_coords, \
                self.gt_traj_gps_coords_m, self.gt_traj_gps_coords_m_interp_fcn, \
                self.gt_traj_abs_orientation_interp_fcn, self.gt_traj_heading_interp_fcn, \
                self.gt_traj_lateral_offset_interp_fcn = load_gt(gt_data_path, bag_n, input_fps, self.smoothing_factors,
                                                                 traj_only=self.traj_only)

    def get_enu_reference(self):
        return self.gt_coords_m_ref

    def get_pose(self, timestamp):
        gt_gps_coords_m = self.gt_traj_gps_coords_m_interp_fcn(timestamp)
        gt_abs_orientation = self.gt_traj_abs_orientation_interp_fcn(timestamp)
        return np.hstack((gt_gps_coords_m, gt_abs_orientation.reshape(timestamp.shape)))

    def get_gt_heading_and_lateral_offset(self, timestamp):
        gt_heading = self.gt_traj_heading_interp_fcn(timestamp)
        gt_lateral_offset = self.gt_traj_lateral_offset_interp_fcn(timestamp)
        return gt_heading, gt_lateral_offset

    def get_displacement(self, cur_timestamp, prev_timestamp):
        cur_orientation = self.gt_traj_abs_orientation_interp_fcn(cur_timestamp)
        prev_orientation = self.gt_traj_abs_orientation_interp_fcn(prev_timestamp)
        prev_traj_gps_coords_m = self.gt_traj_gps_coords_m_interp_fcn(prev_timestamp)
        cur_traj_gps_coords_m = self.gt_traj_gps_coords_m_interp_fcn(cur_timestamp)
        gt_displacement_xy = rot2d(-cur_orientation).dot((prev_traj_gps_coords_m[0:2] - cur_traj_gps_coords_m[0:2]).T).T
        gt_displacement_x, gt_displacement_y = gt_displacement_xy[0:2]
        gt_displacement_orient = np.unwrap([cur_orientation - prev_orientation])[0]
        gt_displacement = np.array([gt_displacement_x, gt_displacement_y, gt_displacement_orient])
        return gt_displacement
