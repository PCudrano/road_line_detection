#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
from math import pi, cos, sin, atan2, radians, degrees, ceil, floor
import warnings
import time

configs = type('obj', (object,), {'use_cpp' : False, 'debug': False, 'debug_only': [33]});


#from py_configs import configs
#import py_configs.configs as configs
from includes.utils import plot_rectangle, plot_polygon, plot_window, rotate_image_fit
from includes.Line import LineBuilderCollection, LineBuilder, Line, Window

from matplotlib import pyplot as plt # should be called centerally to set correct backend

class MovingWindowLineFollower:
    """
    Given any image, it executes the moving-window point selection algorithm for creating lines.
    """

    CENTROID_LARGEST = 0
    CENTROID_CLOSEST_WITH_MIN_AREA = 1

    WINDOW_ENLARGE_FIXED = 0
    WINDOW_ENLARGE_ADAPTIVE = 1

    def __init__(self, windows_init_points,
                 window_init_size=(200, 100),
                 window_init_enlarge_delta = np.array([30, 50]),
                 window_size=(50, 25),
                 window_enlarge_delta=np.array([30, 50]),
                 window_enlarge_type=WINDOW_ENLARGE_FIXED,
                 window_max_resize=3,
                 window_init_dir=pi/2, # upwards
                 window_dir_allowed_range=np.array([0+np.finfo(np.float64).eps,pi-np.finfo(np.float64).eps]), # (0,pi), extremes excluded
                 centroid_type=CENTROID_LARGEST, # CENTROID_CLOSEST_WITH_MIN_AREA,
                 centroid_min_area_percentage=0.01,
                 line_model_type=None, line_model_order=None,
                 line_sides=None,
                 use_ids = False, refine_with_prev_line=False):
        """
        Inizializes the class and the algorithm with the needed params
        :param window_init_point: array of window initial center points
        :param window_size: [width,height] size of the window to be used
        :param window_enlarge_delta: [x,y], increment applied to the window in x and y direction when it is resized as
        no centroid were found.
        :param window_init_dir: initial direction for first shift of the window
        :param window_dir_allowed_range: allowed range of search direction at any step, in rad
        :param centroid_type: method used to determine centroid inside the window.
                              CENTROID_LARGEST: centroid of connected component with largest area
                              CENTROID_CLOSEST_WITH_MIN_AREA: centroid closer to ego-vehicle, provided its connected component is larger than a threshold
        :param centroid_min_area_percentage: threshold area for CENTROID_CLOSEST_WITH_MIN_AREA, measured in percentage wrt window total area
        :param line_model_type: param to init each Line
        :param line_model_order: param to init each Line
        :param line_sides: optional, array of values (LineBuilder.LINE_SIDE_RIGHT|LineBuilder.LINE_SIDE_LEFT) for each
                element in window_init_points
        :param refine_with_prev_line: bool, whether to use previous lines to refine search (by following line if no
               features are found in window). Obs: this is anyway applied only when previous lines are passed with a
               new frame.
        """
        assert use_ids or not refine_with_prev_line, "Option refine_with_prev_line can be used only if option use_ids is active." # use_ids => refine_with_prev_line
        self.window_init_width = window_init_size[0] if window_init_size[0] is not None else 100
        self.window_init_height = window_init_size[1] if window_init_size[1] is not None else 200
        self.window_width = window_size[0] if window_size[0] is not None else 100
        self.window_height = window_size[1] if window_size[1] is not None else 200
        self.window_init_size = np.array([self.window_init_width, self.window_init_height])
        self.window_size = np.array([self.window_width, self.window_height])
        self.window_init_enlarge_delta = window_init_enlarge_delta
        self.window_enlarge_delta = window_enlarge_delta # [x,y]
        self.window_enlarge_type = window_enlarge_type if window_enlarge_type is not None else MovingWindowLineFollower.WINDOW_ENLARGE_FIXED
        self.window_max_resize = window_max_resize
        self.window_move_delta = 1.0 * self.window_height
        self.windows_init_points = windows_init_points
        self.window_init_dir = window_init_dir
        self.window_dir_allowed_range = window_dir_allowed_range
        self.centroid_type = centroid_type
        self.centroid_min_area_percentage = centroid_min_area_percentage
        self.centroid_min_area = self.centroid_min_area_percentage * self.window_height * self.window_width
        self.line_model_type = line_model_type
        self.line_model_order = line_model_order
        self.line_sides = line_sides
        self.use_ids = use_ids
        self.refine_with_prev_line = refine_with_prev_line
        self.previous_lines = None
        # FIXME
        self.image = None

    @staticmethod
    def followLine(image, *args, **kwargs):
        """
        Executes the algorithm on a single image. Useful for one-time operations (while the class can be used also
        for processing multiple frames)
        """
        obj = MovingWindowLineFollower(*args, **kwargs)
        obj.newFrame(image)
        return obj.findLines()

    def newFrame(self, image, image_line_th=0, new_windows_init_points=None, line_sides=None, previous_lines=None):
        """
        Resets the previous computation with the incoming new frame and its feature points.
        :param image: line mask (feature points) in front-view
        :param image_line_th: threshold line mask after projecting it to BEV, in order to avoid having large blob
        features with low value (the projection distorts their shape).
        :param new_windows_init_points: optional, new initial points for window (if not passed, it uses the default ones
               received when the object was created.
        :param previous_lines: optional, used only if refine_with_prev_line is set.
               Lines detected in previous steps to be used for the refinement; they must have aligned indices with
               init_points and, thus, with followed lines.
        """
        self.image = image
        # threshold bev line mask to avoid finding centroids with very low value at extremities
        _, self.image = cv2.threshold(self.image, image_line_th, np.max(self.image), cv2.THRESH_TOZERO)
        # self.current_lines = self._findLines()
        # return self.current_lines

        if new_windows_init_points is not None:
            self.windows_init_points = new_windows_init_points.copy()

        if line_sides is not None:
            self.line_sides = line_sides.copy()

        if self.refine_with_prev_line:
            self.previous_lines = previous_lines

    def findLines(self):
        """
        Generate a Line object for each completed line and returns it. If use_ids is True, the line ids correspond to
        the index position of the window_init_point from which the line is constructed.
        :return: one Line object foreach line found in world frame, front view and bev view respectively.
        """
        # create new line_builders
        self.line_builders = self._follow_line() # FIXME check if good method division/organization
        # fit lines
        if self.line_model_order is not None:
            self.lines = self.line_builders.getCompletedLines(model_type=self.line_model_type,
                                                              model_order=self.line_model_order)
        else:
            self.lines = self.line_builders.getCompletedLinePoints()
        return self.lines

    ## Private methods

    def _init_line_builders(self):
        line_builders = LineBuilderCollection()
        window_default_size = np.array([self.window_width, self.window_height])
        im_bounds = np.array([self.image.shape[0], self.image.shape[1]]) # [rows, cols]
        init_dir = self.window_init_dir
        for window_init_point_i in range(self.windows_init_points.shape[0]):
            window_init_point = self.windows_init_points[window_init_point_i, :]
            init_line_builder = LineBuilder(init_window_boundaries=None,
                                            init_point=window_init_point,
                                            init_window_size=self.window_init_size,
                                            init_dir=init_dir,
                                            dir_allowed_range=self.window_dir_allowed_range,
                                            window_size=window_default_size,
                                            im_bounds=im_bounds,
                                            max_resize=self.window_max_resize,
                                            line_id=window_init_point_i if self.use_ids else None,
                                            line_side=self.line_sides[window_init_point_i] if self.line_sides is not None else None
                                            )
            line_builders.addLineBuilder(init_line_builder)
        return line_builders

    def _follow_line(self):
        return self._py_follow_line()

    def _cpp_follow_line(self):
        line_builders = self._init_line_builders()
        wmp = []
        for wip_i in range(self.windows_init_points.shape[0]):
            wmp.append(cpp_wlf.WindowManagerParams(
                window_init_params=cpp_wlf.WindowInitParams(
                    center=self.windows_init_points[wip_i, :],
                    size=self.window_init_size,
                    dir=self.window_init_dir
                ),
                enlarge_delta=self.window_enlarge_delta,  # window (standard) enlarge delta
                move_delta=self.window_move_delta,  # move_delta: should set to be at least window size to avoid following issues at the end of the line (top of the image)
                enlarge_type=cpp_wlf.WINDOW_ENLARGE_ADAPTIVE if self.window_enlarge_type == MovingWindowLineFollower.WINDOW_ENLARGE_ADAPTIVE else cpp_wlf.WINDOW_ENLARGE_FIXED,
                window_standard_size=self.window_size,
                init_enlarge_delta=self.window_init_enlarge_delta,
                dir_allowed_range=self.window_dir_allowed_range,
                max_resize=self.window_max_resize,
                centroid_type=cpp_wlf.CENTROID_CLOSEST_WITH_MIN_AREA if self.centroid_type == MovingWindowLineFollower.CENTROID_CLOSEST_WITH_MIN_AREA else cpp_wlf.CENTROID_LARGEST,
                centroid_min_area_percentage=self.centroid_min_area_percentage,
                line_side= cpp_wlf.LINE_SIDE_LEFT if self.line_sides[wip_i] == LineBuilder.LINE_SIDE_LEFT else (cpp_wlf.LINE_SIDE_RIGHT if self.line_sides[wip_i] == LineBuilder.LINE_SIDE_RIGHT else cpp_wlf.LINE_SIDE_NA)
            ))
        points_v = np.array(map(lambda v: np.array(v), cpp_wlf.wlf_seq(self.image, wmp))) # cpp_wlf.wlf_parall(self.image, wmp)))
        for lines_i, points in enumerate(points_v):
            line_builders.getLineBuilders()[lines_i].points = points
            if points.size == 0:
                line_builders.getLineBuilders()[lines_i].lost = True
            else:
                # I'm simplifying the output state, technically lot of stuff could have happened inside (but should look at internal objects)
                line_builders.getLineBuilders()[lines_i].setNotJustInit()
                line_builders.getLineBuilders()[lines_i].setCompleted()
        return line_builders

    def _py_follow_line(self):
        line_builders = self._init_line_builders()

        if configs.debug and 33 in configs.debug_only:
            fig33 = plt.figure(33)
            plt.clf()

        save_i = 0
        # check if all completed
        while line_builders.getNotCompleted()[1].size > 0:
            # cycle on currently not completed
            for wip_lb_i,wip_lb in enumerate(line_builders.getNotCompleted()[1]):
                try:
                    if wip_lb.isLost():
                        # lost line (resized too many times)
                        if self.refine_with_prev_line and self.previous_lines is not None:
                            # if option enabled, attempt recovery
                            prev_line_filtered = [l for l in self.previous_lines if l.id == wip_lb.line_id] # self.previous_lines[wip_lb.line_id]
                            prev_line = prev_line_filtered[0] if prev_line_filtered is not None and len(prev_line_filtered)>0 else None
                            could_recover = self._recoverLossWithPrevLine(wip_lb, prev_line)
                            if could_recover:
                                # recovery successful, go back to main algorithm
                                wip_lb.resetLost()
                                wip_lb.resetWindowSize()
                            else:
                                # if really can't recover, then stop this line search
                                wip_lb.setCompleted()
                        else:
                            # if recovery option not enabled, lost line is considered completed
                            wip_lb.setCompleted()
                    else:
                        # line following in progress
                        # line following in progress
                        if configs.debug and 32 in configs.debug_only:
                            fig32 = plt.figure(32)
                            fig32.clf()
                            ax32 = fig32.add_subplot(121) # fig32.add_subplot(121)
                            ax32.imshow(self.image)
                            plt.figure(32);
                            plt.subplot(121);
                            plt.plot(wip_lb.points[:, 0], wip_lb.points[:, 1], 'r.', markersize=1)
                            # plot_rectangle(ax32, wip_lb.window[0, :], wip_lb.window[1, :], edgecolor="g")
                            plot_window(ax32, wip_lb.window, edgecolor="b")
                            if wip_lb.line_id is not None:
                                plt.suptitle("Moving window id={}".format(wip_lb.line_id))
                            else:
                                plt.suptitle("Moving window n={}".format(wip_lb_i))
                        if configs.debug and 33 in configs.debug_only:
                            fig33 = plt.figure(33)
                            ax33 = plt.gca() # fig33.add_subplot(111)
                            plt.imshow(self.image)
                            # plt.plot(wip_lb.points[:, 0], wip_lb.points[:, 1], 'r.', markersize=1)
                            # plot_rectangle(ax32, wip_lb.window[0, :], wip_lb.window[1, :], edgecolor="g")
                            plot_window(ax33, wip_lb.window, edgecolor="b")
                            if wip_lb.line_id is not None:
                                plt.suptitle("Moving window id={}".format(wip_lb.line_id))
                            else:
                                plt.suptitle("Moving window n={}".format(wip_lb_i))

                        # line_mask_window_patch = wip_lb.window.extractImagePatch(self.image) # self._getWindowPatch(self.image, wip_lb.window)
                        # centroid_in_patch = self._getCentroid(line_mask_window_patch, wip_lb)

                        centroid = self._getCentroid(wip_lb,
                                                     adaptive_mode=(self.window_enlarge_type == MovingWindowLineFollower.WINDOW_ENLARGE_ADAPTIVE))

                        if centroid is None: # no good centroid found / not enough pixels found in window
                            # enlarge window and repeat
                            enlarge_delta = self.window_init_enlarge_delta if wip_lb.isJustInit() else self.window_enlarge_delta
                            wip_lb.resizeWindow(enlarge_delta, fixSide=[None, "bottom"])

                            if configs.debug and 32 in configs.debug_only:
                                # plot_rectangle(ax32, wip_lb.window[0, :], wip_lb.window[1, :], edgecolor="r")
                                plot_window(ax32, wip_lb.window, edgecolor="y")
                                print("Resize window")
                            if configs.debug and 33 in configs.debug_only:
                                # plot_rectangle(ax32, wip_lb.window[0, :], wip_lb.window[1, :], edgecolor="r")
                                plot_window(ax33, wip_lb.window, edgecolor="y", linewidth=1)
                                print("Resize window")
                                # plt.savefig("./fig33_wlf_step_{:03d}.png".format(save_i)); save_i += 1;
                        else:
                            was_enlarging = wip_lb.resize_count > 0
                            wip_lb.addPoint(centroid)
                            if was_enlarging:
                                wip_lb.resetWindowSize()
                            forward_pt = wip_lb.moveWindowForward(self.window_move_delta)

                            if configs.debug and 32 in configs.debug_only:
                                ax32.plot(centroid[0], centroid[1], 'g.')  # selected
                                ax32.plot([centroid[0], forward_pt[0]],
                                        [centroid[1], forward_pt[1]], 'g-')  # selected
                            if configs.debug and 33 in configs.debug_only:
                                ax33.plot(wip_lb.points[:, 0], wip_lb.points[:, 1], 'r.', markersize=10)#2)
                                # plt.savefig("./fig33_wlf_step_{:03d}.png".format(save_i)); save_i+=1;
                            #     ax33.plot(centroid[0], centroid[1], 'g.')  # selected
                            #     ax33.plot([centroid[0], forward_pt[0]],
                            #               [centroid[1], forward_pt[1]], 'g-')  # selected
                except Exception as e:
                    warnings.warn("Exception caught in line-following algorithm:\n{}".format(str(e)))
                    wip_lb.setCompleted()

                if configs.debug and 32 in configs.debug_only:
                    plt.pause(0.05)
                    plt.show()

            if configs.debug and 33 in configs.debug_only:
                plt.pause(0.05)
                plt.show()

        return line_builders

    # def _windowInDir(self, starting_point, direction, distance, size=None):
    #     """
    #     Moves window from one point towards a direction of the specified distance. Optionally, the size is specified (otherwise, the
    #     default window size is used).
    #     :param starting_point: initial point
    #     :param direction: direction of the displacement
    #     :param distance: displacement from starting_point
    #     :param size: window size; if None, class default is used
    #     :return: new window computed
    #     """
    #     if size is None:
    #         size = (self.window_width, self.window_height)
    #     new_point = starting_point + distance * np.array([cos(direction), - sin(direction)])
    #     window = self._computeWindow(new_point, size)
    #     return window

    def _getCentroid(self, line_builder, adaptive_mode=False):
        if adaptive_mode:
            can_adapt = self._adaptWindowSizeToLine(line_builder)
            if not can_adapt: # error in finding centroid, probably empty window, no need to continue, it's a missing detection
                return None
        if configs.debug and 32 in configs.debug_only:
            fig32 = plt.figure(32)
            ax32 = fig32.add_subplot(121)
            plot_window(ax32, line_builder.window, edgecolor="g")
        if configs.debug and 33 in configs.debug_only:
            fig33 = plt.figure(33)
            ax33 = plt.gca() # fig33.add_subplot(111)
            if can_adapt:
                # plot_window(ax32, line_builder.window, edgecolor="b")
                plot_window(ax33, line_builder.window, edgecolor="g", linewidth=1)
            else:
                plot_window(ax33, line_builder.window, edgecolor="y", linewidth=1)

        line_mask_window_patch = line_builder.window.extractImagePatch(self.image)
        centroid_in_patch_tuple = self._getCentroidInPatch(line_mask_window_patch, line_builder) # no good centroid found / not enough pixels found in window
        if centroid_in_patch_tuple is None:
            centroid = None
        else:
            centroid_i, centroids_in_patch, stats, labels = centroid_in_patch_tuple
            centroid_in_patch = centroids_in_patch[centroid_i,:]
            centroid = np.round(centroid_in_patch + line_builder.window.getMaskBoundaries()[0, :]).astype(np.int)  # transform to image coord (add window displacement)
        return centroid

    # version 2: rotate once big image, then keep all windows straight (no other rotations) - proved ~2x faster than version 1
    def _adaptWindowSizeToLine(self, line_builder):
        # t = time.time()
        original_window_clone = line_builder.window.clone() # to restore original conditions if can't adapt size

        # TODO I need to think how to do it, but idea is that
        #  now: at every new cycle I take as centroid the one selected as _getCentroidInPatch, then rotate it and then match it with centroids in rotated mask
        #  instead: centroid is selected only once, than stored in convenient format (e.g. in image abs coordinates)
        #           then at every cycle it's transformed back into patch coordinates, and take the label pixel corresp to its position: matching centroid will be the one corresponding to that label!

        initial_line_mask_window_patch = line_builder.window.extractImagePatch(self.image)
        selected_centroid_in_patch_tuple = self._getCentroidInPatch(initial_line_mask_window_patch, line_builder)
        if selected_centroid_in_patch_tuple is None:
            warnings.warn("Adaptive window failed (no centroid in original window patch).")
            line_builder.window = original_window_clone
            return False  # FIXME ok??
        selected_centroid_in_patch_i, initial_centroids_in_patch, initial_stats, initial_labels = selected_centroid_in_patch_tuple
        selected_centroid_in_patch = initial_centroids_in_patch[selected_centroid_in_patch_i, :]
        selected_centroid_in_image = self._transformPointsFromWindowPatchToImage(selected_centroid_in_patch, line_builder.window, round_res=False)
        if configs.debug and 10 in configs.debug_only:
            # TODO add plots of rotated image
            plt.figure(10)
            plt.clf()
            plt.subplot(351)
            plt.imshow(initial_line_mask_window_patch)
            plt.plot(selected_centroid_in_patch[0], selected_centroid_in_patch[1], '.g')
            if original_window_clone.mask is not None:
                plt.subplot(352)
                plt.imshow(original_window_clone.mask)
            ax = plt.subplot(353)
            plt.imshow(image_straight)
            plot_window(ax, window_straight, edgecolor='b')

        # straight window and accordingly rotated image
        # if original_window_clone.dir >= pi/2:
        #     # when image rotated, x',y' has different origin --> need to adapt coordinates
        #     image_origin_in_rotated_image = np.array([self.image.shape[0] * sin(original_window_clone.dir - pi/2), 0])
        # else:
        #     # when image rotated, x',y' has different origin --> need to adapt coordinates
        #     image_origin_in_rotated_image = np.array([0, self.image.shape[1] * sin(pi/2 - original_window_clone.dir)])
        image_straight, image_straight_rotmat = rotate_image_fit(self.image, -degrees(original_window_clone.dir-pi/2))
        window_straight_center = image_straight_rotmat.dot(np.append(original_window_clone.center,1))
        window_straight = Window(window_straight_center, #original_window_clone.center + image_origin_in_rotated_image,
                                 original_window_clone.size,
                                 dir=pi/2, mask_padding=original_window_clone.mask_padding)

        # no need for "+image_origin_in_rotated_image", cause translation already done by matrix trasformation
        selected_centroid_in_image_straight = image_straight_rotmat.dot(np.append(selected_centroid_in_image,1)) # + image_origin_in_rotated_image

        done_left = False
        done_right = False
        max_enlarge = int(ceil(50 / (self.window_enlarge_delta[0] / 2.0))) # enlarge width max 50 px per side
        enlarge_count = 0
        while not (done_left and done_right) and enlarge_count < max_enlarge:
            # get masked patch from image (patch is the minimum rectangle containing the window); obs: window could be rotated, while patch is aligned to image
            line_mask_window_patch = window_straight.extractImagePatch(image_straight)
            # rotate back patch to align with window axis
            # line_mask_window_patch_rot, rotation_matrix = rotate_image_fit(line_mask_window_patch,
            #                                                                -degrees(window_straight.dir-pi/2),
            #                                                                outSize=window_straight.size # assure out size is as expected regardless of rounding errors
            #                                                                )

            # get unmasked patch (patch including also pixels outside of window mask)
            # roi_image = self.image[int((line_builder.window.center[1] - line_builder.window.mask.shape[0] / 2.0)):int((line_builder.window.center[1] + line_builder.window.mask.shape[0] / 2.0)),int(line_builder.window.center[0] - line_builder.window.mask.shape[1] / 2.0):int((line_builder.window.center[0] + line_builder.window.mask.shape[1] / 2.0))]
            # FIXME line_mask_window_patch_rot_roi should be the same as line_mask_window_patch_rot actually, but could be done to avoid dimension mess up by approx --> anyway should be fixed/done better
            # line_mask_window_patch_rot_roi = line_mask_window_patch_rot[
            #                                      int((line_mask_window_patch_rot.shape[0] - line_builder.window.size[1]) / 2.0):
            #                                      line_mask_window_patch_rot.shape[0]-int((line_mask_window_patch_rot.shape[0] - line_builder.window.size[1]) / 2.0),
            #                                      int((line_mask_window_patch_rot.shape[1] - line_builder.window.size[0]) / 2.0):
            #                                      line_mask_window_patch_rot.shape[1]-int((line_mask_window_patch_rot.shape[1] - line_builder.window.size[0]) / 2.0)]

            # line_mask_window_patch_rot_roi is basically the retrieved patch as if seen from window (w/o rotation), with
            # size of window, it's basically exactly what's inside the window and unrotated! --> can use this to check for width and such!
            # see save Figure_\10.png
            # TODO get cluster in this patch, then either do selection here or retrieve the one selected with other algorithm
            #  (through rotation and translation), then its stats will give me bounding box and can decide if extending,
            #  reducing or keeping as is according to how far it's from boundary (if-elif-else)


            # FIXME
            #   the following is done in window coords, but to do it I
            #   construct line_mask_window_patch_rot_roi for nothing, which means rotating the patch each time!
            #   ---> this is necessary tho, is it?
            #   Well, I do it because this way I find in stat the boundaries(top-left, bottom-right) of the
            #   connected component in window coordinates, and then can use it to enlarge window only in x direction.
            #   To do this without rotating patch, need to compute boundaries by myself: can isolate conn comp from 'labels' mask,
            #   and then compute everything along direction x,y of the window (i.e. with axis rotated of window.dir)...
            #   so this would require running another function in additon to connectedComponentsWithStats...
            #   --> need to understand if it s worth it computation-wise !!!

            # get connected components and stats for rotated window content
            patch_analysis = self._analyzePatch(line_mask_window_patch)
            if patch_analysis is None:
                warnings.warn("Adaptive window failed (no centroid in new window patch).")
                line_builder.window = original_window_clone
                return False # FIXME oks??
            centroids, labels, stats, num_labels = patch_analysis # unpack analysis result tuple

            # notice that the patch (line_mask_window_patch_rot_roi) corresponds to the window here
            # patch_size = line_mask_window_patch_rot_roi.shape[0:2][::-1]
            # patch_center = patch_size / 2.0
            # patch_dir = pi/2 # fixed, cause I rotated it s.t. dir points upwards!
            # selected_i = self._selectCentroid(centroids, labels, stats, patch_center, patch_dir, patch_size, line_builder.line_side)
            # # TODO check it selects same as otherwise!

            # convert coords centroid selected outside of loop (originally selected centroid) to coords of straight window
            selected_centroid_in_new_window = \
                    self._transformPointsFromImageToWindowPatch(selected_centroid_in_image_straight,
                                                                window_straight, round_res=False).astype(np.int)

            if np.any(selected_centroid_in_new_window > labels.shape[0:2][::-1]):
                # new window doesn't contain old centroid
                warnings.warn("Adaptive window failed (previous centroid out of bound).")
                line_builder.window = original_windowdist_centroids_from_dir_line_clone
                return False # FIXME oks??

            # get label of originally selected centroid in newly analyzed patch with modified window
            label_of_selected_centroid_in_new_window = labels[selected_centroid_in_new_window[1], selected_centroid_in_new_window[0]]

            # get bounding box (in stats) for connected component corresponding to label of original centroid (same connected component)
            selected_centroid_in_analysis_i = label_of_selected_centroid_in_new_window - 1 # label 0 is background, no analysis returned for it
            selected_centroid_in_analysis = centroids[selected_centroid_in_analysis_i, :]
            selected_stats_in_analysis = stats[selected_centroid_in_analysis_i, :]

            if configs.debug and 10 in configs.debug_only:
                plt.figure(10)
                # plt.subplot(3,5,6); plt.cla(); plt.imshow(roi_image)
                # plt.subplot(3,5,7); plt.cla(); plt.imshow(line_mask_window_patch)
                # plt.plot(selected_centroid_in_new_patch[0], selected_centroid_in_new_patch[1], '.g')
                # plt.subplot(3,5,8); plt.cla(); plt.imshow(window_straight.mask)
                # plt.subplot(3,5,9); plt.cla(); plt.imshow(line_mask_window_patch_rot)
                ax = plt.subplot(3,5,10)
                ax.cla()
                ax.imshow(line_mask_window_patch)
                ax.plot(selected_centroid_in_new_window[0], selected_centroid_in_new_window[1], '.r')
                ax.plot(selected_centroid_in_analysis[0], selected_centroid_in_analysis[1], '.g')
                plot_rectangle(ax,
                               [selected_stats_in_analysis[cv2.CC_STAT_LEFT]-1, selected_stats_in_analysis[cv2.CC_STAT_TOP]-1],
                               [selected_stats_in_analysis[cv2.CC_STAT_LEFT]-1 + selected_stats_in_analysis[cv2.CC_STAT_WIDTH] + 1,
                                selected_stats_in_analysis[cv2.CC_STAT_TOP]-1 + selected_stats_in_analysis[cv2.CC_STAT_HEIGHT] + 1],
                               edgecolor="b")

            # build bounding box for originally selected conn comp, in window coords
            left_bounding_box = selected_stats_in_analysis[cv2.CC_STAT_LEFT]
            right_bounding_box = left_bounding_box + selected_stats_in_analysis[cv2.CC_STAT_WIDTH]
            # compute how much to increase or reduce window lateral width on each side
            left_window_limit = 0
            right_window_limit = window_straight.size[0]
            delta_left = 0 # negstive for removing pixels to windows, positive for adding
            delta_right = 0  # negstive for removing pixels to windows, positive for adding
            if not done_left:
                if left_bounding_box > left_window_limit + 5: # 5: padding allowed
                    delta_left = - (left_bounding_box - (left_window_limit + 5))
                elif left_bounding_box < left_window_limit + 2:
                    delta_left = self.window_enlarge_delta[0] / 2.0
            if not done_right:
                if right_bounding_box < right_window_limit - 5:  # 5: padding allowed
                    delta_right = - ((right_window_limit - 5) - right_bounding_box)
                elif right_bounding_box > right_window_limit - 2:
                    delta_right = self.window_enlarge_delta[0] / 2.0

            # reduce/enlarge window laterally on each side
            window_straight.enlargeWindow(left=delta_left, right=delta_right, top=0, bottom=0)
            if delta_left > 0 or delta_right > 0:
                enlarge_count += 1
            # done on a side if window was not enlarged on that side (either reduced of the right amount or no operation done)
            done_left = delta_left <= 0 # no enlargment performed
            done_right = delta_right <= 0 # no enlargment performed

            if configs.debug and 10 in configs.debug_only:
                plt.figure(10)
                # roi_image = self.image[int((window_straight.center[1] - line_builder.window.mask.shape[0] / 2.0)):int((line_builder.window.center[1] + line_builder.window.mask.shape[0] / 2.0)),int(line_builder.window.center[0] - line_builder.window.mask.shape[1] / 2.0):int((line_builder.window.center[0] + line_builder.window.mask.shape[1] / 2.0))]
                line_mask_window_patch = window_straight.extractImagePatch(image_straight)
                # plt.subplot(3,5,11); plt.cla(); plt.imshow(roi_image)
                plt.subplot(3,5,12); plt.cla(); plt.imshow(line_mask_window_patch)
                # plt.subplot(3,5,13); plt.cla(); plt.imshow(window_straight.mask)

        # finally, once size is adapted, resize actual window
        adapted_window_center = image_straight_rotmat[0:2,0:2].T.dot(window_straight.center - image_straight_rotmat[:,2]) # rotate back and move origin from x',y' to original x,y
        adapted_window_size = window_straight.size
        line_builder.window = Window(adapted_window_center,
                                     adapted_window_size,
                                     original_window_clone.dir,
                                     mask_padding=original_window_clone.mask_padding)

        if line_builder.isJustInit():
            # set window height (not modified) to non-init state
            delta_height = line_builder.window_size[1]- line_builder.window.size[1]
            line_builder.resizeWindow([0, delta_height], fixSide=[None, "top"], countResize=False)

        # tot_t = time.time() - t
        # print(tot_t)
        return True

    def _analyzePatch(self, window_patch):
        connectivity = 8 # 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(window_patch, connectivity, cv2.CV_32S)
        # Obs: label=0 is background --> filter out
        if num_labels == 1: # only background
            return None # no connected components found
        # delete row for background
        num_labels -= 1
        # labels = np.delete(labels, (0), axis=0)
        stats = np.delete(stats, (0), axis=0)
        centroids = np.delete(centroids, (0), axis=0)
        return (centroids, labels, stats, num_labels)

    def _getCentroidInPatch(self, window_patch, line_builder):
        """
        Get all centroids, then select most important
        :param window_patch:
        :return: centroid (int coordinates)
        """

        # assert: CENTROID_CLOSEST_WITH_MIN_AREA => line_builder.line_side well defined
        assert not (self.centroid_type == MovingWindowLineFollower.CENTROID_CLOSEST_WITH_MIN_AREA) or \
                line_builder.line_side in [LineBuilder.LINE_SIDE_LEFT, LineBuilder.LINE_SIDE_RIGHT]

        if configs.debug and 32 in configs.debug_only:
            fig = plt.figure(32)
            ax = fig.add_subplot(122)
            ax.cla()
            ax.imshow(window_patch)

        analysis = self._analyzePatch(window_patch)
        if analysis is None:
            return None
        centroids, labels, stats, num_labels = analysis # unpack analysis result tuple

        if self.centroid_type == MovingWindowLineFollower.CENTROID_LARGEST:
            # if found some centroids
            # get one with max area
            max_area_i = np.argmax(stats[:, cv2.CC_STAT_AREA])
            ret_centroid_i = max_area_i
        elif self.centroid_type == MovingWindowLineFollower.CENTROID_CLOSEST_WITH_MIN_AREA:
            filtered_centroids_i = np.argwhere(stats[:, cv2.CC_STAT_AREA] >= self.centroid_min_area)
            if len(filtered_centroids_i) == 0:
                return None # no centroid large enough
            filtered_centroids_i = filtered_centroids_i.flatten() # where always adds 1 dimension
            filtered_centroids = centroids[filtered_centroids_i,:]
            # Notice: these are coords in the patch reference, I need to add window size before passing to bev
            # filtered_centroids_im = np.round(filtered_centroids + line_builder.window.getMaskBoundaries()[0, :]).astype(np.int) # add window top-left coordinates
            filtered_centroids_im = self._transformPointsFromWindowPatchToImage(filtered_centroids, line_builder.window)
            # filtered_centroids_im_ycoord = filtered_centroids_im[:,1]
            # # FIXME: huge bug here! all these centroids are in image coordinates, but I'm filtering them as if they were
            # #  in world coordinates!!!!!!!!!!!!!! + I don't have BEV object here cause I only work on image...
            # #  One fix could be for this closest centroid option to pass with the frame also BEVLineProxy representing
            # #  the x-axis and to check distance with that or so...
            # #  Otherwise need to change rule, e.g. using line.side attribute, which now is not used... (could be set
            # #  automatically outside tho, no need for another param in main). Moreover, first option not good cause in
            # #  bends I could cross x axis of car, thus would select centroids more external in that region. What I
            # #  should do instead is select centroids closer to centerline --> either use centerline (but many times not
            # #  available), or better set line.side from outside and here check which centroid is closer to centerline
            # #  using forwardDir as a reference (e.g. if right line, compute distance from centroids to line along
            # #  forwardDir, the one more to the left wins).
            # # get centroid closest to x-axis --> min |y_coord|
            # closest_filtered_centroid_i = np.argmin(np.abs(filtered_centroids_im_ycoord))
            # closest_filtered_centroid = filtered_centroids[closest_filtered_centroid_i,:]
            # ret_centroid = closest_filtered_centroid

            # distance centroids to direction line
            #   Get 2 points in direction line: pt1 is window center, pt2 moves from it following direction (and thus
            #   goes always "upwards" if direction is bounded upwards)
            window_center = line_builder.getWindowCenterPoint()
            dir_line_pt1 = window_center
            dir_line_pt2 = window_center + -100 * np.array([cos(line_builder.forwardDir), sin(line_builder.forwardDir)])

            # signed distance from points filtered_centroids_im to line through dir_line_pt2,dir_line_pt1
            # note on the sign: if I sit on window center and look towards direction, distance is positive on my
            # right and negative on my left.
            dist_centroids_from_dir_line = np.cross(dir_line_pt2-dir_line_pt1,filtered_centroids_im-dir_line_pt1)/np.linalg.norm(dir_line_pt2-dir_line_pt1)

            if line_builder.line_side == LineBuilder.LINE_SIDE_LEFT:
                closest_centroid_i = np.argmax(dist_centroids_from_dir_line)
            elif line_builder.line_side == LineBuilder.LINE_SIDE_RIGHT:
                closest_centroid_i = np.argmin(dist_centroids_from_dir_line)
            ret_centroid_i = filtered_centroids_i[closest_centroid_i]

            if configs.debug and 32 in configs.debug_only:
                for component_i in range(0,num_labels):
                    if component_i in filtered_centroids_i:
                        ax.plot(centroids[component_i,0], centroids[component_i,1], 'g.') # candidates
                    else:
                        ax.plot(centroids[component_i, 0], centroids[component_i, 1], 'r.')  # candidates
                    plot_rectangle(ax,
                                   [stats[component_i,cv2.CC_STAT_LEFT]-1, stats[component_i,cv2.CC_STAT_TOP]-1],
                                   [stats[component_i, cv2.CC_STAT_LEFT]-1+stats[component_i, cv2.CC_STAT_WIDTH]+1, stats[component_i, cv2.CC_STAT_TOP]-1+stats[component_i, cv2.CC_STAT_HEIGHT]+1],
                                   edgecolor="b")

        ret_centroid = centroids[ret_centroid_i, :]

        ret_centroid = np.round(ret_centroid).astype(np.int)
        return (ret_centroid_i, centroids, stats, labels)

    def _transformPointsFromWindowPatchToImage(self, points, window, round_res=True):
        pts_img = points + window.getMaskBoundaries()[0, :] # add window top-left coordinates
        if round_res:
            return np.round(pts_img).astype(np.int)
        return pts_img

    def _transformPointsFromImageToWindowPatch(self, points, window, round_res=True):
        pts_patch = points - window.getMaskBoundaries()[0, :]
        if round_res:
            return np.round(pts_patch).astype(np.int)  # add window top-left coordinates
        return pts_patch

    def _selectCentroid(self, centroids, labels, stats, window_center, window_dir, window_mask_boundaries, line_side):
        if self.centroid_type == MovingWindowLineFollower.CENTROID_LARGEST:
            # if found some centroids
            # get one with max area
            max_area_i = np.argmax(stats[:, cv2.CC_STAT_AREA])
            ret_centroid_i = max_area_i
        elif self.centroid_type == MovingWindowLineFollower.CENTROID_CLOSEST_WITH_MIN_AREA:
            filtered_centroids_i = np.argwhere(stats[:, cv2.CC_STAT_AREA] >= self.centroid_min_area)
            if len(filtered_centroids_i) == 0:
                return None # no centroid large enough
            filtered_centroids_i = filtered_centroids_i.flatten() # where always adds 1 dimension
            filtered_centroids = centroids[filtered_centroids_i,:]
            # Notice: these are coords in the patch reference, I need to add window size before passing to bev
            filtered_centroids_im = np.round(filtered_centroids + window_mask_boundaries[0, :]).astype(np.int) # add window top-left coordinates

            # distance centroids to direction line
            #   Get 2 points in direction line: pt1 is window center, pt2 moves from it following direction (and thus
            #   goes always "upwards" if direction is bounded upwards)
            window_center = window_center
            dir_line_pt1 = window_center
            dir_line_pt2 = window_center + -100 * np.array([cos(window_dir), sin(window_dir)])

            # signed distance from points filtered_centroids_im to line through dir_line_pt2,dir_line_pt1
            # note on the sign: if I sit on window center and look towards direction, distance is positive on my
            # right and negative on my left.
            dist_centroids_from_dir_line = np.cross(dir_line_pt2-dir_line_pt1,filtered_centroids_im-dir_line_pt1)/np.linalg.norm(dir_line_pt2-dir_line_pt1)

            if line_side == LineBuilder.LINE_SIDE_LEFT:
                closest_centroid_i = np.argmax(dist_centroids_from_dir_line)
            elif line_side == LineBuilder.LINE_SIDE_RIGHT:
                closest_centroid_i = np.argmin(dist_centroids_from_dir_line)
            ret_centroid_i = filtered_centroids_i[closest_centroid_i]
        return ret_centroid_i


    def _getWindowPatch(self, image, window):
        window_int = np.round(window).astype(np.int)
        topleft_x = window_int[0,0]
        topleft_y = window_int[0,1]
        bottomright_x = window_int[1,0]
        bottomright_y = window_int[1,1]
        return image[topleft_y:bottomright_y, topleft_x:bottomright_x]

    def _recoverLossWithPrevLine(self, wip_line_builder, previous_line):
        if previous_line is None:
            return False

        # compute init point for line builder to be on previous line, at last detected y
        last_detected_pt = wip_line_builder.getWindowCenterPoint()
        if np.isnan(last_detected_pt).any():
            # computeXGivenY can return NaN with arc_length_parametrization model or not return any point if they would be extrapolated
            return False
        last_detected_pt_y = last_detected_pt[1]
        line_corresponding_x = previous_line.computeXGivenY(last_detected_pt_y)
        line_following_init_point = np.array([line_corresponding_x, last_detected_pt_y])
        # compute init dir for line builder to be as the tangent to previous line in last detected y
        line_following_init_dir_delta_y = 1
        line_following_init_dir_delta_x_pts = previous_line.computeXGivenY([last_detected_pt_y, last_detected_pt_y - 1])
        if line_following_init_dir_delta_x_pts.size == 0 or np.isnan(line_following_init_dir_delta_x_pts).any():
            # computeXGivenY can return NaN with arc_length_parametrization model or not return any point if they would be extrapolated
            return False
        line_following_init_dir_delta_x = np.diff(line_following_init_dir_delta_x_pts)
        line_following_init_dir = atan2(line_following_init_dir_delta_y, line_following_init_dir_delta_x )
        line_following_window_size = np.max(np.vstack((wip_line_builder.window_size, wip_line_builder.window.size)), axis=0) # max between standard and current window size

        lb = LineBuilder(init_window_boundaries=None,
                         init_point=line_following_init_point,
                         init_dir= line_following_init_dir,
                         dir_allowed_range=self.window_dir_allowed_range,
                         window_size=line_following_window_size,
                         im_bounds=wip_line_builder.im_bounds)
        lb.addPoint(line_following_init_point)
        # I follow line with lb and move along also wip_line_builder, until wip_line_builder doesn't get a valid centroid

        if configs.debug and 32 in configs.debug_only:
            fig32 = plt.figure(32)
            ax32 = fig32.add_subplot(121)
            ax32.imshow(self.image)
            # plot_rectangle(ax32, wip_line_builder.window[0, :], wip_line_builder.window[1, :], edgecolor="g")
            plot_window(ax32, wip_line_builder.window, edgecolor="b")
            n_pts = 100
            line_y_pts = np.linspace(0, self.image.shape[0], n_pts)
            line_x_pts = previous_line.computeXGivenY(line_y_pts)
            if not (line_x_pts.size == 0 or np.isnan(line_x_pts).any()):
                # computeXGivenY can return NaN with arc_length_parametrization model or not return any point if they would be extrapolated
                ax32.plot(line_x_pts, line_y_pts, ':r')
            ax32.plot(line_following_init_point[0], line_following_init_point[1], 'or')
        if configs.debug and 33 in configs.debug_only:
            fig33 = plt.figure(33)
            ax33 = plt.gca() # fig33.add_subplot(111)
            ax33.imshow(self.image)
            # plot_rectangle(ax33, wip_line_builder.window[0, :], wip_line_builder.window[1, :], edgecolor="g")
            plot_window(ax33, wip_line_builder.window, edgecolor="g", linewidth=1)
            n_pts = 100
            line_y_pts = np.linspace(0, self.image.shape[0], n_pts)
            line_x_pts = previous_line.computeXGivenY(line_y_pts)
            if not (line_x_pts.size == 0 or np.isnan(line_x_pts).any()):
                # computeXGivenY can return NaN with arc_length_parametrization model or not return any point if they would be extrapolated
                ax33.plot(line_x_pts, line_y_pts, ':r')
            ax33.plot(line_following_init_point[0], line_following_init_point[1], 'or')

        found_centroid = False
        while not found_centroid and not lb.isCompleted(): # wip_line_builder.isCompleted():
            # center wip_line_builder on current point on line
            wip_line_builder.centerWindowOnPoint(lb.getWindowCenterPoint())
            # get centroid from there

            # line_mask_window_patch = wip_line_builder.window.extractImagePatch(self.image) # self._getWindowPatch(self.image, wip_line_builder.window)
            # centroid_in_patch = self._getCentroidInPatch(line_mask_window_patch, wip_line_builder)
            centroid = self._getCentroid(wip_line_builder, adaptive_mode=False)

            if centroid is None:
                # no good centroid found --> keep following line
                forward_pt = lb.moveWindowForward(self.window_move_delta)
                forward_pt_y = forward_pt[1]
                line_corresponding_x = previous_line.computeXGivenY(forward_pt_y)
                if np.isnan(line_corresponding_x):
                    # computeXGivenY can return NaN with arc_length_parametrization model or not return any point if they would be extrapolated
                    return False
                line_point = np.array([line_corresponding_x, forward_pt_y])
                lb.addPoint(line_point)

                if configs.debug and 32 in configs.debug_only:
                    # plot_rectangle(ax32, wip_lb.window[0, :], wip_lb.window[1, :], edgecolor="r")
                    plot_window(ax32, wip_lb.window, edgecolor="y")
                    print("Resize window")
                if configs.debug and 33 in configs.debug_only:
                    # plot_rectangle(ax32, wip_lb.window[0, :], wip_lb.window[1, :], edgecolor="r")
                    plot_window(ax33, wip_lb.window, edgecolor="y", linewidth=1)
                    print("Resize window")
                    # plt.savefig("./fig33_wlf_step_{:03d}.png".format(save_i)); save_i += 1;
            else:
                # centroid found: stop recovery and return
                found_centroid = True
                # center wip_line_builder on current point on line
                wip_line_builder.centerWindowOnPoint(lb.getWindowCenterPoint(), window_dir=lb.forwardDir)

                if configs.debug and 32 in configs.debug_only:
                    lb_center_pt = lb.getWindowCenterPoint()
                    # plot_rectangle(ax32, lb.window[0, :], lb.window[1, :], edgecolor="y")
                    ax32.plot(lb_center_pt[0], lb_center_pt[1], 'yo')  # selected
                    ax32.plot(centroid[0], centroid[1], 'go')  # selected

            if configs.debug and 32 in configs.debug_only:
                plt.pause(0.05)
                plt.show()

        return found_centroid
