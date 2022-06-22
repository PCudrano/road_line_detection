#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
from includes.utils import printLinePointsOnImage
from matplotlib import pyplot as plt

def plot_points_on_image(image, points, color=(0, 153, 255), thickness=5):
    # return printLinePointsOnImage(image, points, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_8)
    points_round = np.round(points).astype(np.int) # round to integer (pixel coordinates)
    for pt in points_round: # for each point
        if (pt >= 0).all() and (pt < image.shape[0:2][::-1]).all(): # if inside the image
            cv2.circle(image, center=(pt[0], pt[1]), radius=5, color=(0, 0, 255), thickness=cv2.FILLED) # plot point

def plot_image_line_xy_on_image(image, model, x_lim, n_pts, color=(0, 153, 255), thickness=5):
    x_pts = np.linspace(x_lim[0], x_lim[1], n_pts)
    y_pts = model(x_pts)
    points = np.vstack((x_pts, y_pts))
    printLinePointsOnImage(image, points, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_8)

def plot_world_line_xy_on_front_image(front_image, world_model, bev_obj, x_lim, n_pts, color=(0, 153, 255), thickness=5):
    x_pts = np.linspace(x_lim[0], x_lim[1], n_pts)
    y_pts = world_model(x_pts)
    world_points = np.insert(np.vstack((x_pts, y_pts)).T, 2, (0), axis=1)
    image_points = bev_obj.projectWorldPointsToImagePoints(world_points)
    printLinePointsOnImage(front_image, image_points, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_8)

def plot_world_line_xy_on_bev_image(bev_image, world_model, bev_obj, x_lim, n_pts, color=(0, 153, 255), thickness=5):
    x_pts = np.linspace(x_lim[0], x_lim[1], n_pts)
    y_pts = world_model(x_pts)
    world_points = np.insert(np.vstack((x_pts, y_pts)).T, 2, (0), axis=1)
    bev_points = bev_obj.projectWorldPointsToBevPoints(world_points)
    printLinePointsOnImage(bev_image, bev_points, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_8)

def plot_gt_lines(gt_mgr, *args, **kwargs):
    ax = plt.gca()
    for i, gt_line_gps_coords_m in enumerate(gt_mgr.gt_lines_gps_coords_m):
        plt.plot(-gt_line_gps_coords_m[:, 1], gt_line_gps_coords_m[:, 0], *args, **kwargs)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    ### plot white lines on black background (otherwise, gray grid with black lines)
    ax.grid(False)
    ax.set_facecolor('black')
    ax.axis('off')
