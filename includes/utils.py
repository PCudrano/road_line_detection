#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
from math import cos, sin, atan2, atan, pi, floor, ceil, radians, degrees
import scipy.stats as st
import matplotlib.patches as patches
# from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from scipy.interpolate import UnivariateSpline, splev, splrep
from scipy.optimize import minimize
import scipy
import warnings
from pathlib import Path

def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent) + '/'

def rotX(a_rad):
    return np.float32([
        [1, 0, 0],
        [0, cos(a_rad), -sin(a_rad)],
        [0, sin(a_rad), cos(a_rad)]])
def rotY(a_rad):
    return np.float32([
        [cos(a_rad), 0, sin(a_rad)],
        [0, 1, 0],
        [-sin(a_rad), 0, cos(a_rad)]])
def rotZ(a_rad):
    return np.float32([
        [cos(a_rad), -sin(a_rad), 0],
        [sin(a_rad), cos(a_rad), 0],
        [0, 0, 1]])
def rot2d(a_rad):
    return np.float32([
        [cos(a_rad), -sin(a_rad)],
        [sin(a_rad), cos(a_rad)]])

def overlay_image(image, mask, color_tuple, alpha_image=None): # alpha_image: scaling to apply to image. If None, alpha_image = 1-alpha_mask
    # color_tuple = (r,g,b,alpha)
    if len(image.shape) < 3 or image.shape[2] < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colorImage = np.zeros((image.shape[0],image.shape[1],3), np.uint8)
    colorImage[:, :] = tuple(color_tuple[0:3])
    colorMask = cv2.bitwise_and(colorImage, colorImage, mask=mask)
    alpha_mask = color_tuple[3]
    if alpha_image is None:
        alpha_image = 1 - alpha_mask
    overlayed_image = cv2.addWeighted(image.astype(np.float), alpha_image, colorMask.astype(np.float), alpha_mask, 0) # cv2.addWeighted(image, 1-alpha, colorMask, alpha, 0)
    overlayed_image = np.clip(overlayed_image, None, np.iinfo(image.dtype).max) # clip values overflowing
    return overlayed_image.astype(image.dtype)

def overlay_image_keep_source(image, mask, color_tuple): # alpha_image: scaling to apply to image. If None, alpha_image = 1-alpha_mask
    # color_tuple = (r,g,b,alpha)
    alpha_mask = color_tuple[3]
    alpha_image = 1 - alpha_mask
    if len(image.shape) < 3 or image.shape[2] < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    colorImage = np.zeros((image.shape[0],image.shape[1],3), np.uint8)
    colorImage[:, :] = tuple(np.array(color_tuple[0:3]))
    colorMask = cv2.bitwise_and(colorImage, colorImage, mask=mask)
    image_without_mask_region = cv2.bitwise_and(image, np.zeros_like(image), dst=image.copy(), mask=mask)
    image_masked_region = cv2.bitwise_and(image, image, mask=mask)
    #ZimageWithAlphaedRegion =
    #selective_mask =
    #overlayed_image = cv2.addWeighted(image_without_mask_region.astype(np.float), alpha_image, colorMask.astype(np.float), alpha_mask, 0) # cv2.addWeighted(image, 1-alpha, colorMask, alpha, 0)
    overlayed_image = image_without_mask_region + cv2.addWeighted(image_masked_region.astype(np.float), alpha_image, colorMask.astype(np.float), alpha_mask, 0)
    overlayed_image = np.clip(overlayed_image, None, np.iinfo(image.dtype).max) # clip values overflowing
    return overlayed_image.astype(image.dtype)

def overlay_smaller_image(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def computeGradientMagnitude(gradx, grady):
    return np.hypot(gradx, grady)

def computeGradientDirection(gradx, grady, returnVisualiz=False, angle0_2pi=True): # angle0_2pi: True --> [0,2pi), False --> (-pi, pi]
    grad_magnitude = computeGradientMagnitude(gradx, grady)
    grad_dir = np.zeros_like(grad_magnitude)
    if returnVisualiz:
        grad_dir_visualiz_hsv = np.zeros((grad_magnitude.shape[0], grad_magnitude.shape[1], 3))
    for i in range(grad_magnitude.shape[0]):
        for j in range(grad_magnitude.shape[1]):
            # Retrieve a single value
            valueX = gradx[i, j]
            valueY = grady[i, j]
            # Calculate the corresponding single direction, done by applying the arctangens function
            dir = atan2(valueY, valueX)
            if angle0_2pi:
                dir = dir + pi if dir+pi != 2*pi else 0 # add pi to all, and if becomes 2pi substitute it with 0
            # Store in orientation matrix element
            grad_dir[i, j] = dir
            if returnVisualiz:
                if angle0_2pi:
                    grad_dir_visualiz_hsv[i, j, :] = (dir * 180 / (2 * pi), 100, grad_magnitude[i, j] * 255 / grad_magnitude.max())
                else:
                    grad_dir_visualiz_hsv[i, j, :] = (dir * 180 / (2 * pi) + 90, 100, grad_magnitude[i, j] * 255 / grad_magnitude.max())
    if returnVisualiz:
        grad_dir_visualiz_hsv = grad_dir_visualiz_hsv.astype((np.uint8))
        grad_dir_visualiz = cv2.cvtColor(grad_dir_visualiz_hsv, cv2.COLOR_HSV2RGB)
        return grad_dir, grad_dir_visualiz
    else:
        return grad_dir

def computeGradientDirectionVisualization(grad_dir, grad_magnitude=None, angle0_2pi=True):
    grad_dir_visualiz_hsv = np.zeros((grad_dir.shape[0], grad_dir.shape[1], 3))
    for i in range(grad_dir.shape[0]):
        for j in range(grad_dir.shape[1]):
            dir = grad_dir[i,j]
            if angle0_2pi:
                dir = dir + pi if dir+pi != 2*pi else 0 # add pi to all, and if becomes 2pi substitute it with 0
            grad_mag = grad_magnitude[i, j] if grad_magnitude is not None else 1
            grad_mag_max = grad_magnitude.max() if grad_magnitude is not None else 1
            if angle0_2pi:
                grad_dir_visualiz_hsv[i, j, :] = (dir * 180 / (2 * pi), 100, grad_mag * 255 / grad_mag_max)
            else:
                grad_dir_visualiz_hsv[i, j, :] = (dir * 180 / (2 * pi) + 90, 100, grad_mag * 255 / grad_mag_max)
    grad_dir_visualiz_hsv = grad_dir_visualiz_hsv.astype((np.uint8))
    grad_dir_visualiz = cv2.cvtColor(grad_dir_visualiz_hsv, cv2.COLOR_HSV2RGB)
    return grad_dir_visualiz

def create_gaussian_kern(kernlen=21, nsig=-1):
    """Returns a 2D Gaussian kernel array."""
    if nsig < 0:
        nsig = 0.3 * ((kernlen - 1) * 0.5 - 1) + 0.8
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def binGradientDirection(grad_dir, n_grad_bins, centered_on_0=True):
    grad_dir_binned = grad_dir.copy()
    delta_interval = 2 * pi / n_grad_bins / 2
    if centered_on_0:
        grad_intervals = np.array([(start, start + 2 * pi / n_grad_bins) for start in (np.linspace(0, 2 * pi, n_grad_bins+1) - 2 * pi / n_grad_bins / 2)[:-1]])
        center_intervals = np.array([c for c in (np.linspace(0, 2 * pi, n_grad_bins + 1))[:-1]])
    else:
        grad_intervals = np.array([(start, start + 2 * pi / n_grad_bins) for start in np.linspace(0, 2 * pi, n_grad_bins+1)[:-1]])
        center_intervals = (grad_intervals[:,0] + grad_intervals[:,1]) / 2
    grad_intervals[grad_intervals < 0] = grad_intervals[grad_intervals < 0] + 2 * pi  # convert negative angles to [0,2pi)
    center_intervals[center_intervals < 0] = center_intervals[center_intervals < 0] + 2 * pi  # convert negative angles to [0,2pi)
    for center_interval in center_intervals:
        start_intv = center_interval - delta_interval
        end_intv = center_interval + delta_interval
        start_intv = start_intv + 2 * pi if start_intv < 0 else start_intv
        end_intv = end_intv + 2 * pi if end_intv < 0 else end_intv
        if start_intv <= end_intv: # normal interval
            # center_intv = (start_intv + end_intv) / 2
            grad_dir_binned[(grad_dir >= start_intv) & (grad_dir < end_intv)] = center_interval
        else: # interval includes 0 (and below 0 angles counted from 2pi down)
            # center_intv = (start_intv-2*pi + end_intv) / 2
            grad_dir_binned[(grad_dir >= 0) & (grad_dir < end_intv)] = center_interval # part after 0
            grad_dir_binned[(grad_dir >= start_intv) & (grad_dir < 2*pi)] = center_interval # part before 0
    return grad_dir_binned

def isolateLabelMask(label_img, labels_to_keep):
    labels = np.unique(label_img)
    filtered_label_img = label_img.copy()
    mask = np.ones_like(filtered_label_img)
    for lbl in labels:
        if lbl not in labels_to_keep:
            filtered_label_img[label_img == lbl] = 0
            mask[label_img == lbl] = 0 # FIXME doens't work!! why??
    return mask, filtered_label_img
# def isolateLabelMask(label_img, labels_to_keep):
#     labels = np.unique(label_img)
#     filtered_label_img = label_img.copy()
#     for lbl in labels:
#         if lbl not in labels_to_keep:
#             filtered_label_img[filtered_label_img == lbl] = 0
#     mask = np.uint8((filtered_label_img > 0))
#     return mask, filtered_label_img

def compute_roi(image, limits):
    # limits = [[x_min, y_min], [x_max, y_max]]
    if len(image.shape) > 2: # color image
        roi = image[limits[0, 1]:limits[1, 1], limits[0, 0]:limits[1, 0], :]
    else:
        roi = image[limits[0, 1]:limits[1, 1], limits[0, 0]:limits[1, 0]]
    return roi

def compute_roi_world(image, bev_obj, world_limits):
    # world_limits = [[x_min, y_min], [x_max, y_max]]_w ; assumes z=0
    bev_limits = np.round(bev_obj.projectWorldPointsToBevPoints(
                                np.insert(np.float32(world_limits[::-1, :,]), 2, 0, axis=1))
    ).astype(np.int)  # [[x_max, y_max],[x_min, y_min]]_im = [[col_max, row_max],[col_min, row_min]]
    return compute_roi(image, bev_limits), bev_limits

def plot_rectangle(ax, pt_topleft, pt_bottomright, edgecolor='r', linewidth=1, facecolor='none'):
    width = pt_bottomright[0] - pt_topleft[0]
    height = pt_bottomright[1] - pt_topleft[1]
    ax.add_patch(patches.Rectangle((pt_topleft[0], pt_topleft[1]),
                                   width, height,
                                   edgecolor=edgecolor,
                                   linewidth=linewidth,
                                   facecolor=facecolor
                                   ))
def plot_polygon(ax, points, edgecolor='r', linewidth=1, facecolor='none'):
    ax.add_patch(patches.Polygon(points, closed=True,
                                   edgecolor=edgecolor,
                                   linewidth=linewidth,
                                   facecolor=facecolor
                                   ))

def plot_window(ax, window, edgecolor='r', linewidth=1, facecolor='none'):
    # init_corner = window.center - window.size
    width = window.size[0]
    height = window.size[1]
    init_corner = window.center.copy()
    init_corner[0] += - width/2.0 * sin(window.dir) + height/2.0 * cos(window.dir)
    init_corner[1] += - height/2.0 * sin(window.dir) - width/2.0 * cos(window.dir)
    ax.add_patch(patches.Rectangle(init_corner,
                                   width, height,
                                   angle=-degrees(window.dir-pi/2),
                                   edgecolor=edgecolor,
                                   linewidth=linewidth,
                                   facecolor=facecolor
                                   ))
    # TEMP
    # ax.plot(window.center[0], window.center[1], 'rx')
    # ax.plot([window.center[0], window.center[0]+height/2.0*cos(window.dir)],
    #         [window.center[1], window.center[1]-height/2.0*sin(window.dir)], 'r-')

def resize_image_fit_nocrop(image, dsize, padding_color=0):
    """
    Resizes an image to dshape preserving the aspect ratio and adding a black background where needed.
    :param image:
    :param dsize: (width, height) desired
    :return:
    """
    dshape = np.flip(np.array(dsize)) # rows,cols
    im_shape = np.array(image.shape)
    if len(im_shape) > 2: # if color image
        im_shape = im_shape[0:2] # keep only size (rows,cols)
    # resize as much as can
    possible_ratio = np.true_divide(dshape,im_shape)
    ratio = np.min(possible_ratio)
    resized_img = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
    # then add black padding
    res_im_shape = resized_img.shape
    h_pad = dshape[0] - res_im_shape[0]
    w_pad = dshape[1] - res_im_shape[1]
    top_border = int(ceil(h_pad / 2.0))
    bottom_border = int(floor(h_pad / 2.0))
    left_border = int(ceil(w_pad / 2.0))
    right_border = int(floor(w_pad / 2.0))
    padded_img = cv2.copyMakeBorder(resized_img,
                                    top=top_border, bottom=bottom_border,
                                    left=left_border, right=right_border,
                                    borderType=cv2.BORDER_CONSTANT, value=padding_color)
    return padded_img

def printTextOnImage(image, text, location, color, font_size=28):
    im_p = Image.fromarray(image)
    # Get a drawing context
    draw = ImageDraw.Draw(im_p)
    monospace = ImageFont.truetype("fonts/Andale Mono.ttf", font_size)
    draw.text(location, text, color, font=monospace)
    # Convert back to OpenCV image and save
    ret_image = np.array(im_p)
    return ret_image

def printLinePointsOnImage(image, line_pts, **args_polylines):
    temp_pts = np.array(line_pts, np.int32)
    temp_pts = temp_pts.reshape((-1, 1, 2))
    cv2.polylines(image, [temp_pts], **args_polylines) # isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_8)

def spline_neumann(x, y, k=3, s=0, w=None):
    """
    From https://stackoverflow.com/questions/32046582/spline-with-constraints-at-border
    """
    def guess(x, y, k, s, w=None):
        """Do an ordinary spline fit to provide knots"""
        return splrep(x, y, w, k=k, s=s)

    def err(c, x, y, t, k, w=None):
        """The error function to minimize"""
        diff = y - splev(x, (t, c, k))
        if w is None:
            diff = np.einsum('...i,...i', diff, diff)
        else:
            diff = np.dot(diff * diff, w)
        return np.abs(diff)

    i_sorted = np.argsort(x)
    sorted_x = x[i_sorted]
    sorted_y = y[i_sorted]

    t, c0, k = guess(sorted_x, sorted_y, k, s, w=w)
    x0 = sorted_x[0] # point at which zero slope is required
    xn = sorted_x[-1]
    # con = {'type': 'eq',
    #        'fun': lambda c: splev(x0, (t, c, k), der=1)
    #        #'jac': lambda c: splev(x0, (t, c, k), der=2) # doesn't help, dunno why
    #        }
    cons = ({'type': 'eq',
           'fun': lambda c: splev(x0, (t, c, k), der=1)
           # 'jac': lambda c: splev(x0, (t, c, k), der=2) # doesn't help, dunno why
           },
           {'type': 'eq',
            'fun': lambda c: splev(xn, (t, c, k), der=1)
           })
    opt = minimize(err, c0, (sorted_x, sorted_y, t, k, w), constraints=cons)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def compute_skeleton_from_image(image, dt_mask_size=3, blur_size=30):
    # DT
    image_dt = cv2.distanceTransform(image, cv2.DIST_L2, maskSize=dt_mask_size)  # 3 is mask size (3x3)
    # Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
    # image_dt_norm = cv2.normalize(image_dt, 0, 1.0, cv2.NORM_MINMAX)

    # DT ridges
    # bev_line_dt_preprocessed = cv2.blur(image, ksize=(20,20)) #cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    # image_dt_preprocessed = cv2.threshold(cv2.blur(image_dt_norm, ksize=(blur_size, blur_size)), 0, np.max(image), cv2.THRESH_BINARY)[1]
    image_dt_preprocessed = image_dt.copy()
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
    image_dt_ridges = ridge_filter.getRidgeFilteredImage(-image_dt_preprocessed)
    # road_bev_distt_ridges = cv2.threshold(road_bev_distt_ridges, 0, 1, cv2.THRESH_BINARY)[1] # try make binary...
    image_main_ridges_th = 0.5 * image_dt_ridges.max()
    _, image_dt_main_ridges = cv2.threshold(image_dt_ridges, image_main_ridges_th, 1.0, cv2.THRESH_BINARY)
    # bev_line_dt_main_ridges_adapt = cv2.adaptiveThreshold(bev_line_dt_ridges,
    #                                                   maxValue=1.0,
    #                                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                                   thresholdType=cv2.THRESH_BINARY, blockSize=21, C=-21)
    return image_dt_main_ridges, image_dt_ridges, image_dt

def morph_skeletonization(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

def rotate_image_fit(image, angleInDegrees, outSize=None):
    """
    Rotate image enlarging its width and height to fit entirely with some zero-padding
    :param image:
    :param angleInDegrees:
    :param outSize: [width, height]; if None, no scaling done (but some rounding errors could lead to a resizing of +/-1 px)
    :return:
    """
    h, w = image.shape[:2]
    img_c = (w / 2.0, h / 2.0)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    if outSize is None: # only rotation, no paddings
        # perform less expensive rotations (0,90,180,270)
        wrapped_angle = degrees(atan2(sin(radians(angleInDegrees)), cos(radians(angleInDegrees))))
        if wrapped_angle == 0:
            return image.copy(), rot
        if wrapped_angle == 90:
            return cv2.rotate(image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE), rot
        if wrapped_angle == 180:
            return cv2.rotate(image, rotateCode=cv2.ROTATE_180), rot
        if wrapped_angle == -90:
            return cv2.rotate(image, rotateCode=cv2.ROTATE_90_CLOCKWISE), rot
    # else general case:

    rad = radians(angleInDegrees)
    sine = sin(rad)
    cosine = cos(rad)

    if outSize is None:
        b_w = int((h * abs(sine)) + (w * abs(cosine)))
        b_h = int((h * abs(cosine)) + (w * abs(sine)))
    else:
        b_w = outSize[0]
        b_h = outSize[1]

    rot[0, 2] += ((b_w / 2.0) - img_c[0])
    rot[1, 2] += ((b_h / 2.0) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg, rot

def distance_point_from_line_through_points(pts, line_pts1, line_pts2):
    """
    :param pts: [[x1,y1,1], ...]
    :param line_pts1: [[x1,y1,1],...]
    :param line_pts2: [[x1,y1,1],...]
    :return: [d1, d2, ...]
    """
    return np.cross(line_pts2 - line_pts1, line_pts1 - pts) / np.linalg.norm(line_pts2 - line_pts1)

def crossratio(x1, x2, x3, x4):
    def det2(a, b): return a[0]*b[1] - a[1]*b[0]
    return det2(x1,x2)*det2(x3,x4) / det2(x1,x3)*det2(x2,x4)

def associate_data_timestamp(t, timestamps_list):
    timestamps_before = np.where(t <= timestamps_list)[0]
    if len(timestamps_before) == 0:
        return 0  # first element
    prev_i = timestamps_before[0]
    if prev_i == len(timestamps_list) - 1:
        return prev_i  # last element
    next_i = prev_i + 1
    delta_prev = t - timestamps_list[prev_i]
    delta_next = timestamps_list[next_i] - t
    if delta_prev <= delta_next:
        return prev_i
    else:
        return next_i

def polyfit_reg_first(x, y, deg, reg=0.0):

    # n_col = A.shape[1]
    # return np.linalg.lstsq(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y))

    order = int(deg) + 1
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0

    # check arguments.
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    # set up least squares equation for powers of x
    lhs = np.vander(x, order)
    rhs = y

    # scale lhs to improve condition number
    scale = np.sqrt((lhs*lhs).sum(axis=0))
    lhs /= scale

    # solve
    # c, resids, rank, s = lstsq(lhs, rhs, rcond)
    # c = (c.T/scale).T  # broadcast scale coefficients

    c = np.linalg.solve(lhs.T.dot(lhs) + reg * np.identity(lhs.shape[1]), lhs.T.dot(rhs))

    return c

def polyfit_reg(x, y, deg, reg=0.0):
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    lmax = deg
    order = lmax + 1
    van = np.polynomial.polynomial.polyvander(x, lmax)

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = y.T

    # set rcond
    rcond = len(x) * np.finfo(x.dtype).eps

    scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    # c, resids, rank, s = np.linalg.lstsq(lhs.T/scl, rhs.T, rcond) # TODO modify
    lhs_fit = lhs.T / scl
    rhs_fit = rhs.T
    c = np.linalg.solve(lhs_fit.T.dot(lhs_fit) + reg * np.identity(lhs_fit.shape[1]), lhs_fit.T.dot(rhs_fit))
    c = (c.T/scl).T

    # warn on rank reduction
    # if rank != order:
    #     msg = "The fit may be poorly conditioned"
    #     warnings.warn(msg, np.polynomial.polyutils.RankWarning, stacklevel=2)

    return c[::-1]

def compute_lines_h_from_pts_and_orientations(pts, orientations):
    return np.vstack((-np.tan(orientations), np.ones_like(orientations), np.tan(orientations) * pts[:, 0] - pts[:, 1])).T

def compute_line_h_from_two_pts(pt1, pt2):
    if pt1.shape[0] == 2 and pt2.shape[0] == 2:
        # convert to homog
        pt1 = np.insert(pt1, 2, 1)
        pt2 = np.insert(pt2, 2, 1)
    X = np.vstack((pt1, pt2))
    return scipy.linalg.null_space(X).T[0]

def line_h_to_poly1d(line_h):
    return np.poly1d(np.array([-line_h[0]/line_h[1], -line_h[2]/line_h[1]]))

def lines_h_to_poly1d(lines_h):
    lines_poly1d = []
    for i, l_h in enumerate(lines_h):
        lines_poly1d.append(line_h_to_poly1d(l_h))
    return np.array(lines_poly1d)

def vehicle2enu(points_camera_w, vehicle_pose, vehicle_cm, enu_ref):
    # convert between camera ref frame and vehicle ref frame
    points_vehicle = np.vstack((points_camera_w[:,0] - vehicle_cm[0],
                                points_camera_w[:,1] - vehicle_cm[1],
                                np.zeros_like(points_camera_w[:,0]))).T
    # convert to enu
    points_enu = vehicle_pose[0:2] + rot2d(vehicle_pose[2]).dot(points_vehicle[:, 0:2].T).T
    return points_enu

def vehicle2image(points_camera, bev_obj):
    points_w = np.hstack((points_camera, np.zeros((points_camera.shape[0], 1))))
    points_bev = bev_obj.projectWorldPointsToBevPoints(points_w)
    points_im = bev_obj.projectBevPointsToImagePoints(points_bev)
    return points_im
