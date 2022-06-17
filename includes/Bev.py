#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
from math import cos, sin, pi

class Bev:
    """
    Bird's Eye View class. Constructed on top of a Camera instance, it enhances it with the capability to construct a
    BEV image and project points between the BEV plane and the world frame.
    Notice that the world coordinate frame is assumed to be centered at the tip of the car, on the front side, at ground
    level; the x axis is pointing forward, while the y axis is pointing towards the left of the car. The z axis is,
    obviously, pointing upwards.
    Notice moreover that, each time a method requires some points as parameters, they are passed as row vectors
    (stacked if necessary).
    """

    def __init__(self, camera, outView, outImageSize):
        """
        Initializes the BEV object with a camera object, the world coordinates to be projected and the size of the BEV
        output image to be produced, all necessary information for the projection of images and points between the
        two frames.
        :param camera: an instance of Camera
        :param outView: [dist_min, dist_max, -right_max, +left_max]; distances measured in meters in the world frame,
        they define the portion of world ground plane to be centered on.
        :param outImageSize: [mrows,ncols]; dimensions of the output image. One of the two coordinates could assume the
        value np.nan: in such case, its value is automatically computed to conserve the aspect ratio of the observed
        patch in the world ground.
        """
        self.camera = camera
        self.outView = outView
        self.outImageSize = outImageSize
        self.computeBevProjection(outView, outImageSize)

    def updateCamera(self, camera):
        self.camera = camera
        self.computeBevProjection(self.outView, self.outImageSize)

    def getCamera(self):
        return self.camera

    def getCameraToBevProjection(self):
        return self.H_bev

    def getBevToCameraProjection(self):
        return np.linalg.inv(self.H_bev)

    def getWorldToBevProjection(self):
        return self.H_bev.dot(self.camera.getWorldToCameraProjection()) # @

    def getBevToWorldDirProjection(self):
        return np.linalg.inv(self.getWorldToBevProjection()[:,0:3]) # if H = (M|m), return M

    def computeSize(self, outView, outImageSize):
        worldHW = np.abs([outView[1] - outView[0], outView[3] - outView[2]])
        if not np.any(np.bitwise_or(np.isnan(outImageSize), np.asarray(outImageSize) < 0)):
            self.scaleXY = np.flip((np.array(outImageSize) - 1) / worldHW)
            self.outSize = outImageSize
        else:
            notSpecifSize_i = np.argwhere(np.bitwise_or(np.isnan(outImageSize), np.asarray(outImageSize) < 0))[0][0]
            specifSize_i = np.argwhere(np.logical_not(np.bitwise_or(np.isnan(outImageSize), np.asarray(outImageSize) < 0)))[0][0]
            scale = float(outImageSize[specifSize_i] - 1) / worldHW[specifSize_i]
            self.scaleXY = [scale, scale]

            otherDim = int(round(scale * worldHW[notSpecifSize_i]) + 1)
            self.outSize = outImageSize
            self.outSize[notSpecifSize_i] = otherDim

    def computeBevProjection(self, outView, outImageSize):
        self.computeSize(outView, outImageSize)
        P = self.camera.getWorldToCameraProjection()
        tform2D_toimage = P[:, [0, 1, 3]] # drop Z, 2d homography
        tform2D_tovehicle = np.linalg.inv(tform2D_toimage)
        adjTform = np.float32([[0, -1, 0],
                               [-1, 0, 0],
                               [0, 0, 1]]).T
        bevTform = adjTform.dot(tform2D_tovehicle) # tform2D_tovehicle @ adjTform     # @
        # scaleXY = np.float32([1, 1]);
        dYdXVehicle = np.float32([outView[3], outView[1]]) # top left corner in world -> becomes 0,0 in image --> tXY is pixel translation
        tXY = self.scaleXY * dYdXVehicle  # X,Y translation in pixels
        viewMatrix = np.float32([[self.scaleXY[0], 0, 0],
                                 [0, self.scaleXY[1], 0],
                                 [tXY[0] + 1, tXY[1] + 1, 1]]).T
        self.H_bev = viewMatrix.dot(bevTform)  # bevTform @ viewMatrix;      # @

    def getBevImageSize(self):
        return tuple(np.flip(self.outSize))

    def computeBev(self, img, flags=cv2.INTER_CUBIC):
        return cv2.warpPerspective(img, self.H_bev, self.getBevImageSize(), flags=flags)

    def projectImagePointsToBevPoints(self, image_points):
        image_points_h = np.hstack((image_points, np.ones((image_points.shape[0], 1)))).T
        bev_points_h = self.H_bev.dot(image_points_h)   # @
        bev_points = bev_points_h[0:2] / bev_points_h[2]
        bev_points = bev_points.T
        return bev_points

    def projectBevPointsToImagePoints(self, bev_points):
        bev_points_h = np.hstack((bev_points, np.ones((bev_points.shape[0], 1)))).T
        image_points_h = np.linalg.inv(self.H_bev).dot(bev_points_h) # @
        image_points = image_points_h[0:2,:] / image_points_h[2,:]
        image_points = image_points.T
        return image_points

    def projectWorldPointsToBevPoints(self, world_points): # world_points = [[x1, y1, z1],...,[xn,yn,zn]]
        image_points = self.camera.projectWorldPointsToImagePoints(world_points)
        bev_points = self.projectImagePointsToBevPoints(image_points)
        return bev_points

    def projectBevPointsToWorldGroundPlane(self, bev_points):
        world_points_z0_h = np.linalg.inv(self.camera.getWorldToCameraProjection()[:, [0, 1, 3]]).dot(
                                    np.insert(self.projectBevPointsToImagePoints(bev_points), 2, 1, axis=1).T
                                    ).T
        world_points_h = np.insert(world_points_z0_h, 2, 0, axis=1)  # insert z=0 column
        world_points = world_points_h[:,0:3] / world_points_h[:,[3]]
        return world_points
