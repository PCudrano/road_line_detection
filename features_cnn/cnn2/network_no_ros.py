#!/usr/bin/env python
######versions 
from __future__ import print_function

import cv2
# from std_msgs.msg import String
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from numpy import newaxis
import time
from datetime import timedelta

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_model_lines():

    # NOTE: works on Grayscale images
    inputs = Input((  int(376/2), int(640/2),1))
 
    conv1 = Conv2D(8, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = Conv2D(8, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    #conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(pool1)
    #conv2 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(pool2)
    #conv3 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv3)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    #conv4 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool3)
    #conv4 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(16, (3, 3), padding="same", activation="relu")(pool1)
    conv5 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv5)
     
    #up6 = Concatenate(axis = 3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    #conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")(up6)
    #conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv6)
     
    #up7 = Concatenate(axis = 3)([UpSampling2D(size=(2, 2))(conv5), conv3])
    #conv7 = Conv2D(32, (3, 3), padding="same", activation="relu")(up7)
    #conv7 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv7)
     
    #up8 = Concatenate(axis = 3)([UpSampling2D(size=(2, 2))(conv5), conv2])
    #conv8 = Conv2D(16, (3, 3), padding="same", activation="relu")(up8)
    #conv8 = Conv2D(16, (3, 3), padding="same", activation="relu")(conv8)
    
    up9 = Concatenate(axis = 3)([UpSampling2D(size=(2, 2))(conv5), conv1])
    conv9 = Conv2D(8, (3, 3), padding="same", activation="relu")(up9)
    conv9 = Conv2D(8, (3, 3), padding="same", activation="relu")(conv9)
     
    conv10 = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(conv9)
     
    model_lines = Model(inputs=inputs, outputs=conv10)
    
    model_lines.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5))

    print(model_lines.summary()) 
    
    return model_lines

class image_converter:

  def __init__(self, threshold_param=65, size_param=20):
    
    global model_lines

    self.threshold_param = threshold_param
    self.size_param=size_param
   
    model_lines = create_model_lines()
    # model_lines._make_predict_function()
    model_lines.load_weights("./features_cnn/cnn2/weight.h5")

    # cap = cv2.VideoCapture('vid.mp4')
    # cap = cv2.VideoCapture('./data/v1_2.mp4')
    #
    # if (cap.isOpened()== False):
    #   print("Error opening video stream or file")
    #
    # while(cap.isOpened()):
    # # Capture frame-by-frame
    #   ret, frame = cap.read()
    #   if ret == True:
    #     image_converter.predict (self,frame)

  # def predict(self,frame):
  #   print ("asddd")
  #
  #   global model_lines
  #
  #   final_resolution_row = int(376/2)
  #   final_resolution_col = int(640/2)
  #
  #   (rows,cols,channels) = frame.shape
  #
  #   img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #   #cv2.imwrite ("/home/mentasti/Data/spiking/image.png", img)
  #   img = np.delete(img, slice(0, 104), axis=0)
  #   (rows2,cols2) = img.shape
  #   img = cv2.resize (img, (final_resolution_col, final_resolution_row), interpolation=cv2.INTER_CUBIC) #376,640
  #   imgs_t = np.ndarray((1, final_resolution_row, final_resolution_col,1), dtype=np.uint8)
  #   img = img[:,:,newaxis]
  #   imgs_t[0]=img
  #   img_mask_lines= model_lines.predict (imgs_t,batch_size=1 ,verbose =1)
  #   mask_lines = img_mask_lines[0]
  #   mask_lines = np.uint8(mask_lines*255)
  #
  #   size_param_lines =20
  #   threshold_param_lines = 65
  #
  #
  #     #size_param = rospy.get_param("size")
  #     #threshold_param= rospy.get_param("thr")
  #
  #   mask_lines = image_converter.cleaner (self,mask_lines, threshold_param_lines,size_param_lines,img)
  #
  #   mask_lines = cv2.resize (mask_lines, (cols2, rows2), interpolation=cv2.INTER_CUBIC)
  #   out = np.zeros ((104,cols))
  #   print (mask_lines.shape)
  #   print (out.shape)
  #   out = np.append(out, mask_lines,  axis=0)
  #   lines_found = frame
  #   #lines_found = np.zeros((256,512,channels)).astype(np.uint8)
  #   lines_found[out!=0] = (0,0,255)

    # cv2.namedWindow ("Image window2", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow ("Image window2", 1500,750)
    # cv2.imshow("Image window2", lines_found)

    # cv2.waitKey(3)

  def predict(self, frame):
      # print("asddd")

      global model_lines

      self.frame_shape = frame.shape

      final_resolution_row = int(376 / 2)
      final_resolution_col = int(640 / 2)

      (rows, cols, channels) = frame.shape

      # preprocess
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # cv2.imwrite ("/home/mentasti/Data/spiking/image.png", img)
      img = np.delete(img, slice(0, 104), axis=0) # remove top 104 pixels
      (rows2, cols2) = img.shape
      img = cv2.resize(img, (final_resolution_col, final_resolution_row), interpolation=cv2.INTER_CUBIC)  # 376,640

      # predict
      imgs_t = np.ndarray((1, final_resolution_row, final_resolution_col, 1), dtype=np.uint8)
      img = img[:, :, newaxis]
      imgs_t[0] = img
      img_mask_lines = model_lines.predict(imgs_t, batch_size=1, verbose=1)
      mask_lines = img_mask_lines[0]
      mask_lines = np.uint8(mask_lines * 255)
      size_param_lines = self.size_param # size_param_lines = 20
      threshold_param_lines = self.threshold_param # threshold_param_lines = 65
      # size_param = rospy.get_param("size")
      # threshold_param= rospy.get_param("thr")
      # mask_lines = image_converter.cleaner(self, mask_lines, threshold_param_lines, size_param_lines, img)
      orig_mask_lines = mask_lines.copy()

      mask_lines = self.cleaner(mask_lines, threshold_param_lines, size_param_lines) #, img)

      # postprocess
      mask_lines = cv2.resize(mask_lines, (cols2, rows2), interpolation=cv2.INTER_CUBIC)
      # mask_lines = cv2.resize(mask_lines, dsize=(self.frame_shape[1], self.frame_shape[0]), interpolation=cv2.INTER_CUBIC)
      mask_lines = cv2.copyMakeBorder(mask_lines, 104, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)  # add 104px zeros on top
      out_mask_lines = mask_lines

      # postprocess orig_mask_lines
      orig_mask_lines = cv2.resize(orig_mask_lines, (cols2, rows2), interpolation=cv2.INTER_CUBIC)
      orig_mask_lines = cv2.copyMakeBorder(orig_mask_lines, 104, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0) # add 104px zeros on top

      # overlay on frame
      # out = np.zeros((104, cols))
      # print(mask_lines.shape)
      # print(out.shape)
      # out = np.append(out, mask_lines, axis=0)
      # lines_found = frame
      # # lines_found = np.zeros((256,512,channels)).astype(np.uint8)
      # lines_found[out != 0] = (0, 0, 255)

      return out_mask_lines, orig_mask_lines

  def cleaner (self,img,thr,size=None): #,orig):
    if size is None:
      size = self.size_param

    #filtro tutto quello che sta sotto thr
    th2, ret = cv2.threshold (img, thr, 255, cv2.THRESH_TOZERO)
    # return ret

    # TODO should I leave this part??
    #cerco componenti
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(ret, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    img2 = np.zeros((ret.shape))
    #tengo solo componenti piu grandi di size
    for j in range(0, nb_components):
        if sizes[j] >= size:
            img2[output == j + 1] = 255
    img2 = np.uint8(img2*1)

    #sistemo un po'l'immagine
    # result = orig.copy()
    # result[img2!=0] = 255 #(255,0,255)

    return img2

# def main(args):
#
#   ic = image_converter()
#
# if __name__ == '__main__':
#     main(sys.argv)
