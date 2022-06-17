#
# Author: Paolo Cudrano (archnnj)
#

import numpy as np
import cv2
from features_cnn.cnn1.network_no_ros import image_converter as image_converter1
from features_cnn.cnn2.network_no_ros import image_converter as image_converter2

class FeatureExtraction:
    CNN_FEATURES_POSTPROCESS_NONE = 0
    CNN_FEATURES_POSTPROCESS_MORPH = 1
    CNN_FEATURES_POSTPROCESS_DT = 2
    CNN_FEATURES_POSTPROCESS_MORPH_CPP = 3 # TEST

    feature_cnn = None
    feature_cnn_line_mask_th = None
    feature_cnn_line_mask_size_comp = None

    @classmethod
    def init(cls, line_mask_th, line_mask_size_comp, cnn_id=1):
        if cnn_id == 1:
            image_converter = image_converter1
        elif cnn_id == 2:
            image_converter = image_converter2
        cls.feature_cnn_line_mask_th = line_mask_th
        cls.feature_cnn_line_mask_size_comp = line_mask_size_comp
        cls.feature_cnn = image_converter(threshold_param=line_mask_th, size_param=line_mask_size_comp)

    @classmethod
    def extraction(cls, frame, bev_obj, postprocess_mode=CNN_FEATURES_POSTPROCESS_NONE):
        # get feature points from cnn
        line_feature_mask, line_feature_predictions = cls.feature_cnn.predict(frame)
        bev_line_feature_mask, bev_line_feature_predictions =\
            cls._postprocess(line_feature_mask, line_feature_predictions, bev_obj, postprocess_mode)
        return line_feature_mask, line_feature_predictions, bev_line_feature_mask, bev_line_feature_predictions

    ### Private ###
    @classmethod
    def _postprocess(cls, line_feature_mask, line_feature_predictions, bev_obj, postprocess_mode):
        if postprocess_mode == FeatureExtraction.CNN_FEATURES_POSTPROCESS_NONE:
            # no post-processing
            bev_line_feature_predictions = bev_obj.computeBev(line_feature_predictions)
            bev_line_feature_mask = cls.feature_cnn.cleaner(bev_line_feature_predictions, cls.feature_cnn_line_mask_th)
        # elif postprocess_mode == FeatureExtraction.CNN_FEATURES_POSTPROCESS_MORPH_CPP:
        #     strel_size_morph1 = np.round(0.010 * self.bev_obj.getBevImageSize()[1]).astype(np.int)
        #     strel_size_morph2 = np.round(0.020 * self.bev_obj.getBevImageSize()[1]).astype(np.int)
        #     bev_line_feature_mask, bev_line_feature_predictions = \
        #         feature_postprocessing.postprocessing_pipeline(line_feature_mask,
        #                                                        line_feature_predictions,
        #                                                        self.bev_obj,
        #                                                        [strel_size_morph1, strel_size_morph2],
        #                                                        configs.line_mask_th,
        #                                                        self.feature_cnn.size_param)
        elif postprocess_mode == FeatureExtraction.CNN_FEATURES_POSTPROCESS_MORPH:
            # post-processing projecting to bev original prediction mask, then thresholding and applying morph op.
            bev_line_feature_predictions = bev_obj.computeBev(line_feature_predictions)
            bev_line_feature_mask = cls.feature_cnn.cleaner(bev_line_feature_predictions, cls.feature_cnn_line_mask_th)
            strel_size_morph1 = np.round(0.010 * bev_obj.getBevImageSize()[1]).astype(np.int)
            strel_size_morph2 = np.round(0.020 * bev_obj.getBevImageSize()[1]).astype(np.int)
            # if not configs.use_cpp:
            bev_line_feature_mask1 = bev_line_feature_mask.copy()
            bev_line_feature_mask1 = cv2.morphologyEx(
                bev_line_feature_mask1, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strel_size_morph1, strel_size_morph1)))
            bev_line_feature_mask1 = cv2.morphologyEx(
                bev_line_feature_mask1, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strel_size_morph2, strel_size_morph2)))
            bev_line_feature_mask = bev_line_feature_mask1
            # else:
            #     bev_line_feature_mask2 = bev_line_feature_mask.copy()
            #     feature_postprocessing.postprocess_cnn_predictions(bev_line_feature_mask2, bev_line_feature_mask2, [strel_size_morph1, strel_size_morph2])
            #     bev_line_feature_mask = bev_line_feature_mask2
        return bev_line_feature_mask, bev_line_feature_predictions
