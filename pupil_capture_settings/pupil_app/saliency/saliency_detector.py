import sys
<<<<<<< HEAD
sys.path.append('F:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings\\pupil_app')
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings\\pupil_app\\saliency\top-down')
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings\\pupil_app\\saliency\\itti_koch_saliency')
sys.path.append('C:\\Users\\drivesense\\AppData\\Roaming\\Python\\Python36\\site-packages')

import cv2
from saliency.itti_koch_model.itti_koch_saliency import Itti_Koch_Saliency
from saliency.top_down.vp_detector import VP_Detector

# from file_methods import Persistent_Dict,load_object
# from pyglui.cygl.utils import draw_points,draw_polyline,RGBA
# from pyglui import ui
# from OpenGL.GL import GL_POLYGON
# from methods import normalize,denormalize
#
# from glfw import *
# from square_marker_detect import detect_markers,detect_markers_robust, draw_markers,m_marker_to_screen
# from reference_surface import Reference_Surface

=======
sys.path.append('C:\\Users\\Win10 Pro x64\\pupil_capture_settings\\pupil_app')
sys.path.append('C:\\Users\\Win10 Pro x64\\pupil_capture_settings\\pupil_app\\saliency\top-down')
sys.path.append('C:\\Users\\Win10 Pro x64\\pupil_capture_settings\\pupil_app\\saliency\\itti_koch_saliency')
from saliency.itti_koch_model.itti_koch_saliency import Itti_Koch_Saliency
from saliency.top_down.vp_detector import VP_Detector
from saliency.marker_detector import  DS_Marker_Detector
>>>>>>> e1a726945483bfb6a6b7813f680aac3b2d2d4ba5
import numpy as np
import cv2

class Saliency_Detector:

    bottom_up_models = {'ik':'itti_koch','sr':'spectral_residual','gbvs':'graph_based_saliency'}
    top_down_models = {'vp':'vanishing_point','swd':'steering_wheels_detection'}

    def __init__(self, bottom_up_model):
        #self.input_image = input_image
        self.bottom_up_model = bottom_up_model

    def get_bottom_up_saliency(self, input_image, width, height):
        bottom_up_saliency = None
        if self.bottom_up_model == self.bottom_up_models['ik']:
            bottom_up_saliency = Itti_Koch_Saliency(input_image, width, height)
        return  bottom_up_saliency

    def get_top_down_saliency(self, input_image):
        vp_detector = VP_Detector(input_image)
        vp_img = vp_detector.detect_vp()
        return vp_img

    def get_saliency_weights(self):
        #TODO Calculate statisitically
        bottom_up, top_down = 0.8,0.2
        return bottom_up, top_down


    # 2D Gaussian function
    def twoD_Gaussian(x, y, xo, yo, sigma_x, sigma_y):
        a = 1. / (2 * sigma_x ** 2) + 1. / (2 * sigma_y ** 2)
        c = 1. / (2 * sigma_x ** 2) + 1. / (2 * sigma_y ** 2)
        g = np.exp(- (a * ((x - xo) ** 2) + c * ((y - yo) ** 2)))
        return g.ravel()


    def heat_map(self,src):
        dst = cv2.GaussianBlur(src, (5, 5), 10)
        dst = cv2.applyColorMap(dst, cv2.COLORMAP_JET)
        return dst

    def vis_heatmap(self, frame,vp, bottom_up_saliency, top_down_saliency):
        ik_binalized = bottom_up_saliency.get_binalized_saliency_map()
        vp_binalized = top_down_saliency.get_binalized_saliency_map()
        b,t = self.get_saliency_weights()
        heatmap = self.heat_map(ik_binalized + vp_binalized)
        cv2.imshow("Heatmap", heatmap)


    def vis_binalized(self, bottom_up_saliency, top_down_saliency):
        bn_bottom_up_map = bottom_up_saliency.get_binalized_saliency_map()
        # bn_top_down_map = top_down_saliency.get_binalized_saliency_map()
        # bottom_up_weight, top_down_weight = self.get_saliency_weights()
        #cv2.imshow("Binalized", bottom_up_weight*bn_bottom_up_map+top_down_weight*bn_top_down_map)
        cv2.imshow("IK+VP Binalized", bn_bottom_up_map)

    def vis_saliency_map(self, frame, bottom_up_saliency, top_down_saliency):
        ik_saliency_map = bottom_up_saliency.get_saliency_map()
        cv2.imshow("Saliency map", ik_saliency_map)


    def detect_saliency(self, frame, is_screen_detected):
        vp_detector = None
        if is_screen_detected:
            vp_detector = VP_Detector(frame)
            vp_detector.detect_vp()
        bottom_up_saliency = self.get_bottom_up_saliency(frame, frame.shape[1], frame.shape[0])
        self.vis_saliency_map(frame, bottom_up_saliency, vp_detector)
        self.vis_binalized(bottom_up_saliency, vp_detector)

    def get_salient_region(self):
        src = self.input_image
        # get a binarized saliency map
        binarized_SM = self.get_binalized_saliency_map()
        # GrabCut
        img = src.copy()
        mask = np.where((binarized_SM != 0), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        rect = (0, 0, 1, 1)  # dummy
        iterCount = 1
        cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgdmodel, fgdModel=fgdmodel, iterCount=iterCount,
                    mode=cv2.GC_INIT_WITH_MASK)
        # post-processing
        mask_out = np.where((mask == cv2.GC_FGD) + (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img, img, mask=mask_out)
        return output

    def get_binalized_saliency_map(self, saliency_map):
        # convert scale
        SM_I8U = np.uint8(255 * saliency_map)
        # binarize
        thresh, binarized_saliency_map = cv2.threshold(SM_I8U, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized_saliency_map

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)

    saliency_detector = Saliency_Detector(Saliency_Detector.bottom_up_models['ik'])

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #Initialize
        if frame is None:
            print("Frame is None")
            break
<<<<<<< HEAD
        #cv2.imshow("Frame", frame)
        saliency_detector.detect_saliency(frame)
=======

        marker_detector = DS_Marker_Detector()
        cv2.imshow('Frame',frame)
        is_screen_detected = marker_detector.screen_detected_by_multi_markers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        saliency_detector.detect_saliency(frame, is_screen_detected)
>>>>>>> e1a726945483bfb6a6b7813f680aac3b2d2d4ba5

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


