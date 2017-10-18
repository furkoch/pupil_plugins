import sys
sys.path.append('C:\\Soft\\pupil_v0912_windows_x64\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Soft\\pupil_v0912_windows_x64\\pupil\\pupil_src')
import cv2
import numpy as np
from glfw import *
from shared_modules.square_marker_detect import detect_markers

class DS_Marker_Detector:
    def __init__(self):
        self.aperture = 11
        self.min_screen_markers = 4
        self.min_marker_perimeter=1
        self.screen_markers = []
        self.wheel_markers = []

    #Use this one when only screen is marked not steering wheel(real driving scenario)
    def screen_detected_by_one_marker(self,gray):
        markers = detect_markers(gray, grid_size=5, aperture=self.aperture,
            min_marker_perimeter=self.min_marker_perimeter)
        return len(markers) > 0

    #Use this one for experiment
    def screen_detected_by_multi_markers(self, gray):
        markers = detect_markers(gray, grid_size=5, aperture=self.aperture,
                                      min_marker_perimeter=self.min_marker_perimeter)
        return len(markers) > self.min_screen_markers

    def screen_detected_by_one_marker_buf(self,gray):
        markers = detect_markers(gray, grid_size=5, aperture=self.aperture,
            min_marker_perimeter=self.min_marker_perimeter)
        if len(markers)==0:
            return False
        for sm in self.screen_markers:
            if sm in markers:
                return True


if __name__=="__main__":
    pass