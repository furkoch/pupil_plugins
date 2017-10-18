import sys
sys.path.append('C:\\Projects\\repo\\sensorics\\EyeTracker\\pupil\\pupil_src\\shared_modules')
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings')
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings\\pupil_app')
sys.path.append('C:\\Users\\drivesense\\pupil_capture_settings\\pupil_app\\saliency')
sys.path.append('C:\\Users\\drivesense\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages')

#from pupil_app.saliency.saliency_detector import Saliency_Detector
from plugin import Plugin
from pyglui import ui
import cv2

class Saliency_Detector_Plugin(Plugin):

    def __init__(self, g_pool, itti_koch_method = True, vanishing_point = True, spectral_residual = False, vp_coverage_weight = 0.8):
        super().__init__(g_pool)
        self.order = .8
        self.pupil_display_list = []
        self.itti_koch_method = itti_koch_method
        self.vanishing_point = vanishing_point
        self.spectral_residual = spectral_residual
        self.vp_coverage_weight = vp_coverage_weight
        self.show_saliency = True
        self.frames_fifo = []
        self.current_frame = None

    def recent_events(self,events):
        # for pt in events.get('gaze_positions',[]):
        #     self.pupil_display_list.append((pt['norm_pos'] , pt['confidence']))
        # self.pupil_display_list[:-3] = []
        frame = events.get("frame")


        #cv2.imshow('Saliency map', cv2.flip(frame.img, 1))

        # if frame and frame.jpeg_buffer and self.show_saliency:
        #     self.current_frame = frame
        #     self.notify_all({'subject': 'frame_changed'})

    def on_notify(self, notification):
        pass
        # if notification['subject'] == 'frame_changed':
        #     sal_etector = Saliency_Detector(Saliency_Detector.bottom_up_models['ik'])
        #     sal_etector.detect_saliency(self.current_frame.img)
        #     cv2.imshow('Saliency map', cv2.flip(self.current_frame.img, 1))

    def open_close_saliency_window(self):
        if self.current_frame is not None:
            self.show_saliency = True
            cv2.imshow('Input image', cv2.flip(self.current_frame.img, 1))

    def set_vp_coverage_weight(self, vp_weight):
        self.vp_coverage_weight = vp_weight

    def init_gui(self):
        self.menu = ui.Growing_Menu('Saliency Detector')
        self.menu.collapsed = True
        self.menu.append(ui.Button('Open Saliency Window', self.open_close_saliency_window))
        self.saliency_model_menu = ui.Growing_Menu('Saliency Model')
        self.saliency_model_menu.collapsed = True
        self.saliency_model_menu.append(ui.Switch('itti_koch_method', self, on_val=True, off_val=False, label='Itti-Koch Model'))
        self.saliency_model_menu.append(ui.Switch('spectral_residual', self, on_val=True, off_val=False, label='Spectral Residual'))
        # self.menu.append(ui.Selector('format', self, selection=["jpeg", "yuv", "bgr", "gray"],
        #                              labels=["JPEG", "YUV", "BGR", "Gray Image"], label='Format'))

        self.menu.append(self.saliency_model_menu)
        self.feature_maps_menu = ui.Growing_Menu('Feature Maps')
        self.feature_maps_menu.collapsed = True
        self.feature_maps_menu.append(ui.Switch('vanishing_point', self, on_val=True, off_val=False, label='Vanishing Point'))

        self.feature_maps_menu.append(ui.Slider('vp_coverage_weight', self, min=0.0, step=0.1, max=1.0, label='VP Coverage Weight',
                                       setter=self.set_vp_coverage_weight))


        self.menu.append(self.feature_maps_menu)
        self.g_pool.sidebar.append(self.menu)


    def deinit_gui(self):
        if self.menu:
            self.g_pool.sidebar.remove(self.menu)
            self.menu = None
        if self.button:
            self.g_pool.quickbar.remove(self.button)
            self.button = None


    def gl_display(self):
        for pt,a in self.pupil_display_list:
            #This could be faster if there would be a method to also add multiple colors per point
            draw_points_norm([pt],
                        size=35,
                        color=RGBA(1.,.2,.4,a))

    def get_init_dict(self):
        return {}