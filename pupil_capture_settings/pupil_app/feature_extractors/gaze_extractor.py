from msgpack import loads
import ctypes

from  processing.data_smoothing import Data_Smoothing
import time
class Gaze_Extractor:

    def __init__(self, socket, udp_socket):
        self.surface_name = "screen"
        self.smoother = Data_Smoothing()
        self.socket = socket
        self.x_dim, self.y_dim = self.get_screen_size()
        self.smooth_x, self.smooth_y = 0.5, 0.5
        self.udp_socket = udp_socket
        self.frequent_gazes  = []
        self.max_gaze_count = 7

    def get_screen_size(self):
        user32 = ctypes.windll.user32
        screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        return screensize

    def process_gaze_from_screen(self, gaze_on_screen):
        #gaze_on_screen = gaze_position['gaze_on_srf']
        if len(gaze_on_screen) > 0:
            recent_gaze = gaze_on_screen[0]
            raw_x, raw_y = self.smoother.normalize_gaze_by_length(gaze_on_screen)

            self.smooth_x += 0.35 * (raw_x - self.smooth_x)
            self.smooth_y += 0.35 * (raw_y - self.smooth_y)

            x = self.smooth_x
            y = self.smooth_y

            y = 1 - y  # inverting y so it shows up correctly on screen

            x = round(min(1, max(0, x)), 2)
            y = round(min(1, max(0, y)), 2)

            # x *= int(self.x_dim)
            # y *= int(self.y_dim)
            ts = int(time.time())
            print("X : {}\t,Y : {}".format(x, y))
            self.send_udp_data(x, y, ts)
            fr_gazes = self.frequent_gazes
            self.frequent_gazes = []


    def extract_gaze_from_screen(self):
        topic, msg = self.socket.recv_multipart()
        fixation = loads(msg, encoding='utf-8')

        #self.send_udp_data(x, y, ts)

        print(fixation['timestamp'])

        norm_pos = fixation['norm_pos']

        self.send_udp_data(norm_pos[0],norm_pos[1], fixation['timestamp'])





        # if gaze_position['name'] == self.surface_name:
        #     gaze_on_screen = gaze_position['gaze_on_srf']
        #     if len(gaze_on_screen) > 0:
        #         recent_gaze = gaze_on_screen[0]
        #         raw_x, raw_y = self.smoother.normalize_gaze_by_length(gaze_on_screen)
        #
        #         self.smooth_x += 0.35 * (raw_x - self.smooth_x)
        #         self.smooth_y += 0.35 * (raw_y - self.smooth_y)
        #
        #         x = self.smooth_x
        #         y = self.smooth_y
        #
        #         y = 1 - y  # inverting y so it shows up correctly on screen
        #
        #         x = round(min(1, max(0, x)), 2)
        #         y = round(min(1, max(0, y)), 2)
        #
        #         # x *= int(self.x_dim)
        #         # y *= int(self.y_dim)
        #         ts = int(time.time())
        #         print("X : {}\t,Y : {}".format(x, y))
        #
        #         self.send_udp_data(x, y, ts)
        #         fr_gazes = self.frequent_gazes
        #         self.frequent_gazes = []

                # if len(self.frequent_gazes)<self.max_gaze_count:
                #     #Add more gazes
                #     self.frequent_gazes.append(recent_gaze)
                #     return
                # else:
                #     # there may be multiple gaze positions per frame, so we could average them.
                #     raw_x,raw_y = self.smoother.normalize_gaze_by_length(self.frequent_gazes)
                #
                #     self.smooth_x += 0.35 * (raw_x - self.smooth_x)
                #     self.smooth_y += 0.35 * (raw_y - self.smooth_y)
                #
                #     x = self.smooth_x
                #     y = self.smooth_y
                #
                #     y = 1 - y  # inverting y so it shows up correctly on screen
                #
                #     x = round(min(1, max(0, x)), 2)
                #     y = round(min(1, max(0, y)), 2)
                #
                #     # x *= int(self.x_dim)
                #     # y *= int(self.y_dim)
                #     ts = int(time.time())
                #     print("X : {}\t,Y : {}".format(x, y))
                #     self.send_udp_data(x,y,ts)
                #     self.frequent_gazes = []


    def send_udp_data(self, x,y,ts):
        self.udp_socket.send_message(str((x, y, ts)))

    def extract_pupil_data(self):
        topic = self.socket.recv_string()
        msg = self.socket.recv()
        msg = loads(msg, encoding='utf-8')
        print("\n{}: {}".format(topic, msg))


