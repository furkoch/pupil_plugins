from msgpack import loads
import ctypes
from  processing.data_smoothing import Data_Smoothing
from remote.udp_socket import UDP_Socket


class SecondaryObject_Tracker:
    def __init__(self, socket):
        self.socket = socket
        self.smoother = Data_Smoothing()
        self.handy_srf = "handy"
        nPages = 15
        self.book_srf = ["page{}".format(i) for i in range(0, nPages)]

    def socket_recv(self):
        topic, msg = self.socket.recv_multipart()
        gaze_position = loads(msg, encoding='utf-8')
        return gaze_position

    def track_handy(self):
        gaze_position = self.socket_recv()
        if gaze_position['name'] == self.handy_srf:
            gazes_on_srf = gaze_position['gaze_on_srf']
            if len(gazes_on_srf) > 0:
                print("Handy is tracked")
                raw_x, raw_y = self.smoother.smooth_gaze_by_length(gazes_on_srf)
                #Send to OpenDS

    def track_book(self):
        gaze_position = self.socket_recv()
        for srf in self.book_srf:
            if gaze_position['name'] == srf:
                gazes_on_srf = gaze_position['gaze_on_srf']
                if len(gazes_on_srf) > 0:
                    print("Book at {} is tracked".format(srf))
                    raw_x, raw_y = self.smoother.smooth_gaze_by_length(gazes_on_srf)
                    # Send to OpenDS