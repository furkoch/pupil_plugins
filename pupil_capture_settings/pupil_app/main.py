from remote.zmq_socket import ZMQSocket
from remote.udp_socket import UDP_Socket
from feature_extractors.gaze_extractor import Gaze_Extractor
from feature_extractors.secondary_objects import  SecondaryObject_Tracker
from msgpack import loads

if __name__ == "__main__":
    zmq_socket = ZMQSocket()
    socket = zmq_socket.connect_tcp_server()
    udp_socket = UDP_Socket()

    zmq_socket.subscribe_to_event(socket,'fixations')
    zmq_socket.subscribe_to_event(socket, 'gaze')
    gaze_extractor = Gaze_Extractor(socket,udp_socket)
    so_tracker = SecondaryObject_Tracker(socket)

    while True:
        gaze_extractor.extract_gaze_from_screen()
        #so_tracker.track_handy()