
import zmq
from msgpack import loads

class ZMQSocket:
    context = zmq.Context()
    # open a req port to talk to pupil
    addr = '127.0.0.1'  # remote ip or localhost
    req_port = "50020"  # same as in the pupil remote gui

    def __init__(self):
        self.req = self.context.socket(zmq.REQ)

    def connect_tcp_server(self):
        self.req.connect("tcp://{}:{}".format(self.addr, self.req_port))
        # ask for the sub port
        self.req.send_string('SUB_PORT')
        sub_port = self.req.recv_string()
        # open a sub port to listen to pupil
        sub = self.context.socket(zmq.SUB)
        sub.connect("tcp://{}:{}".format(self.addr, sub_port))
        return sub

    def subscribe_to_event(self,sub,event=''):
        # set subscriptions to topics
        # recv just pupil/gaze/notifications
        sub.setsockopt_string(zmq.SUBSCRIBE, event)
        # sub.setsockopt_string(zmq.SUBSCRIBE, 'gaze')
        # sub.setsockopt_string(zmq.SUBSCRIBE, 'notify.')
        # sub.setsockopt_string(zmq.SUBSCRIBE, 'logging.')
        # or everything:
        # sub.setsockopt_string(zmq.SUBSCRIBE, '')

if __name__ == "__main__":
    zmq_socket = ZMQSocket()
    socket = zmq_socket.connect_tcp_server()
    while True:
        print("Trying to acquire")
        topic, msg = socket.recv_multipart()
        gaze_position = loads(msg, encoding='utf-8')
        print(gaze_position)
        print("{}, {} : ".format(topic,msg))

    #zmq_socket.subscribe_to_event(socket, 'onRecPathCreated')










