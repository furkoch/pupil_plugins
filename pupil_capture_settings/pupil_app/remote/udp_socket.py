import socket

class UDP_Socket:
    def __init__(self, host="127.0.0.1", port=2010):
        self.udp_host = host
        self.udp_port = port
        self.sock = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP

    def is_connected(self):
        return self.sock is not None

    def send_message(self, jsonMessage):
        if self.sock is None:
            return None
        self.sock.sendto(jsonMessage.encode(),(self.udp_host, self.udp_port))