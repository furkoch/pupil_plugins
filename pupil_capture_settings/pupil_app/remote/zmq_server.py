#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

# import time
# import zmq
#
# context = zmq.Context()
# socket = context.socket(zmq.REP)
# socket.bind("tcp://127.0.0.1:3434")
#
# while True:
#     #  Wait for next request from client
#     message = socket.recv()
#     print("Received request: {}".format(message))
#
#     #  Do some 'work'
#     time.sleep(1)
#
#     #  Send reply back to client
#     socket.send(b"World")
import zmq
import random
import sys
import time

port = "3535"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:{}".format(port))

while True:
    topic = random.randrange(10000,10002)
    messagedata = random.randrange(1,215) - 80
    print("{} {}".format(topic, messagedata))
    socket.send_string("{} {}".format(topic, messagedata))
    time.sleep(1)