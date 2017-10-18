from ws4py.client.threadedclient import WebSocketClient
import json
class WSClient(WebSocketClient):
    def __init__(self, url, gazeRecorder):
        super().__init__(url)
        self.gaze_recorder = gazeRecorder

    def opened(self):
        self.gaze_recorder.on_open(self)
        # self.closed = onOpened
        # self.send(json.dumps({"event": "registerForUpdates", "value": "onRecPathCreated"}))

    def closed(self, code, reason=None):
        self.gaze_recorder.on_close(self,code,reason)
        #print("Closed down", code, reason)

    def received_message(self, m):
        self.gaze_recorder.on_message_received(self, m)