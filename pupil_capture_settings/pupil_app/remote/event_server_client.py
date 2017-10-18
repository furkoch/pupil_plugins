from remote.websocket_client import WebSocketClient

class EventServerClient(WebSocketClient):

    def _on_message(self, msg):
        print(msg)

    def _on_connection_success(self):
        print('Connected!')

    def _on_connection_close(self):
        print('Connection closed!')

    def _on_connection_error(self, exception):
        print('Connection error: %s', exception)