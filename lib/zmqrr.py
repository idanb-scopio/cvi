# ZMQ Robust Request Response helper class
# Wrapper for ZMQ socket with easy api for recv(msg, timeout)
# If recv fails, the wrapper closes and reinitialize the connection
import zmq
import json


class TimeoutException(Exception):
    pass


class ZmqRR:

    def __init__(self, context=None):
        if not context:
            self.context = zmq.Context()
        else:
            self.context = context

        self.socket = None
        self.poller = zmq.Poller()
        self.endpoint = None

    def connect(self, endpoint):
        self.endpoint = endpoint
        self._init_connect_socket()

    def send(self, msg):
        self.socket.send(msg)

    def recv(self, timeout_ms):
        # read reply from socket
        socks = dict(self.poller.poll(timeout_ms))
        if self.socket in socks and socks[self.socket] == zmq.POLLIN:
            # there's a message waiting to be read
            msg = self.socket.recv()
            return msg

        # otherwise - timed out. close and reopen socket to clean pending outgoing
        # zmq messages.
        self.poller.unregister(self.socket)
        self.socket.close()
        self._init_connect_socket()
        return None

    def _init_connect_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.socket.connect(self.endpoint)
        self.poller.register(self.socket, zmq.POLLIN)


class JsonRRSocket(ZmqRR):

    def send(self, msg):
        if not type(msg) is dict:
            raise ValueError('input for send must be a dictionary type')

        message = json.dumps(msg).encode('ascii')
        super().send(message)

    def recv(self, timeout_ms):
        json_message = super().recv(timeout_ms=timeout_ms)
        if json_message is None:
            raise TimeoutException
        try:
            reply_dict = json.loads(json_message)
        except json.JSONDecodeError:
            raise ValueError(f'not a JSON: [{json_message}]')
        return reply_dict
