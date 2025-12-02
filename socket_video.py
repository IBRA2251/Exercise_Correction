import socket
import struct
import cv2
import numpy as np
import time
class SocketVideoClient:
    def __init__(self, host="127.0.0.1", port=5000):
        print(f'using port {port}')
        print(f'using host {host}')
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.data = b""

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def read_frame(self):
        # Each frame prefixed with 4-byte length
        while len(self.data) < 4:
            packet = self.sock.recv(4096)
            if not packet:
                return None
            self.data += packet

        msg_len = struct.unpack(">I", self.data[:4])[0]
        self.data = self.data[4:]

        while len(self.data) < msg_len:
            packet = self.sock.recv(4096)
            if not packet:
                return None
            self.data += packet

        frame_data = self.data[:msg_len]
        self.data = self.data[msg_len:]

        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame


import socket
import struct
import cv2
import numpy as np

class SocketVideoClient:
    def __init__(self, host="127.0.0.1", port=5000):
        print(f"Listening on {host}:{port}")
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.data = b""

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print("Waiting for Node to connect...")
        self.conn, addr = self.sock.accept()
        print(f"Connected by {addr}")

    def read_frame(self):
        # Each frame prefixed with 4-byte length
        while len(self.data) < 4:
            packet = self.conn.recv(4096)
            if not packet:
                return None
            self.data += packet

        msg_len = struct.unpack(">I", self.data[:4])[0]
        self.data = self.data[4:]

        while len(self.data) < msg_len:
            packet = self.conn.recv(4096)
            if not packet:
                return None
            self.data += packet

        frame_data = self.data[:msg_len]
        self.data = self.data[msg_len:]

        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

if __name__ == '__main__':
    tries = 0
    while True:
        try:
            server = SocketVideoServer()
            server.start()
            print('here')
            while True:
                print('in true')
                frame = server.read_frame()
                print(f'frame is: {frame}')
                if frame is None:
                    break
                cv2.imshow("Received", frame)
                if cv2.waitKey(1) == 27:
                    break
        except Exception as err:
            print(err)
            time.sleep(tries * 5)
            tries += 1
            if tries == 100:
                tries = 50
