import socket
import struct
import cv2
import numpy as np
import time

class SocketVideoClient:
    """
    This class listens on host:port, accepts one connection and provides read_frame().
    (Name kept as SocketVideoClient so your trainer.py import doesn't need changes.)
    """
    def __init__(self, host="0.0.0.0", port=5000):
        print(f"[SocketVideo] Listening on {host}:{port}")
        self.host = host
        self.port = port
        self.sock = None
        self.conn = None
        self.data = b""
        self.addr = None

    def connect(self):
        """Bind and accept a single incoming connection. Blocks until a sender connects."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print("[SocketVideo] Waiting for sender to connect...")
        self.conn, self.addr = self.sock.accept()
        print(f"[SocketVideo] Connected by {self.addr}")

    def read_frame(self):
        """
        Read a single length-prefixed frame from the connected sender.
        Format: 4-byte big-endian length followed by JPEG bytes.
        Returns decoded OpenCV BGR image or None if disconnected.
        """
        # read 4-byte length
        while len(self.data) < 4:
            packet = self.conn.recv(4096)
            if not packet:
                return None
            self.data += packet

        msg_len = struct.unpack(">I", self.data[:4])[0]
        self.data = self.data[4:]

        # read frame bytes
        while len(self.data) < msg_len:
            packet = self.conn.recv(4096)
            if not packet:
                return None
            self.data += packet

        frame_data = self.data[:msg_len]
        self.data = self.data[msg_len:]

        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def close(self):
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        print("[SocketVideo] Closed")
