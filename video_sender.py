import socket
import struct
import cv2
import time
import numpy as np

HOST = "127.0.0.1"   # trainer machine IP (use server IP if remote)
PORT = 5000

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Sender] Connecting to {HOST}:{PORT} ...")
    sock.connect((HOST, PORT))
    print("[Sender] Connected to trainer.")

    cap = cv2.VideoCapture(0)  # change to video file path if needed

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # encode to JPEG
            ret, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            data = encoded.tobytes()
            msg = struct.pack(">I", len(data)) + data
            sock.sendall(msg)
            # small sleep to avoid flooding - adjust as needed
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("[Sender] Interrupted by user")
    finally:
        cap.release()
        sock.close()
        print("[Sender] Closed")

if __name__ == "__main__":
    main()
