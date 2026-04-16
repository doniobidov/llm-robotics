import requests
import json
import time
import threading

ROBOT_IP = "XXX.XXX.XXX.XXX"
ROBOT_PORT = 5000
ROBOT_URL = f"http://{ROBOT_IP}:{ROBOT_PORT}/receive_data"

class RobotInterface:
    def __init__(self):
        self.session = requests.Session()
        self.lock = threading.Lock()
        
    def send(self, key):
        payload = {"key": key}
        with self.lock:
            try:
                self.session.post(ROBOT_URL, json=payload, timeout=0.1)
            except: pass

    def stand_up(self):
        for _ in range(3):
            self.send("P")
            time.sleep(0.5)

    def stop(self):
        self.send("B")

    def emergency_stop(self):
        for _ in range(5):
            self.send("B")
            time.sleep(0.1)
