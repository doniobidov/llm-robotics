import numpy as np
import redis
import pickle

r_db = redis.Redis(host='localhost', port=6379, db=0)

def load_data():
    """
    Returns: (state_dict, scan_points_np, timestamp, frame_id)
    """
    try:
        raw = r_db.get("robot_packet")
        if not raw: return None, None, 0.0, -1
        
        packet = pickle.loads(raw)
        # Return frame_id (default -1 if missing)
        return packet["pose"], packet["scan"], packet["ts"], packet.get("frame_id", -1)

    except Exception as e:
        print(f"Redis Load Error: {e}")
        return None, None, 0.0, -1
