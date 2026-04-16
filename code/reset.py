import redis
import os

def clean_slate():
    print("[System] Connecting to Redis...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # 1. Disable Persistence (Fixes the "Jitter" issue)
        print("[System] Disabling Redis disk persistence to prevent latency spikes...")
        try:
            r.config_set('save', '')  # Disable snapshotting
            r.config_set('appendonly', 'no') # Disable AOF
        except Exception as e:
            print(f"[Warning] Could not set Redis config (permissions?): {e}")

        # 2. Delete specific keys
        keys_to_delete = [
            "robot_packet", 
            "robot_state", 
            "robot_scan", 
            "robot_memory", 
            "robot_debug",
            "robot_vlm_image",
            "robot_camera_raw",
            "robot_object_memory",
            "robot_camera_debug_img"
        ]
        
        print(f"[System] Deleting {len(keys_to_delete)} Redis keys...")
        r.delete(*keys_to_delete)
        
        # 3. Delete visit history file using absolute path to guarantee deletion
        visit_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visit_grid.json")
        if os.path.exists(visit_file):
            os.remove(visit_file)
            print(f"[System] Deleted {visit_file}")
        else:
            print(f"[System] No existing {visit_file} found. Starting fresh.")

        print("[System] System Cleaned. Ready to start services.")
        
    except Exception as e:
        print(f"[Error] Redis connection failed: {e}\nIs your Redis server running?")

if __name__ == "__main__":
    clean_slate()
