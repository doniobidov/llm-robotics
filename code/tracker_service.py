import os
import time
import json
import numpy as np
import math
import socket
import struct
import open3d as o3d
import copy
import pickle
import redis
from threading import Thread, Lock

# --- CONFIG ---
UDP_IP = "0.0.0.0"
UDP_PORT = 9000
HDR_FMT = "!I I H H I Q"
HDR_SIZE = struct.calcsize(HDR_FMT)
MAGIC = 0xBEEFBEEF

VOXEL_SIZE = 0.05        
MAP_VOXEL_SIZE = 0.08   
ICP_THRESHOLD = 0.5      
MIN_DIST = 0.9          
MAX_DIST = 30.0

r_db = redis.Redis(host='localhost', port=6379, db=0)

state = {
    "x": 0.0, "y": 0.0, "yaw": 0.0,
    "global_transform": np.eye(4),
    "frame_count": 0
}

def load_initial_state():
    global state
    try:
        saved_raw = r_db.get("robot_packet")
        if saved_raw:
            packet = pickle.loads(saved_raw)
            saved_pose = packet["pose"]
            state["x"] = saved_pose.get("x", 0.0)
            state["y"] = saved_pose.get("y", 0.0)
            state["yaw"] = saved_pose.get("yaw", 0.0)
            
            T = np.eye(4)
            T[0, 3] = state["x"]
            T[1, 3] = state["y"]
            yaw_rad = np.radians(state["yaw"])
            T[0, 0] = np.cos(yaw_rad)
            T[0, 1] = -np.sin(yaw_rad)
            T[1, 0] = np.sin(yaw_rad)
            T[1, 1] = np.cos(yaw_rad)
            state["global_transform"] = T
            print(f"[Tracker] Restored State: ({state['x']:.2f}, {state['y']:.2f}) | Yaw: {state['yaw']:.1f}°")
    except Exception as e:
        print(f"[Tracker] Load Error: {e}")

global_map = o3d.geometry.PointCloud()
processing_lock = Lock()
reassembly_buffer = {}

def preprocess_point_cloud(points_np):
    dists_sq = points_np[:, 0]**2 + points_np[:, 1]**2
    mask_safety = (points_np[:, 2] < 1.2) & (points_np[:, 2] > -0.8) & \
                  (dists_sq > MIN_DIST**2) & (dists_sq < MAX_DIST**2)
                  
    points = points_np[mask_safety]
    if len(points) < 50: return None

    CELL_SIZE = 0.2
    idx_x = np.floor(points[:, 0] / CELL_SIZE).astype(np.int64)
    idx_y = np.floor(points[:, 1] / CELL_SIZE).astype(np.int64)
    
    grid_keys = idx_x * 10000 + idx_y
    sort_idx = np.argsort(grid_keys)
    sorted_keys = grid_keys[sort_idx]
    sorted_z = points[sort_idx, 2]
    
    unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
    min_z_per_cell = np.minimum.reduceat(sorted_z, start_indices)
    
    counts = np.diff(np.append(start_indices, len(sorted_keys)))
    local_ground_z = np.repeat(min_z_per_cell, counts)
    
    is_obstacle = sorted_z > (local_ground_z + 0.10)
    cleaned_points = points[sort_idx][is_obstacle]

    if len(cleaned_points) < 10: return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cleaned_points)
    return pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

def update_slam(local_pcd):
    global global_map
    local_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    if len(global_map.points) == 0:
        global_map = copy.deepcopy(local_pcd)
        return np.asarray(local_pcd.points, dtype=np.float32)

    try:
        reg = o3d.pipelines.registration.registration_icp(
            local_pcd, global_map, ICP_THRESHOLD, state["global_transform"],
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
        )
        
        state["global_transform"] = reg.transformation
        T = state["global_transform"]
        state["x"], state["y"] = T[0, 3], T[1, 3]
        state["yaw"] = np.degrees(math.atan2(T[1, 0], T[0, 0]))

        pcd_transformed = copy.deepcopy(local_pcd).transform(T)
        
        if state["frame_count"] % 5 == 0:
            global_map += pcd_transformed
            global_map = global_map.voxel_down_sample(voxel_size=MAP_VOXEL_SIZE)

        return np.asarray(pcd_transformed.points, dtype=np.float32)
    except Exception as e:
        print(f"SLAM Error: {e}")
        return None

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[Tracker] Listening on port {UDP_PORT} (Heartbeat Mode)...")
    
    load_initial_state()
    
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            if len(data) < HDR_SIZE: continue
            header = struct.unpack(HDR_FMT, data[:HDR_SIZE])
            magic, frame_id, chunk_id, total, _, _ = header
            
            if magic != MAGIC: continue
            
            if frame_id not in reassembly_buffer:
                reassembly_buffer[frame_id] = {"total": total, "chunks": {}, "ts": time.time()}
            
            reassembly_buffer[frame_id]["chunks"][chunk_id] = data[HDR_SIZE:]
            
            if len(reassembly_buffer[frame_id]["chunks"]) == total:
                payload = b"".join([reassembly_buffer[frame_id]["chunks"][i] for i in sorted(reassembly_buffer[frame_id]["chunks"].keys())])
                pts = np.frombuffer(payload, dtype=np.float32).reshape(-1, 3).astype(np.float64)
                
                with processing_lock:
                    state["frame_count"] += 1
                    clean_pcd = preprocess_point_cloud(pts)
                    
                    # Safe initialization as (0, 3) prevents downstream vstack errors
                    final_scan = np.empty((0, 3), dtype=np.float32)
                    if clean_pcd:
                        global_pts = update_slam(clean_pcd)
                        if global_pts is not None:
                            final_scan = global_pts
                    
                    # Always publish packet (Heartbeat)
                    packet = {
                        "ts": time.time(),
                        "frame_id": state["frame_count"],
                        "pose": {"x": state["x"], "y": state["y"], "yaw": state["yaw"]},
                        "scan": final_scan 
                    }
                    r_db.setex("robot_packet", 3, pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL))
                    
                    if state["frame_count"] % 20 == 0:
                        # Added yaw output to terminal here
                        print(f"[Tracker] Frame {state['frame_count']} | Pos: ({state['x']:.2f}, {state['y']:.2f}) | Yaw: {state['yaw']:.1f}°")

                del reassembly_buffer[frame_id]
                
                now = time.time()
                keys = list(reassembly_buffer.keys())
                for k in keys:
                    if now - reassembly_buffer[k]["ts"] > 1.0: del reassembly_buffer[k]
                    
        except Exception as e:
            print(f"UDP Error: {e}")

if __name__ == "__main__":
    t = Thread(target=udp_listener, daemon=True)
    t.start()
    while True: time.sleep(1.0)
