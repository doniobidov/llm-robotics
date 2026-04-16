import time
import math
import json
import os
import numpy as np
import redis
import pickle
import threading
import cv2
from concurrent.futures import ThreadPoolExecutor
import planner_service 
from navigation_core import NavigationCore
from vlm_strategist import VLMStrategist 
from vlm_map_renderer import generate_snapshot_image 
from robot_interface import RobotInterface

# --- CONFIG ---
SEARCH_OBJECTIVE = "door"
GOAL_TOLERANCE = 0.5
MAX_CROSS_TRACK_ERROR = 0.4
WATCHDOG_TIMEOUT = 3.0 

r_db = redis.Redis(host='localhost', port=6379, db=0)

nav = NavigationCore()
brain = VLMStrategist()
robot = RobotInterface()
executor = ThreadPoolExecutor(max_workers=1)

current_path = [] 
current_goal = None
state_status = "IDLE" 
reasoning_text = "Init"
last_ctx = ""
last_raw_llm = "" 
last_semantic = "No objects"
path_start_time = 0
last_uv = [0.5, 0.5]

planning_future = None
planning_id = 0 
last_frame_id = -1
last_data_time = time.time()

def apply_room_bounds(nav_core, inflation_m=0.05):
    """
    Fits a bounding rectangle around the current global map walls,
    and masks all unknown space outside this rectangle as occupied (1).
    This safely restricts VLM targets, A* routing, and clearance checks.
    """
    occupied_indices = np.argwhere(nav_core.grid == 1)
    
    if len(occupied_indices) >= 50:
        pts_cv = np.column_stack((occupied_indices[:, 1], occupied_indices[:, 0])).astype(np.float32)
        
        rect = cv2.minAreaRect(pts_cv)
        (cx, cy), (w, h), angle = rect
        
        GRID_RES = 0.05
        inflation_cells = inflation_m / GRID_RES
        
        new_w = max(1.0, w + inflation_cells * 2)
        new_h = max(1.0, h + inflation_cells * 2)
        
        inflated_rect = ((cx, cy), (new_w, new_h), angle)
        
        box = cv2.boxPoints(inflated_rect).astype(np.int32)
        
        mask = np.zeros_like(nav_core.grid, dtype=np.uint8)
        cv2.fillPoly(mask, [box], 1)
        
        # Set everything outside the bounding box to occupied (1)
        nav_core.grid[mask == 0] = 1

    # Guarantee the robot's current cell is ALWAYS free space (0), 
    # even if the map is sparse and we skipped the bounding box.
    gx, gy = nav_core.world_to_grid(nav_core.origin_x, nav_core.origin_y)
    
    if 0 <= gx < nav_core.grid.shape[0] and 0 <= gy < nav_core.grid.shape[1]:
        nav_core.grid[gx, gy] = 0

def get_pure_pursuit(pos, yaw, path):
    if not path: return "B"
    target = path[-1]
    for p in path:
        if math.hypot(p[0]-pos[0], p[1]-pos[1]) > 0.6:
            target = p; break
    angle = math.degrees(math.atan2(target[1]-pos[1], target[0]-pos[0]))
    diff = (angle - yaw + 180) % 360 - 180
    if abs(diff) > 20: return "Q" if diff > 0 else "E"
    return "W"

def check_path_validity(pos, path, nav_core):
    if len(path) > 2:
        pts = np.array(path)
        dists = np.hypot(pts[:,0]-pos[0], pts[:,1]-pos[1])
        if np.min(dists) > MAX_CROSS_TRACK_ERROR:
            return False, "Drifted too far"
    for p in path[::2]: 
        gx, gy = nav_core.world_to_grid(*p)
        if 0 <= gx < nav_core.grid.shape[0] and 0 <= gy < nav_core.grid.shape[1]:
            if nav_core.grid[gx, gy] == 1:
                return False, "Path Blocked"
    return True, "OK"

def get_semantic_memory_text(pos, yaw, size_m):
    raw_mem = r_db.get("robot_object_memory")
    if not raw_mem: return "No objects detected recently."
    
    try:
        mem_data = json.loads(raw_mem)
        
        if time.time() - mem_data.get("ts", 0) > 45.0:
            return "No objects detected recently (data stale)."
            
        objects = mem_data.get("objects", [])
        if not objects: return "No objects detected recently."
        
        yaw_rad = math.radians(yaw)
        c, s = math.cos(-yaw_rad), math.sin(-yaw_rad)
        
        text_lines = []
        for obj in objects:
            dx_g = obj["x"] - pos[0]
            dy_g = obj["y"] - pos[1]
            
            u_val = 0.5 + (dx_g / size_m)
            v_val = 0.5 - (dy_g / size_m)
            
            uv_text = ""
            if 0.0 <= u_val <= 1.0 and 0.0 <= v_val <= 1.0:
                uv_text = f"Map UV: [u: {u_val:.3f}, v: {v_val:.3f}]"
            else:
                uv_text = f"(Currently Off-Map)"
            
            dx_local = dx_g * c - dy_g * s
            dy_local = dx_g * s + dy_g * c
            
            dist = math.hypot(dx_local, dy_local)
            if dist < 10.0:
                text_lines.append(f"- {obj['label']}: {uv_text}")
                
        if not text_lines: return "No objects nearby."
        return "\n".join(text_lines)
    except:
        return "Error reading memory."

def async_plan_callback(pos, yaw, grid_snap, origin_snap, visit_snap, plan_id_snapshot, nav_ref, objective):
    VIEW_SIZE_M = 16.0
    candidates = nav_ref.get_exploration_candidates(grid_snap, origin_snap, pos[0], pos[1], VIEW_SIZE_M, bin_size_m=1.5)
    candidates_txt = json.dumps(candidates) if candidates else "[]"
    
    b64_img, meta = generate_snapshot_image(grid_snap, origin_snap[0], origin_snap[1], pos[0], pos[1], yaw, visit_snap, candidates)
    r_db.setex("robot_vlm_image", 30, b64_img)
    
    semantic_text = get_semantic_memory_text(pos, yaw, meta["view_size_m"])
    raw_target, uv, reason, _, ctx, raw_resp = brain.select_goal(b64_img, meta, objective, semantic_text, candidates_txt)
    
    validated_target = nav_ref.get_best_clearance_target(
        pos[0], pos[1], raw_target[0], raw_target[1], 
        grid_snap, origin_snap[0], origin_snap[1], max_radius_m=0.5
    )
    
    if validated_target is None:
        reason += " | REJECTED: Target isolated, unreachable, or blocked."
        validated_target = pos 
    else:
        shift_dist = math.hypot(validated_target[0] - raw_target[0], validated_target[1] - raw_target[1])
        if shift_dist > 0.05:
            reason += f" | ADJUSTED: Shifted {shift_dist:.2f}m to safest clearance."
        
    return (validated_target, reason, uv, ctx, raw_resp, semantic_text), plan_id_snapshot


def main():
    global current_path, current_goal, state_status
    global reasoning_text, last_ctx, last_raw_llm, path_start_time
    global planning_future, planning_id, last_uv, last_semantic
    global last_frame_id, last_data_time
    
    print("--- ROBUST AUTONOMY START (STOP & THINK MODE) ---")
    
    try:
        while True:
            start_loop = time.time()
            state_dict, scan_points, ts, frame_id = planner_service.load_data()
            
            if frame_id != last_frame_id and frame_id != -1:
                last_frame_id = frame_id
                last_data_time = time.time()
            
            time_since_last_new_frame = time.time() - last_data_time
            
            if state_dict is None or time_since_last_new_frame > WATCHDOG_TIMEOUT:
                if state_status != "IDLE":
                    print(f"[Watchdog] No new frames for {time_since_last_new_frame:.2f}s. STOPPING.")
                    robot.stop()
                    state_status = "IDLE"
                    current_path = []
                    current_goal = None
                    planning_id += 1 
                time.sleep(0.1)
                continue
            
            pos = [state_dict["x"], state_dict["y"]]
            yaw = state_dict["yaw"]
            
            _, history_points = nav.update_map(pos[0], pos[1], yaw, scan_points)
            
            apply_room_bounds(nav, inflation_m=0.05)
            
            nav.update_visit_status(pos[0], pos[1])
            cmd_sent = "IDLE"

            if planning_future and planning_future.done():
                try:
                    res, plan_id_res = planning_future.result()
                    target, reason, uv, ctx, raw_resp, generated_semantic_text = res
                    
                    if plan_id_res == planning_id:
                        reasoning_text = reason
                        last_ctx = ctx
                        last_raw_llm = raw_resp
                        last_uv = uv
                        last_semantic = generated_semantic_text 
                        
                        if state_status == "IDLE":
                            path = nav.a_star(pos, target)
                            if path:
                                current_path = path
                                current_goal = target
                                state_status = "MOVING"
                                path_start_time = time.time()
                                print(f"-> NEW GOAL: {reason}")
                            else:
                                print("-> VLM TARGET UNREACHABLE. Retrying...")
                except Exception as e: print(f"Planning Thread Error: {e}")
                planning_future = None

            if state_status == "IDLE":
                robot.stop()
                cmd_sent = "B"
                if planning_future is None:
                    grid_snap = nav.grid.copy()
                    origin_snap = (nav.origin_x, nav.origin_y)
                    visit_snap = nav.visit_counts.copy()
                    planning_id += 1 
                    planning_future = executor.submit(async_plan_callback, pos, yaw, grid_snap, origin_snap, visit_snap, planning_id, nav, SEARCH_OBJECTIVE)
            
            elif state_status == "MOVING":
                dist = math.hypot(current_goal[0]-pos[0], current_goal[1]-pos[1])
                
                if dist < GOAL_TOLERANCE:
                    print("-> ARRIVED! Stopping to think...")
                    state_status = "IDLE"
                    current_goal = None
                    current_path = []
                    continue
                
                if time.time() - path_start_time > 40.0:
                    print("-> TIMEOUT. Stopping to rethink...")
                    state_status = "IDLE"
                    current_goal = None
                    current_path = []
                    continue
                
                is_valid, err = check_path_validity(pos, current_path, nav)
                if not is_valid:
                    print(f"-> REPLANNING: {err}")
                    new_path = nav.a_star(pos, current_goal)
                    if new_path:
                        current_path = new_path
                    else:
                        print("-> PATH COMPLETELY BLOCKED. Stopping to rethink...")
                        state_status = "IDLE"
                        current_goal = None
                        current_path = []
                        continue

                current_path = [p for p in current_path if math.hypot(p[0]-pos[0], p[1]-pos[1]) > 0.5]
                if not current_path: current_path = [current_goal]
                
                cmd = get_pure_pursuit(pos, yaw, current_path)
                robot.send(cmd)
                cmd_sent = cmd

            visit_export = [[k[0]*nav.visit_cell_size, k[1]*nav.visit_cell_size, v] for k, v in nav.visit_counts.items()]
            debug = {
                "status": state_status, "path": current_path, "reasoning": reasoning_text,
                "llm_input": last_ctx, "llm_raw_output": last_raw_llm, 
                "semantic_text": last_semantic, "pos": pos, "yaw": yaw, "last_cmd": cmd_sent,
                "visit_heatmap": visit_export, "vlm_uv": last_uv
            }
            
            pipe = r_db.pipeline()
            pipe.setex("robot_debug", 3, json.dumps(debug))
            pipe.setex("robot_memory", 3, pickle.dumps(history_points, protocol=pickle.HIGHEST_PROTOCOL))
            pipe.execute()
            
            elapsed = time.time() - start_loop
            if elapsed < 0.1: time.sleep(0.1 - elapsed)
            
    except KeyboardInterrupt:
        robot.emergency_stop()
        nav.save_visits()

if __name__ == "__main__": main()
