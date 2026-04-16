import numpy as np
import math
import heapq
import os
import json
import time
from collections import deque
from scipy.ndimage import binary_dilation, generate_binary_structure, distance_transform_edt

# --- CONFIG ---
GRID_RES = 0.05    
MAP_SIZE = 22.0  
GRID_DIM = int(MAP_SIZE / GRID_RES)
CENTER_IDX = GRID_DIM // 2

# --- ROBOT CLEARANCE ---
ROBOT_WIDTH = 0.28
SIDE_CLEARANCE = 0.05
INFLATION_M = (ROBOT_WIDTH / 2.0) + SIDE_CLEARANCE
WALL_THICKNESS_ITERATIONS = int(np.ceil(INFLATION_M / GRID_RES))

# --- VISIT CONFIG ---
VISIT_CELL_SIZE = 0.5 
VISIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visit_grid.json")

class NavigationCore:
    def __init__(self):
        self.grid = np.zeros((GRID_DIM, GRID_DIM), dtype=np.int8)
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        # --- PROXIMITY MEMORY BUFFER ---
        self.local_memory_points = np.empty((0, 3), dtype=np.float32)
        self.HISTORY_RADIUS_ADD = 0   
        self.HISTORY_RADIUS_KEEP = 0 
        self.MEMORY_GRID_RES = 0.05     
        self.memory_update_counter = 0  
        
        self.visit_cell_size = VISIT_CELL_SIZE
        self.visit_counts = {} 
        self.last_visit_pos = None
        self.load_visits()

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin_x) / GRID_RES) + CENTER_IDX
        gy = int((wy - self.origin_y) / GRID_RES) + CENTER_IDX
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = (gx - CENTER_IDX) * GRID_RES + self.origin_x
        wy = (gy - CENTER_IDX) * GRID_RES + self.origin_y
        return wx, wy

    # --- VISIT LOGIC ---
    def load_visits(self):
        if os.path.exists(VISIT_FILE):
            try:
                with open(VISIT_FILE, 'r') as f:
                    raw = json.load(f)
                    self.visit_counts = {tuple(map(int, k.split(','))): v for k, v in raw.items()}
            except: pass

    def save_visits(self):
        try:
            export = {f"{k[0]},{k[1]}": v for k, v in self.visit_counts.items()}
            with open(VISIT_FILE, 'w') as f: json.dump(export, f)
        except: pass

    def update_visit_status(self, wx, wy):
        vx = int(np.floor(wx / VISIT_CELL_SIZE))
        vy = int(np.floor(wy / VISIT_CELL_SIZE))
        current_pos = (vx, vy)
        if current_pos != self.last_visit_pos:
            self.visit_counts[current_pos] = self.visit_counts.get(current_pos, 0) + 1
            self.last_visit_pos = current_pos

    def get_visit_count_at_coord(self, wx, wy):
        vx = int(np.floor(wx / VISIT_CELL_SIZE))
        vy = int(np.floor(wy / VISIT_CELL_SIZE))
        return self.visit_counts.get((vx, vy), 0)

    # --- MAPPING LOGIC ---
    def update_map(self, rx, ry, ryaw, current_global_points):
        self.origin_x, self.origin_y = rx, ry
        
        current_visible = np.empty((0, 3), dtype=np.float32)
        if len(current_global_points) > 0:
            dx = current_global_points[:, 0] - rx
            dy = current_global_points[:, 1] - ry
            mask_map = (np.abs(dx) < MAP_SIZE/2.0) & (np.abs(dy) < MAP_SIZE/2.0)
            current_visible = current_global_points[mask_map]

        if len(current_visible) > 0:
            dx_vis = current_visible[:, 0] - rx
            dy_vis = current_visible[:, 1] - ry
            mask_add = (dx_vis**2 + dy_vis**2) < self.HISTORY_RADIUS_ADD**2
            pts_to_add = current_visible[mask_add]
            self.local_memory_points = np.vstack((self.local_memory_points, pts_to_add))

        history_points_only = np.empty((0, 3), dtype=np.float32)
        if len(self.local_memory_points) > 0:
            dx_mem = self.local_memory_points[:, 0] - rx
            dy_mem = self.local_memory_points[:, 1] - ry
            mask_keep = (dx_mem**2 + dy_mem**2) < self.HISTORY_RADIUS_KEEP**2
            self.local_memory_points = self.local_memory_points[mask_keep]
            
            self.memory_update_counter += 1
            if self.memory_update_counter % 5 == 0 and len(self.local_memory_points) > 0:
                quantized_xy = np.floor(self.local_memory_points[:, :2] / self.MEMORY_GRID_RES).astype(np.int32)
                _, unique_indices = np.unique(quantized_xy, axis=0, return_index=True)
                self.local_memory_points = self.local_memory_points[unique_indices]
                
            history_points_only = self.local_memory_points

        points_list = []
        if len(current_visible) > 0: points_list.append(current_visible)
        if len(history_points_only) > 0: points_list.append(history_points_only)
        
        self.grid.fill(0)
        
        if points_list:
            all_points = np.vstack(points_list)
            dx = all_points[:, 0] - rx
            dy = all_points[:, 1] - ry
            
            mask_map = (np.abs(dx) < MAP_SIZE/2.0) & (np.abs(dy) < MAP_SIZE/2.0)
            valid_dx = dx[mask_map]
            valid_dy = dy[mask_map]
            
            gx_obs = (valid_dx / GRID_RES).astype(int) + CENTER_IDX
            gy_obs = (valid_dy / GRID_RES).astype(int) + CENTER_IDX
            
            valid = (gx_obs >= 0) & (gx_obs < GRID_DIM) & (gy_obs >= 0) & (gy_obs < GRID_DIM)
            self.grid[gx_obs[valid], gy_obs[valid]] = 1 

            struct = generate_binary_structure(2, 2)
            dilated = binary_dilation(self.grid == 1, structure=struct, iterations=WALL_THICKNESS_ITERATIONS)
            self.grid[dilated] = 1
        
        final_points = np.vstack(points_list) if points_list else np.empty((0, 3), dtype=np.float32)
        return final_points, history_points_only

    def cast_rays(self, robot_yaw):
        options = []
        robot_gx, robot_gy = self.world_to_grid(self.origin_x, self.origin_y)
        
        for rel_angle in range(-180, 180, 10): 
            abs_rad = np.radians(robot_yaw + rel_angle)
            dx = np.cos(abs_rad)
            dy = np.sin(abs_rad)
            curr_x, curr_y = float(robot_gx), float(robot_gy)
            
            found_wall = False
            for i in range(int(8.0/GRID_RES)):
                curr_x += dx
                curr_y += dy
                ix, iy = int(curr_x), int(curr_y)
                if not (0 <= ix < GRID_DIM and 0 <= iy < GRID_DIM): break
                if self.grid[ix, iy] == 1: 
                    found_wall = True
                    break
            
            hit_dist_m = np.hypot((ix-robot_gx)*GRID_RES, (iy-robot_gy)*GRID_RES)
            safe_dist = hit_dist_m - 1.2 if found_wall else 7.0
            
            if safe_dist > 1.0: 
                target_wx = self.origin_x + (np.cos(abs_rad) * safe_dist)
                target_wy = self.origin_y + (np.sin(abs_rad) * safe_dist)
                visits = self.get_visit_count_at_coord(target_wx, target_wy)
                
                lbl = "AHEAD"
                if abs(rel_angle) > 120: lbl = "BEHIND"
                elif rel_angle > 45: lbl = "LEFT"
                elif rel_angle < -45: lbl = "RIGHT"

                options.append({
                    "angle": rel_angle,
                    "label": lbl,
                    "dist": safe_dist,
                    "visits": visits,
                    "coords": [target_wx, target_wy]
                })
        return options

    # --- REACHABILITY & CLEARANCE ---
    def get_best_clearance_target(self, start_wx, start_wy, target_wx, target_wy, grid_snap, origin_x, origin_y, max_radius_m=0.5):
        """BFS flood-fill for reachability + Distance Transform for clearance."""
        sgx = int((start_wx - origin_x) / GRID_RES) + CENTER_IDX
        sgy = int((start_wy - origin_y) / GRID_RES) + CENTER_IDX
        tgx = int((target_wx - origin_x) / GRID_RES) + CENTER_IDX
        tgy = int((target_wy - origin_y) / GRID_RES) + CENTER_IDX

        if not (0 <= sgx < GRID_DIM and 0 <= sgy < GRID_DIM): return None
        # if grid_snap[sgx, sgy] == 1: return None

        free_space_mask = (grid_snap == 0)
        clearance_map_m = distance_transform_edt(free_space_mask) * GRID_RES

        reachable_mask = np.zeros_like(grid_snap, dtype=bool)
        reachable_mask[sgx, sgy] = True
        
        queue = deque([(sgx, sgy)])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                    if grid_snap[nx, ny] == 0 and not reachable_mask[nx, ny]:
                        if abs(dx) + abs(dy) == 2:
                            if grid_snap[cx+dx, cy] == 1 or grid_snap[cx, cy+dy] == 1:
                                continue
                        reachable_mask[nx, ny] = True
                        queue.append((nx, ny))

        radius_px = int(max_radius_m / GRID_RES)
        best_node = None
        max_clearance_m = -1.0
        min_dist_to_target_m = float('inf')

        for dx in range(-radius_px, radius_px + 1):
            for dy in range(-radius_px, radius_px + 1):
                if dx*dx + dy*dy <= radius_px*radius_px:
                    nx, ny = tgx + dx, tgy + dy
                    if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                        if reachable_mask[nx, ny]:
                            clearance = clearance_map_m[nx, ny]
                            dist_to_target = math.hypot(dx, dy) * GRID_RES
                            
                            if clearance > max_clearance_m + 0.05: 
                                max_clearance_m = clearance
                                min_dist_to_target_m = dist_to_target
                                best_node = (nx, ny)
                            elif abs(clearance - max_clearance_m) <= 0.05:
                                if dist_to_target < min_dist_to_target_m:
                                    min_dist_to_target_m = dist_to_target
                                    best_node = (nx, ny)

        if best_node is not None:
            wx = (best_node[0] - CENTER_IDX) * GRID_RES + origin_x
            wy = (best_node[1] - CENTER_IDX) * GRID_RES + origin_y
            return [wx, wy]
            
        return None

    # --- EXPLORATION CANDIDATES FOR VLM ---
    def get_exploration_candidates(self, grid_snap, origin_snap, robot_x, robot_y, view_size_m, bin_size_m=1.5):
        origin_x, origin_y = origin_snap
        sgx = int((robot_x - origin_x) / GRID_RES) + CENTER_IDX
        sgy = int((robot_y - origin_y) / GRID_RES) + CENTER_IDX
        
        if not (0 <= sgx < GRID_DIM and 0 <= sgy < GRID_DIM) or grid_snap[sgx, sgy] == 1:
            return []
            
        free_mask = (grid_snap == 0)
        clearance_m = distance_transform_edt(free_mask) * GRID_RES
        
        reachable = np.zeros_like(grid_snap, dtype=bool)
        reachable[sgx, sgy] = True
        queue = deque([(sgx, sgy)])
        
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_DIM and 0 <= ny < GRID_DIM:
                    if grid_snap[nx, ny] == 0 and not reachable[nx, ny]:
                        if abs(dx) + abs(dy) == 2: # Prevent diagonal wall clipping
                            if grid_snap[cx+dx, cy] == 1 or grid_snap[cx, cy+dy] == 1:
                                continue
                        reachable[nx, ny] = True
                        queue.append((nx, ny))
        
        candidates_dict = {}
        for x in range(GRID_DIM):
            for y in range(GRID_DIM):
                # Must be reachable and at least 30cm from walls
                if reachable[x, y] and clearance_m[x, y] >= 0.3: 
                    wx = (x - CENTER_IDX) * GRID_RES + origin_x
                    wy = (y - CENTER_IDX) * GRID_RES + origin_y
                    
                    dx = wx - robot_x
                    dy = wy - robot_y
                    if abs(dx) > view_size_m/2 or abs(dy) > view_size_m/2:
                        continue
                        
                    bin_x = int(math.floor(dx / bin_size_m))
                    bin_y = int(math.floor(dy / bin_size_m))
                    bin_key = (bin_x, bin_y)
                    
                    if bin_key not in candidates_dict or clearance_m[x, y] > candidates_dict[bin_key][2]:
                        candidates_dict[bin_key] = (wx, wy, clearance_m[x, y])
        
        uv_list = []
        for (wx, wy, c) in candidates_dict.values():
            u = 0.5 + ((wx - robot_x) / view_size_m)
            v = 0.5 - ((wy - robot_y) / view_size_m)
            if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                uv_list.append([round(u, 2), round(v, 2)])
                
        return uv_list

    def a_star(self, start, end):
        sgx, sgy = self.world_to_grid(*start)
        egx, egy = self.world_to_grid(*end)
        
        if not (0 <= sgx < GRID_DIM and 0 <= sgy < GRID_DIM): return None
        if not (0 <= egx < GRID_DIM and 0 <= egy < GRID_DIM): return None

        # if self.grid[sgx, sgy] == 1: return None 
        if self.grid[egx, egy] == 1: return None

        open_set = []
        heapq.heappush(open_set, (0, (sgx, sgy)))
        came_from = {}
        g_score = {(sgx, sgy): 0}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == (egx, egy):
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                return path[::-1]

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nx, ny = current[0]+dx, current[1]+dy
                if not (0<=nx<GRID_DIM and 0<=ny<GRID_DIM): continue
                if self.grid[nx, ny] == 1: continue
                
                if abs(dx) + abs(dy) == 2:
                    if self.grid[current[0]+dx, current[1]] == 1 or \
                       self.grid[current[0], current[1]+dy] == 1:
                        continue
                
                cost = 1.414 if abs(dx)+abs(dy)==2 else 1.0
                tentative_g = g_score.get(current, float('inf')) + cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + np.hypot(nx-egx, ny-egy)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current
        return None
