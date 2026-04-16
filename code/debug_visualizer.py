import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import math
import redis
import pickle
import matplotlib.patches as mpatches
import base64
import io
from PIL import Image

# --- CONFIG ---
r_db = redis.Redis(host='localhost', port=6379, db=0)

plt.ion() 
fig_map, ax_map = plt.subplots(figsize=(8, 8))
fig_map.canvas.manager.set_window_title('Map & Telemetry')

fig_log, ax_log = plt.subplots(figsize=(6, 5))
fig_log.canvas.manager.set_window_title('VLM Brain Logic & Memory')
ax_log.axis('off') 

fig_visit, ax_visit = plt.subplots(figsize=(8, 8))
fig_visit.canvas.manager.set_window_title('Visit Frequency Heatmap')

fig_vlm, ax_vlm = plt.subplots(figsize=(6, 6))
fig_vlm.canvas.manager.set_window_title('VLM Optical Feed')
fig_vlm.subplots_adjust(left=0, right=1, bottom=0, top=1) 
ax_vlm.set_axis_off()

fig_cam, ax_cam = plt.subplots(figsize=(6, 4))
fig_cam.canvas.manager.set_window_title('Live Camera Feed')
fig_cam.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax_cam.set_axis_off()

KEY_MEANINGS = {
    "W": "FORWARD", "S": "REVERSE/STUCK", "A": "ROTATE LEFT", "D": "ROTATE RIGHT",
    "Q": "CURVE LEFT", "E": "CURVE RIGHT", "B": "STOP (BRAKE)", "P": "STAND UP", "IDLE": "IDLE"
}

def load_data():
    try:
        raw_packet = r_db.get("robot_packet")
        if not raw_packet: return None, [], [], {}, None, None
        packet = pickle.loads(raw_packet)
        state = packet["pose"]
        points = packet["scan"]
        
        mem_raw = r_db.get("robot_memory")
        memory_points = pickle.loads(mem_raw) if mem_raw else []

        debug_raw = r_db.get("robot_debug")
        debug_data = json.loads(debug_raw) if debug_raw else {}
        
        b64_img = r_db.get("robot_vlm_image")
        vlm_image = b64_img.decode('utf-8') if b64_img else None
        
        cam_raw = r_db.get("robot_camera_debug_img")
        cam_image = cam_raw.decode('utf-8') if cam_raw else None
        
        return state, points, memory_points, debug_data, vlm_image, cam_image
    except: return None, [], [], {}, None, None

def update_map_view(state, points, memory_points, debug_data):
    ax_map.clear()
    ax_map.set_facecolor('black')
    WINDOW_SIZE = 8.0 
    ax_map.set_xlim(-WINDOW_SIZE, WINDOW_SIZE)
    ax_map.set_ylim(-WINDOW_SIZE, WINDOW_SIZE)
    ax_map.grid(True, color='#333333', linestyle='--')
    ax_map.set_aspect('equal')

    if state is None: return

    rx, ry = state["x"], state["y"]
    yaw_rad = np.radians(state["yaw"])

    if len(memory_points) > 0:
        ax_map.scatter(memory_points[:, 0] - rx, memory_points[:, 1] - ry, s=2, c='orange', alpha=0.3)
    if len(points) > 0:
        ax_map.scatter(points[:, 0] - rx, points[:, 1] - ry, s=1, c='white', alpha=0.8)

    ax_map.scatter(0, 0, c='cyan', s=150, zorder=5)
    ax_map.arrow(0, 0, 0.8*np.cos(yaw_rad), 0.8*np.sin(yaw_rad), head_width=0.3, fc='cyan', ec='cyan', zorder=5)

    path = debug_data.get("path", [])
    last_px, last_py = 0, 0
    if path:
        px = [p[0] - rx for p in path]
        py = [p[1] - ry for p in path]
        ax_map.plot(px, py, c='red', linewidth=2)
        ax_map.scatter(px[-1], py[-1], c='lime', s=120, marker='*')
        last_px, last_py = px[-1], py[-1]

    patches = [
        mpatches.Patch(color='white', label='Live Scan'),
        mpatches.Patch(color='orange', label='Memory (History)'),
        mpatches.Patch(color='red', label='Current Path')
    ]
    ax_map.legend(handles=patches, loc='lower right', fontsize=8, facecolor='#222222', labelcolor='white')
    
    status = debug_data.get('status', 'INIT')
    raw_cmd = debug_data.get('last_cmd', 'IDLE')
    cmd_human = KEY_MEANINGS.get(raw_cmd, "UNKNOWN")
    
    telemetry_text = (
        f"STATUS:  {status}\n"
        f"POS:     ({rx:.2f}, {ry:.2f})\n"
        f"CMD:     {raw_cmd} ({cmd_human})" 
    )
    
    props = dict(boxstyle='round', facecolor='#222222', alpha=0.9, edgecolor='cyan')
    ax_map.text(-WINDOW_SIZE*0.95, WINDOW_SIZE*0.95, telemetry_text, 
                fontsize=11, fontname='monospace', color='cyan', 
                verticalalignment='top', bbox=props)

def update_log_view(debug_data):
    ax_log.clear()
    ax_log.set_facecolor('#1e1e1e')
    
    objective = debug_data.get("llm_input", "Objective: (unknown)")
    raw_out = debug_data.get("llm_raw_output", "No Output Yet")
    semantic = debug_data.get("semantic_text", "No objects detected recently.")

    text = (
        f"USER OBJECTIVE:\n{objective}\n\n"
        f"SEMANTIC MEMORY FED TO LLM:\n{semantic}\n\n"
        f"LLM OUTPUT (RAW):\n{raw_out}\n"
    )

    ax_log.text(
        0.02, 0.98, text,
        fontsize=10, fontname='monospace',
        verticalalignment='top', color='lime', wrap=True
    )

def update_visit_view(state, debug_data):
    ax_visit.clear()
    ax_visit.set_facecolor('black')
    if state is None: return
    rx, ry = state["x"], state["y"]
    ax_visit.scatter(rx, ry, c='cyan', s=100, marker='X')
    
    heatmap_data = debug_data.get("visit_heatmap", [])
    if heatmap_data:
        data = np.array(heatmap_data)
        if len(data) > 0:
            ax_visit.scatter(data[:,0], data[:,1], c=data[:,2], cmap='jet', s=150, marker='s')
            if len(data) < 200:
                for x, y, c in zip(data[:,0], data[:,1], data[:,2]):
                    if c > 0: ax_visit.text(x, y, str(int(c)), color='white', fontsize=6, ha='center', va='center')

    WINDOW_SIZE = 10.0
    ax_visit.set_xlim(rx - WINDOW_SIZE, rx + WINDOW_SIZE)
    ax_visit.set_ylim(ry - WINDOW_SIZE, ry + WINDOW_SIZE)

def update_vlm_feed_view(vlm_image, debug_data):
    if not vlm_image: return
    try:
        image_data = base64.b64decode(vlm_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        width, height = image.size

        ax_vlm.clear()
        ax_vlm.imshow(image, interpolation="nearest", origin="upper")
        ax_vlm.set_xlim(-0.5, width - 0.5)
        ax_vlm.set_ylim(height - 0.5, -0.5)
        ax_vlm.set_aspect("equal")
        ax_vlm.set_axis_off()
        ax_vlm.set_autoscale_on(False)

        u, v = debug_data.get("vlm_uv", [0.5, 0.5])
        u = max(0.0, min(1.0, float(u)))
        v = max(0.0, min(1.0, float(v)))

        px = u * (width - 1)
        py = v * (height - 1)

        ax_vlm.scatter(px, py, c="yellow", marker="*", s=220, edgecolors="black", linewidths=1.0)
        ax_vlm.text(5, 15, "VLM Input", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.4, pad=2))
    except Exception: return

def update_cam_feed_view(cam_image):
    if not cam_image: return
    try:
        image_data = base64.b64decode(cam_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        ax_cam.clear()
        ax_cam.imshow(image)
        ax_cam.set_axis_off()
        ax_cam.text(5, 15, "YOLO Vision", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.4, pad=2))
    except Exception: return

while True:
    try:
        state, points, mem, dbg, vlm_img, cam_img = load_data()
        if state:
            update_map_view(state, points, mem, dbg)
            update_log_view(dbg)
            update_visit_view(state, dbg)
            update_vlm_feed_view(vlm_img, dbg)
            update_cam_feed_view(cam_img)
            
            fig_map.canvas.draw(); fig_map.canvas.flush_events()
            fig_log.canvas.draw(); fig_log.canvas.flush_events()
            fig_visit.canvas.draw(); fig_visit.canvas.flush_events()
            fig_vlm.canvas.draw(); fig_vlm.canvas.flush_events()
            fig_cam.canvas.draw(); fig_cam.canvas.flush_events()
        time.sleep(0.1)
    except Exception as e: print(e); time.sleep(1.0)
