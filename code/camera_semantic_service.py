import time
import math
import json
import pickle
import redis
from collections import deque

# --- CONFIG ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

POSE_HISTORY_SEC = 5.0

# Camera extrinsics relative to robot center (approx; tune this)
# Robot frame convention: +x forward, +y left
CAM_EXTR = {
    "x_forward_m": 0.10,   # camera forward offset from robot reference
    "y_left_m": 0.00,      # camera lateral offset (left positive)
    "yaw_deg": 0.0,        # camera yaw offset relative to robot forward
}

# D435i color stream @ 640x480: use a more realistic HFOV than 86 (86 is too wide)
CAM_HFOV_DEG = 69.0
CAM_MIN_RANGE_M = 0.25
CAM_MAX_RANGE_M = 6.0

# Expand overwrite wedge slightly so old edge objects don't linger due to small pose jitter
OVERWRITE_HFOV_MARGIN_DEG = 8.0
OVERWRITE_RANGE_MARGIN_M = 0.5

# Trusted distance band for deletion to prevent amnesia when too close or too far
OVERWRITE_DELETE_MIN_RANGE_M = 1.2
OVERWRITE_DELETE_MAX_RANGE_M = 4.5

# Radius to merge identical objects to prevent memory bloat outside the delete band
MERGE_RADIUS_M = 0.5

PRINT_EVERY_NEW_CAMERA_FRAME = True

# Safer at first if laptop/server clocks are not tightly synced:
# use server receive timestamp (same machine as tracker + redis)
USE_SERVER_RX_TIME = True
# --------------

r_db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def normalize_label(label: str) -> str:
    """
    Normalize detector labels into a simpler semantic map label set.
    """
    s = str(label).upper().strip()

    # Door-related labels -> DOOR
    if s in ("DOORWAY", "DOOR FRAME", "DOORFRAME"):
        return "DOOR"

    # Monitor/PC variants -> PC
    if s in ("MONITOR", "COMPUTER MONITOR", "PC", "DESKTOP COMPUTER"):
        return "PC"

    if s == "HUMAN":
        return "PERSON"
        
    # Bag variants -> BACKPACK
    if s in ("BACKPACK", "HANDBAG", "PURSE"):
        return "BACKPACK"

    return s


class PoseHistory:
    def __init__(self, horizon_s=5.0):
        self.horizon_s = horizon_s
        self.buf = deque()
        self.last_robot_frame_id = -1

    def update_from_robot_packet(self):
        raw = r_db.get("robot_packet")
        if not raw:
            return

        pkt = pickle.loads(raw)
        frame_id = pkt.get("frame_id", -1)
        if frame_id == self.last_robot_frame_id:
            return
        self.last_robot_frame_id = frame_id

        pose = pkt["pose"]
        self.buf.append({
            "ts": float(pkt["ts"]),
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": float(pose["yaw"]),
            "frame_id": int(frame_id),
        })

        cutoff = time.time() - self.horizon_s
        while self.buf and self.buf[0]["ts"] < cutoff:
            self.buf.popleft()

    def nearest(self, ts_sec):
        if not self.buf:
            return None
        return min(self.buf, key=lambda p: abs(p["ts"] - ts_sec))


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def get_camera_world_pose(pose, cam_extr):
    """
    Camera 2D pose in world coords, using robot pose + camera extrinsics.
    Returns (cx, cy, cyaw_deg).
    """
    rx, ry = pose["x"], pose["y"]
    ryaw_deg = pose["yaw"]

    yaw = math.radians(ryaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)

    # robot frame: +x forward, +y left
    x_off = cam_extr.get("x_forward_m", 0.0)
    y_off = cam_extr.get("y_left_m", 0.0)

    cx = rx + c * x_off - s * y_off
    cy = ry + s * x_off + c * y_off
    cyaw_deg = ryaw_deg + cam_extr.get("yaw_deg", 0.0)
    return cx, cy, cyaw_deg


def in_fov_wedge(gx, gy, pose, cam_extr, hfov_deg=69.0, r_min=0.25, r_max=6.0):
    """
    Efficient approximation of the current camera visible region in world space.
    """
    cam_x, cam_y, cam_yaw_deg = get_camera_world_pose(pose, cam_extr)

    dx = gx - cam_x
    dy = gy - cam_y
    r = math.hypot(dx, dy)
    if r < r_min or r > r_max:
        return False

    bearing_deg = math.degrees(math.atan2(dy, dx))
    diff = angle_diff_deg(bearing_deg, cam_yaw_deg)
    return abs(diff) <= (hfov_deg / 2.0)


def det_to_global_xy(camera_xyz_m, pose, cam_extr):
    """
    camera_xyz_m is OpenCV camera convention from server:
      x_cam = right, y_cam = down, z_cam = forward

    We convert to robot 2D frame (+x forward, +y left), then to global XY.
    """
    x_cam, y_cam, z_cam = map(float, camera_xyz_m)

    if z_cam < 0.2 or z_cam > 8.0:
        return None

    # Camera -> robot frame (2D approximation)
    # x_r: forward, y_r: left
    x_r = z_cam + cam_extr["x_forward_m"]
    y_r = (-x_cam) + cam_extr["y_left_m"]

    # Apply camera yaw offset relative to robot if needed
    yaw_off = math.radians(cam_extr.get("yaw_deg", 0.0))
    c0, s0 = math.cos(yaw_off), math.sin(yaw_off)
    x_r2 = c0 * x_r - s0 * y_r
    y_r2 = s0 * x_r + c0 * y_r

    # Robot -> global
    yaw = math.radians(pose["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    gx = pose["x"] + c * x_r2 - s * y_r2
    gy = pose["y"] + s * x_r2 + c * y_r2
    return gx, gy


def remove_all_objects_in_current_region(objects, pose):
    """
    Hard overwrite policy, but constrained to a trusted distance band.
    Prevents close-proximity amnesia and limits far-range ghost deletions.
    """
    kept = []
    removed = []

    hfov = CAM_HFOV_DEG + OVERWRITE_HFOV_MARGIN_DEG
    
    rmin_del = max(CAM_MIN_RANGE_M, OVERWRITE_DELETE_MIN_RANGE_M)
    rmax_del = min(CAM_MAX_RANGE_M + OVERWRITE_RANGE_MARGIN_M, OVERWRITE_DELETE_MAX_RANGE_M)

    for o in objects:
        if in_fov_wedge(
            o["x"], o["y"], pose, CAM_EXTR,
            hfov_deg=hfov,
            r_min=rmin_del,
            r_max=rmax_del
        ):
            removed.append(o)
        else:
            kept.append(o)

    return kept, removed


def add_objects_from_frame(objects, detections, pose):
    """
    Insert exactly what is seen in the current frame.
    Deduplicates close objects of the same label to prevent memory bloat 
    outside the overwrite/delete band.
    """
    added = 0
    updated = 0
    
    for det in detections:
        label = normalize_label(det.get("label", "UNKNOWN"))
        cam_xyz = det.get("camera_xyz_m", None)
        if cam_xyz is None:
            continue

        gxy = det_to_global_xy(cam_xyz, pose, CAM_EXTR)
        if gxy is None:
            continue
        gx, gy = gxy

        # Deduplication / Merge logic
        merged = False
        for obj in objects:
            if obj["label"] == label:
                dist = math.hypot(obj["x"] - gx, obj["y"] - gy)
                if dist < MERGE_RADIUS_M:
                    # Update existing object to the latest observed position
                    obj["x"] = float(gx)
                    obj["y"] = float(gy)
                    obj["conf"] = float(det.get("conf", 0.0))
                    merged = True
                    updated += 1
                    break
        
        if not merged:
            objects.append({
                "label": label,
                "x": float(gx),
                "y": float(gy),
                "conf": float(det.get("conf", 0.0)),
            })
            added += 1

    return added, updated


def publish_object_memory(objects):
    """
    Publishes the current global semantic object map.
    """
    payload = {
        "ts": time.time(),
        "objects": objects
    }
    r_db.setex("robot_object_memory", 180, json.dumps(payload))


def print_object_map(objects, pose=None, header="GLOBAL CAMERA OBJECT MAP"):
    """
    Prints the ENTIRE current global object map in terminal for debugging.
    """
    print("\n" + "=" * 100)
    print(header)
    if pose:
        print(f"Robot pose used: x={pose['x']:.2f}, y={pose['y']:.2f}, yaw={pose['yaw']:.1f} deg")
    print("-" * 100)

    if not objects:
        print("(empty)")
        print("=" * 100)
        return

    objs = sorted(objects, key=lambda o: (o["label"], o["x"], o["y"]))

    print(f"{'LABEL':<16} {'X':>10} {'Y':>10} {'CONF':>8}")
    print("-" * 100)
    for o in objs:
        print(f"{o['label']:<16} {o['x']:>10.2f} {o['y']:>10.2f} {o.get('conf', 0.0):>8.2f}")
    print("=" * 100)


def main():
    pose_hist = PoseHistory(horizon_s=POSE_HISTORY_SEC)
    last_camera_frame_id = -1
    objects = []  # Global semantic object map: [{"label","x","y","conf"}, ...]

    print("[Semantic] Starting camera semantic map service (trusted distance band + dedupe mode)...")
    print(f"[Semantic] Pose history: {POSE_HISTORY_SEC:.1f}s")
    print(f"[Semantic] Camera FOV wedge: {CAM_HFOV_DEG:.1f} deg, range [{CAM_MIN_RANGE_M:.2f}, {CAM_MAX_RANGE_M:.2f}] m")
    print(f"[Semantic] Delete band: [{OVERWRITE_DELETE_MIN_RANGE_M:.2f}, {OVERWRITE_DELETE_MAX_RANGE_M:.2f}] m")
    print(f"[Semantic] Dedupe merge radius: {MERGE_RADIUS_M:.2f} m")
    print("[Semantic] Waiting for robot_packet + robot_camera_raw ...")

    while True:
        try:
            # 1) Update pose history from tracker heartbeat
            pose_hist.update_from_robot_packet()

            # 2) Read latest camera packet
            raw = r_db.get("robot_camera_raw")
            if not raw:
                time.sleep(0.05)
                continue

            pkt = json.loads(raw)
            if pkt.get("status") != "success":
                time.sleep(0.05)
                continue

            frame_id = int(pkt.get("frame_id", -1))
            if frame_id == last_camera_frame_id:
                time.sleep(0.05)
                continue
            last_camera_frame_id = frame_id

            # 3) Select camera timestamp for pose alignment
            if USE_SERVER_RX_TIME:
                cam_ts_sec = float(pkt.get("server_rx_ts_ms", 0)) / 1000.0
            else:
                cam_ts_sec = float(pkt.get("capture_ts_ms", 0)) / 1000.0

            # 4) Nearest pose from recent history
            pose = pose_hist.nearest(cam_ts_sec)
            if pose is None:
                print(f"[Semantic] frame={frame_id}: no pose match in history (cam_ts={cam_ts_sec:.3f})")
                time.sleep(0.05)
                continue

            detections = pkt.get("detections", [])

            # 5) Overwrite current visible region in global object map
            objects, removed_now = remove_all_objects_in_current_region(objects, pose)

            # 6) Add current detections (with spatial deduplication)
            added_count, updated_count = add_objects_from_frame(objects, detections, pose)

            # 7) Publish map for downstream integration
            publish_object_memory(objects)

            # 8) Print ENTIRE global map in terminal (debug)
            if PRINT_EVERY_NEW_CAMERA_FRAME:
                print_object_map(objects, pose=pose, header=f"GLOBAL CAMERA OBJECT MAP | frame={frame_id}")
                print(f"[Semantic] Removed in visible region: {len(removed_now)}")
                print(f"[Semantic] Added: {added_count} | Updated/Merged: {updated_count}")

            time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[Semantic] Stopping.")
            break
        except Exception as e:
            print(f"[Semantic] Error: {e}")
            time.sleep(0.2)


if __name__ == "__main__":
    main()
