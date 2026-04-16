import os
import io
import json
import time
import cv2
import base64
import numpy as np
import redis
from threading import Lock
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from ultralytics import YOLO

# --- CONFIG ---
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # keep VLM on GPU 1, camera on GPU 0
PORT = 8001
MODEL_NAME = "yolov8x-worldv2.pt"

CONF_THRES = 0.20

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

CUSTOM_CLASSES = [
    "door", "doorway", "door frame", "person", "laptop",
    "desk", "chair", "table", "whiteboard", "monitor",
    "pc", "backpack", "handbag", "purse"
]

# --------------
app = FastAPI()
predict_lock = Lock()
r_db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

print(f"[Server] Loading {MODEL_NAME} ...")
model = YOLO(MODEL_NAME)
if CUSTOM_CLASSES:
    model.set_classes(CUSTOM_CLASSES)
    print(f"[Server] Custom classes set: {CUSTOM_CLASSES}")

def get_robust_depth(depth_img: np.ndarray, x1, y1, x2, y2) -> float:
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    half_w = 6
    min_y = max(0, cy - half_w)
    max_y = min(depth_img.shape[0], cy + half_w)
    min_x = max(0, cx - half_w)
    max_x = min(depth_img.shape[1], cx + half_w)

    region = depth_img[min_y:max_y, min_x:max_x]
    if region.size == 0: return 0.0

    valid = region[(region > 0.2) & (region < 8.0)]
    if len(valid) == 0: return 0.0

    return float(np.median(valid))

def publish_debug_image(result, detections):
    try:
        debug_img = result.plot()
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            label = det["label"]
            conf = det["conf"]
            x, y, z = det["camera_xyz_m"]
            txt = f"{label} {conf:.2f} Z:{z:.2f}m"
            cv2.putText(
                debug_img, txt, (int(x1), max(20, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        _, buffer = cv2.imencode('.jpg', debug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64_img = base64.b64encode(buffer).decode('utf-8')
        r_db.setex("robot_camera_debug_img", 10, b64_img)
    except Exception as e:
        print(f"[Server] Debug stream error: {e}")

@app.post("/predict")
async def predict(
    color_file: UploadFile = File(...),
    depth_file: UploadFile = File(...),
    intrinsics_json: str = Form(...),
    frame_id: str = Form(...),
    capture_ts_ms: str = Form(...),
    camera_id: str = Form("front_realsense"),
):
    if not predict_lock.acquire(blocking=False):
        return {"status": "skipped"}

    server_rx_ts_ms = int(time.time() * 1000)

    try:
        color_bytes = await color_file.read()
        depth_bytes = await depth_file.read()

        color_np = np.frombuffer(color_bytes, np.uint8)
        img = cv2.imdecode(color_np, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "error", "message": "Failed to decode color image"}

        depth_img = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
        
        # Restored sanity check
        if depth_img.ndim != 2:
            return {"status": "error", "message": "Depth image must be 2D"}
            
        intr = json.loads(intrinsics_json)
        fx, fy = float(intr["fx"]), float(intr["fy"])
        ppx, ppy = float(intr["ppx"]), float(intr["ppy"])

        results = model.predict(img, conf=CONF_THRES, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = str(model.names[cls_id]).upper()

            z = get_robust_depth(depth_img, x1, y1, x2, y2)
            if z <= 0: continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            x = (cx - ppx) * z / fx
            y = (cy - ppy) * z / fy

            detections.append({
                "label": label, "conf": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "camera_xyz_m": [x, y, z],
            })

        server_done_ts_ms = int(time.time() * 1000)

        payload = {
            "status": "success", "camera_id": camera_id, "frame_id": int(frame_id),
            "capture_ts_ms": int(capture_ts_ms), "server_rx_ts_ms": server_rx_ts_ms,
            "server_done_ts_ms": server_done_ts_ms, "detections": detections,
        }

        r_db.setex("robot_camera_raw", 10, json.dumps(payload))
        
        # Stream image to debug visualizer
        publish_debug_image(result, detections)

        return payload

    except Exception as e:
        print(f"[Server] Error processing frame: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        predict_lock.release()

if __name__ == "__main__":
    print(f"[Server] Starting camera server on :{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
