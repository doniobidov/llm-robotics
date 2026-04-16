import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import math
from navigation_core import GRID_RES, VISIT_CELL_SIZE

# Expanded to 16.0m to match the ±8.0m telemetry window
VIEW_SIZE_M = 16.0
# Add +1 to ensure an odd dimension (321x321) for a true center pixel
VIEW_DIM = int(VIEW_SIZE_M / GRID_RES) + 1


def generate_snapshot_image(grid, origin_x, origin_y, robot_x, robot_y, robot_yaw, visits, candidates=None):
    """Generates a perfectly scaled top-down RGB image centered on the robot using fast NumPy vectorization."""
    CENTER_IDX = grid.shape[0] // 2
    img_arr = np.full((VIEW_DIM, VIEW_DIM, 3), [0, 0, 50], dtype=np.uint8)  # Dark Blue base
    half_dim = VIEW_DIM // 2

    # 1. Paint Visits (Loop over the small dictionary instead of every pixel)
    for (vx, vy), vc in visits.items():
        if vc <= 0:
            continue

        # World boundaries for this visit cell
        w_left = vx * VISIT_CELL_SIZE
        w_right = w_left + VISIT_CELL_SIZE
        w_bottom = vy * VISIT_CELL_SIZE
        w_top = w_bottom + VISIT_CELL_SIZE

        # Convert to pixel boundaries using math.floor to prevent zero-crossing jitter
        j_min = int(math.floor((w_left - robot_x) / GRID_RES)) + half_dim
        j_max = int(math.floor((w_right - robot_x) / GRID_RES)) + half_dim
        i_min = half_dim - int(math.floor((w_top - robot_y) / GRID_RES))  # Image Y goes down
        i_max = half_dim - int(math.floor((w_bottom - robot_y) / GRID_RES))

        # Clip to image bounds
        j_min, j_max = max(0, j_min), min(VIEW_DIM, j_max)
        i_min, i_max = max(0, i_min), min(VIEW_DIM, i_max)

        if j_min < j_max and i_min < i_max:
            intensity = min(255, 50 + vc * 30)
            img_arr[i_min:i_max, j_min:j_max] = [intensity, 0, 255 - intensity]

    # 2. Paint Obstacles (Vectorized mapping with np.rint for float precision stability)
    j, i = np.meshgrid(np.arange(VIEW_DIM), np.arange(VIEW_DIM))
    wx = robot_x + (j - half_dim) * GRID_RES
    wy = robot_y - (i - half_dim) * GRID_RES

    gx = np.rint((wx - origin_x) / GRID_RES).astype(np.int32) + CENTER_IDX
    gy = np.rint((wy - origin_y) / GRID_RES).astype(np.int32) + CENTER_IDX

    # Identify valid grid coordinates and apply white pixels where obstacles exist
    valid = (gx >= 0) & (gx < grid.shape[0]) & (gy >= 0) & (gy < grid.shape[1])
    obs_mask = np.zeros_like(valid, dtype=bool)
    obs_mask[valid] = grid[gx[valid], gy[valid]] == 1
    img_arr[obs_mask] = [255, 255, 255]

    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)

    # Optional: border frame (helps visually, safe)
    draw.rectangle([(0, 0), (VIEW_DIM - 1, VIEW_DIM - 1)], outline=(0, 0, 0), width=2)

    # --- Draw U, V Axes with outline so they remain visible over white walls ---
    axis_fg = (255, 255, 255)  # white
    axis_bg = (0, 0, 0)        # black outline
    tick_len = 10
    tick_w = 2

    def draw_text_outline(x, y, s):
        # 1px outline around text
        draw.text((x - 1, y), s, fill=axis_bg)
        draw.text((x + 1, y), s, fill=axis_bg)
        draw.text((x, y - 1), s, fill=axis_bg)
        draw.text((x, y + 1), s, fill=axis_bg)
        draw.text((x, y), s, fill=axis_fg)

    for k in range(1, 10):
        val = k / 10.0
        label = f"{val:.1f}"

        # U Axis (Top edge)
        px = int(val * (VIEW_DIM - 1))
        draw.line([(px, 0), (px, tick_len)], fill=axis_bg, width=tick_w + 2)
        draw.line([(px, 0), (px, tick_len)], fill=axis_fg, width=tick_w)
        draw_text_outline(px - 8, tick_len + 2, label)

        # V Axis (Left edge)
        py = int(val * (VIEW_DIM - 1))
        draw.line([(0, py), (tick_len, py)], fill=axis_bg, width=tick_w + 2)
        draw.line([(0, py), (tick_len, py)], fill=axis_fg, width=tick_w)
        draw_text_outline(tick_len + 2, py - 6, label)

    # --- Draw Candidate Points ---
    if candidates:
        for u, v in candidates:
            c_px = int(u * (VIEW_DIM - 1))
            c_py = int(v * (VIEW_DIM - 1))
            draw.ellipse([(c_px - 3, c_py - 3), (c_px + 3, c_py + 3)], fill=(0, 255, 0))

    # 3. Draw robot and heading (Crisp arrowhead)
    yaw_rad = math.radians(robot_yaw)
    cx, cy = half_dim, half_dim
    length = 22
    head_len = 7
    head_w = 5
    line_w = 3

    ex = cx + math.cos(yaw_rad) * length
    ey = cy - math.sin(yaw_rad) * length  # Image Y goes down

    cx_i, cy_i = int(round(cx)), int(round(cy))
    ex_i, ey_i = int(round(ex)), int(round(ey))

    draw.line([(cx_i, cy_i), (ex_i, ey_i)], fill=(0, 255, 255), width=line_w)

    # Arrowhead triangle
    bx = ex - math.cos(yaw_rad) * head_len
    by = ey + math.sin(yaw_rad) * head_len

    # Perpendicular vector for the width of the arrow
    px = -math.sin(yaw_rad)
    py = -math.cos(yaw_rad)

    left_x = bx + px * head_w
    left_y = by + py * head_w
    right_x = bx - px * head_w
    right_y = by - py * head_w

    draw.polygon(
        [
            (ex_i, ey_i),
            (int(round(left_x)), int(round(left_y))),
            (int(round(right_x)), int(round(right_y))),
        ],
        fill=(0, 255, 255),
    )

    # Robot body dot
    draw.ellipse([(cx_i - 4, cy_i - 4), (cx_i + 4, cy_i + 4)], fill=(0, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    metadata = {
        "view_size_m": VIEW_SIZE_M,
        "view_dim_px": VIEW_DIM,
        "robot_x": robot_x,
        "robot_y": robot_y,
    }
    return b64, metadata
