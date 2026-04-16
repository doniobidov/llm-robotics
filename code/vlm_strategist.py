import os
import torch
import time
import json
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- CONFIG ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct" 

class VLMStrategist:
    def __init__(self):
        print(f"--- LOADING VLM ({MODEL_ID}) ON GPU 1 ---")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    def select_goal(self, base64_image, metadata, user_objective, semantic_context, candidate_points_text, current_locked_target=None):
        size_m = metadata["view_size_m"]
        top_k = 3
        
        system_prompt = f"""You are a navigation strategist for an autonomous robot. Select one target point.

USER_OBJECTIVE: "{user_objective}"
SEMANTIC MEMORY: {semantic_context}
AVAILABLE CANDIDATE POINTS [u, v]: {candidate_points_text}

MAP:
- Top-down occupancy + visit heatmap centered on the robot.
- Robot is at the center.
- White = obstacles/walls.
- Dark blue = less visited, red/purple = more visited.
- Green dots = available candidate points.
- Axis labels and borders are UI only, not obstacles.

RULES:
1) If USER_OBJECTIVE (or obvious equivalent) is present in SEMANTIC MEMORY, choose that object.
2) If the exact target is not in SEMANTIC MEMORY, choose exactly one point from AVAILABLE CANDIDATE POINTS.
3) If AVAILABLE CANDIDATE POINTS are non-empty, do not invent another point unless the exact target is in SEMANTIC MEMORY.
4) When exploring, prefer less visited candidate points and avoid highly visited areas when possible.
5) Do not choose a point immediately next to the robot for exploration (0.37 < u < 0.62 and 0.37 < v < 0.62), unless the exact target is there in SEMANTIC MEMORY.
6) Choose only free-space points, not white obstacles.
7) Output valid JSON only.

OUTPUT FORMAT:
{{
  "reasoning": "short reason",
  "target_uv": [u, v]
}}

Return only one JSON object. u and v must be numbers in [0, 1]. No markdown, no extra text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                    {"type": "text", "text": f"Select the next navigation target for {user_objective}."},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        start_t = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=250, temperature=0.1)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        resp = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        latency = (time.time() - start_t) * 1000

        try:
            match = re.search(r'\{.*?\}', resp, re.DOTALL)
            if match:
                data = json.loads(match.group())
                
                u, v = data.get("target_uv", [0.5, 0.5]) 
                u, v = float(u), float(v)
                
                # REJECT hallucinated coordinates instead of silently clamping them
                if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                    raise ValueError(f"LLM output UV out of bounds: [{u}, {v}]")
                
                rx, ry = metadata["robot_x"], metadata["robot_y"]
                target_wx = rx + (u - 0.5) * size_m
                target_wy = ry - (v - 0.5) * size_m 
                
                return [target_wx, target_wy], [u, v], data.get("reasoning", ""), latency, f"Target: {user_objective}", resp
        except Exception as e:
            print(f"VLM Parse Error: {e}")
            
        return [metadata["robot_x"], metadata["robot_y"]], [0.5, 0.5], "Parse Error", latency, f"Target: {user_objective}", resp
