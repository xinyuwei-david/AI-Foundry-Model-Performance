#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
call_muse_iterative_debug.py  (2025‑04‑27)

Functionality
-------------
1. Command‑line flag --steps N controls the number of frames to generate (default 10)
2. Calls a Muse / WHAM endpoint; saves raw frames to raw/ and 4× super‑resolved
   frames to sr/
3. Creates an animated GIF (sr/dream_x4.gif)
4. Prints the first 2 KB of the server response; automatically captures any
   16‑dimensional action array
5. If the endpoint returns no action vector → falls back to a more diverse
   random input (sticks + buttons) to keep the scene changing

Dependencies
------------
pip install pillow imageio
(optional SR) pip install realesrgan torch
"""

import argparse
import base64
import getpass
import io
import json
import os
import random
import time
import urllib.request
from typing import List, Optional

from PIL import Image
import imageio.v3 as imageio

# ---------------------------------------------------------------------------
# Super‑resolution settings
# ---------------------------------------------------------------------------
USE_SR = False  # Set to True → requires realesrgan + torch

if USE_SR:
    try:
        from realesrgan import RealESRGAN
        import torch
        import numpy as np

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _sr = RealESRGAN(device, scale=4)
        _sr.load_weights("RealESRGAN_x4.pth")
        print("✅ Real‑ESRGAN 4× up‑scaling enabled.")
    except Exception as e:
        print("⚠️  Failed to initialize Real‑ESRGAN; falling back to Lanczos:", e)
        USE_SR = False


def upsample(img: Image.Image) -> Image.Image:
    """Upsample a PIL image by 4× using Real‑ESRGAN or Lanczos."""
    if USE_SR:
        import numpy as np

        return Image.fromarray(_sr.predict(np.array(img)))
    return img.resize((img.width * 4, img.height * 4), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# File system constants
# ---------------------------------------------------------------------------
RAW_DIR, SR_DIR = "raw", "sr"
GIF_PATH, CSV_PATH = "sr/dream_x4.gif", "actions.csv"
PAYLOAD_PATH = "musePayload.txt"

# ---------------------------------------------------------------------------
# 16‑D action head names (for CSV output)
# ---------------------------------------------------------------------------
ACTION_HEAD = [
    "left_stick_x",
    "left_stick_y",
    "right_stick_x",
    "right_stick_y",
    "trigger_LT",
    "trigger_RT",
    "button_A",
    "button_B",
    "button_X",
    "button_Y",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "skill_1",
    "skill_2",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def build_headers(api_key: str):
    """Build HTTP headers for the Azure/HF endpoint."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer " + api_key,
    }


def fallback_action() -> List[float]:
    """
    Generate a richer random action vector:
    - Sticks sampled from [-1, 1]
    - RT pressed with 40 % probability
    - Exactly one random d‑pad key is pressed
    """
    v = [0.0] * 16
    v[0], v[1] = random.uniform(-1, 1), random.uniform(-1, 1)  # left stick
    v[2], v[3] = random.uniform(-1, 1), random.uniform(-1, 1)  # right stick
    v[5] = 1.0 if random.random() < 0.4 else 0.0  # RT
    dpad_index = 10 + random.randint(0, 3)  # choose one d‑pad key
    v[dpad_index] = 1.0
    return v


def pil_to_b64(img: Image.Image, size=(300, 180)) -> str:
    """Downscale a PIL image and return its Base64‑encoded PNG bytes."""
    buf = io.BytesIO()
    img.resize(size, Image.Resampling.LANCZOS).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def muse_call(payload: dict, hdr: dict, url: str):
    """Send an HTTP request to the Muse / WHAM endpoint and parse the response."""
    req = urllib.request.Request(url, json.dumps(payload).encode(), hdr)
    with urllib.request.urlopen(req) as r:
        js = json.loads(r.read().decode())

    # Debug: print first 2 KB of the response
    print("── server response (first 2 KB) ──")
    print(json.dumps(js, indent=2)[:2048], "\n────────────────────────")

    img_b64 = js["results"][0]["image"]
    img = Image.open(io.BytesIO(base64.b64decode(img_b64)))

    act, act_key = None, None
    for k, v in js["results"][0].items():
        if (
            isinstance(v, (list, tuple))
            and len(v) == 16
            and all(isinstance(x, (int, float)) for x in v)
        ):
            act, act_key = list(map(float, v)), k
            break
    return img, act, act_key, list(js["results"][0].keys())


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Iteratively call a Muse / WHAM endpoint and build a GIF."
    )
    parser.add_argument("--steps", type=int, default=10, help="number of frames (default 10)")
    parser.add_argument("--endpoint", type=str, help="AI endpoint URL")
    parser.add_argument("--key", type=str, help="API key for the endpoint")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve endpoint & key: CLI flag → ENV var → interactive prompt
    # -----------------------------------------------------------------------
    ENDPOINT_URL = (
        args.endpoint
        or os.getenv("MUSE_ENDPOINT_URL")
        or input("Enter ENDPOINT_URL: ").strip()
    )
    API_KEY = (
        args.key
        or os.getenv("MUSE_API_KEY")
        or getpass.getpass("Enter API_KEY (input hidden): ").strip()
    )

    # -----------------------------------------------------------------------
    # Create output folders & load initial payload
    # -----------------------------------------------------------------------
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SR_DIR, exist_ok=True)

    payload = json.load(open(PAYLOAD_PATH, "r", encoding="utf-8"))
    ctx, ctx_len = payload["input_data"]["context"], len(payload["input_data"]["context"])

    # Prepare CSV
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("step," + ",".join(ACTION_HEAD) + "\n")

    headers = build_headers(API_KEY)
    total_iter = args.steps

    # -----------------------------------------------------------------------
    # Main inference loop
    # -----------------------------------------------------------------------
    for step in range(total_iter):
        print(f"\n🚀 Inference {step + 1}/{total_iter}")
        try:
            img, act, act_key, keys = muse_call(payload, headers, ENDPOINT_URL)
        except Exception as e:
            print("❌ HTTP error:", e)
            break

        if act is None:
            act = fallback_action()
            print(f"⚠️  No 16‑D action found; using random fallback (keys={keys})")
        else:
            print(f"✅ Captured action field: '{act_key}'")

        # Save the raw image
        raw_path = f"{RAW_DIR}/{step + 1:02d}.png"
        img.save(raw_path)

        # Save the 4× up‑sampled image
        upsample(img).save(f"{SR_DIR}/{step + 1:02d}_x4.png")

        # Append to CSV
        with open(CSV_PATH, "a", encoding="utf-8") as f:
            f.write(f"{step + 1}," + ",".join(map(str, act)) + "\n")

        # Update context for the next call
        if len(ctx) >= ctx_len:
            ctx.pop(0)
        ctx.append(
            {"image": pil_to_b64(img), "actions": act, "actions_output": act, "tokens": []}
        )

        # Optional sleep between calls
        if step < total_iter - 1:
            time.sleep(3)

    # -----------------------------------------------------------------------
    # Build the animated GIF
    # -----------------------------------------------------------------------
    frames = [imageio.imread(f"{SR_DIR}/{i + 1:02d}_x4.png") for i in range(step + 1)]
    imageio.imwrite(GIF_PATH, frames, duration=0.25, loop=0)
    print(f"\n🎉 Done. Generated {step + 1} frames.")
    print(f"   GIF : {GIF_PATH}")
    print(f"   CSV : {CSV_PATH}")


if __name__ == "__main__":
    main()