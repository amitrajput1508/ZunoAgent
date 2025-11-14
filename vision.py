#!/usr/bin/env python3
"""
vision_openrouter.py
- Capture full-screen screenshot every INTERVAL seconds
- Optionally blur a sensitive area (like center) to avoid sending passwords
- Send image to OpenRouter Llama 3.2 11B Vision-Instruct
- Print and append model's text description to a log file
"""
#amit bahi h apana
import os
import time
import base64
import json
import requests
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFilter
import mss

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- Configuration ----------------
INTERVAL = 15                         # seconds between screenshots
MODEL = "meta-llama/llama-3.2-11b-vision-instruct"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT_DIR = Path("screen_logs")
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
LOG_FILE = OUTPUT_DIR / "descriptions.log"
DELETE_SCREENSHOT_AFTER_SEND = True   # set False to keep screenshots
BLUR_SENSITIVE_REGION = False         # If True, blur a box region before sending
# If BLUR_SENSITIVE_REGION True, configure this box (left, top, right, bottom)
SENSITIVE_BOX = None  # Example: (100, 100, 800, 400) or None to use center crop
# ------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Load API key
API_KEY = "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"
if not API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY in environment or .env file.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def take_screenshot(save_path: Path) -> None:
    with mss.mss() as sct:
        monitor = sct.monitors[0]  # full virtual screen
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        img.save(save_path)

def blur_region(img: Image.Image, box=None, radius=20) -> Image.Image:
    """
    Blur a rectangular region to obscure sensitive data.
    box = (left, top, right, bottom). If None, blur center 30% area.
    """
    if box is None:
        w, h = img.size
        bw, bh = int(w * 0.3), int(h * 0.2)
        left = (w - bw) // 2
        top = (h - bh) // 2
        box = (left, top, left + bw, top + bh)

    # Crop the region, blur it, and paste back
    region = img.crop(box)
    blurred = region.filter(ImageFilter.GaussianBlur(radius))
    img.paste(blurred, box)
    return img

def encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("utf-8")

def build_payload(base64_png: str, user_prompt: str = None):
    """
    Build the OpenRouter payload in the format that supports images.
    The 'content' array contains text and the image_url object.
    """
    if user_prompt is None:
        user_prompt = "Describe everything visible in this screenshot in detail. Mention objects, text, windows, and any notable activity."

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_png}"}
                    }
                ],
            }
        ],
        # optional: you can customise e.g. temperature, max tokens if openrouter supports them
        # "temperature": 0.1,
        # "max_tokens": 800
    }
    return payload

def send_to_openrouter(payload: dict, timeout: int = 60) -> dict:
    resp = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=payload, timeout=timeout)
    # raise for status to aid debugging
    resp.raise_for_status()
    return resp.json()

def extract_description_from_response(resp_json: dict) -> str:
    """
    OpenRouter's structure can vary slightly. This tries common places:
    - resp_json["choices"][0]["message"]["content"] could be a string or list/dict.
    We will fallback to dumping the full 'choices' for debugging.
    """
    try:
        choices = resp_json.get("choices", [])
        if not choices:
            return json.dumps(resp_json, indent=2)
        msg = choices[0].get("message", {})
        content = msg.get("content")
        # If content is a string, return it
        if isinstance(content, str):
            return content
        # If content is a list (blocks), try to extract text blocks
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                elif isinstance(block, str):
                    texts.append(block)
            if texts:
                return "\n".join(texts)
        # Else fallback to raw field "message"->"content" or the whole choices
        return json.dumps(msg, indent=2)
    except Exception as e:
        return f"Failed to parse response: {e}\nFull response: {json.dumps(resp_json)[:2000]}"

def append_log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def main_loop():
    counter = 0
    print(f"{get_timestamp()} Starting screen description tool. Press Ctrl+C to stop.")
    try:
        while True:
            counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = SCREENSHOT_DIR / f"screenshot_{ts}.png"
            try:
                take_screenshot(filename)
            except Exception as e:
                err = f"[{get_timestamp()}] Failed to take screenshot: {e}"
                print(err)
                append_log(err)
                time.sleep(INTERVAL)
                continue

            # optionally blur sensitive area
            if BLUR_SENSITIVE_REGION:
                try:
                    img = Image.open(filename)
                    img = blur_region(img, box=SENSITIVE_BOX)
                    img.save(filename)
                except Exception as e:
                    warn = f"[{get_timestamp()}] Warning: failed to blur sensitive region: {e}"
                    print(warn)
                    append_log(warn)

            try:
                b64 = encode_image_base64(filename)
                payload = build_payload(b64)
                # send
                resp_json = send_to_openrouter(payload)
            except requests.HTTPError as he:
                err = f"[{get_timestamp()}] HTTP error when calling OpenRouter: {he} -- Response: {getattr(he, 'response', None)}"
                print(err)
                append_log(err)
                # keep or delete screenshot as configured
                if DELETE_SCREENSHOT_AFTER_SEND and filename.exists():
                    filename.unlink(missing_ok=True)
                time.sleep(INTERVAL)
                continue
            except Exception as e:
                err = f"[{get_timestamp()}] Error when sending screenshot: {e}"
                print(err)
                append_log(err)
                if DELETE_SCREENSHOT_AFTER_SEND and filename.exists():
                    filename.unlink(missing_ok=True)
                time.sleep(INTERVAL)
                continue

            # parse response
            description = extract_description_from_response(resp_json)
            output_text = f"[{get_timestamp()}] {description}"
            print(output_text + "\n")
            append_log(output_text)

            if DELETE_SCREENSHOT_AFTER_SEND and filename.exists():
                try:
                    filename.unlink()
                except Exception:
                    pass

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print(f"\n[{get_timestamp()}] Stopped by user.")
    except Exception as e:
        print(f"[{get_timestamp()}] Unexpected error: {e}")
        append_log(f"[{get_timestamp()}] Unexpected error: {e}")

if __name__ == "__main__":
    main_loop()
