import time
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import queue
from collections import deque, defaultdict

# ---------------- CONFIG ----------------
CRITICAL = {
    'person','bicycle','car','motorcycle','bus','truck',
    'traffic light','stop sign','bench','chair','potted plant',
    'dog','cat','backpack','umbrella','suitcase'
}
CONF_THRESHOLD = 0.45          # lower slightly to not miss things (use caution)
AREA_CLOSE = 0.11              # normalized area -> very close (stop)
AREA_NEAR = 0.045              # normalized area -> nearby (go-around)
ALERT_COOLDOWN = 2.0           # base cooldown between spoken alerts
TTS_QUEUE_MAX = 4
FRAME_RESIZE = (640, 360)      # inference size (width, height) - lowers latency
DRAW_EVERY_N_FRAMES = 2        # only draw every N frames to save CPU
SMOOTH_ALPHA = 0.4             # EMA smoothing factor (0..1). Higher = faster response.
PERSISTENCE_FRAMES = 3         # require object to appear for N frames before reacting
USE_GPU = torch.cuda.is_available()
# ----------------------------------------

# load model (yolov5s) and move to device + half precision if possible
device = 'cuda' if USE_GPU else 'cpu'
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    model.to(device)
    if USE_GPU:
        model.half()  # use FP16 for speed on CUDA
    model.eval()
except Exception as e:
    raise SystemExit(f"Failed to load model: {e}")

# TTS background thread (non-blocking)
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

tts_queue = queue.Queue(maxsize=TTS_QUEUE_MAX)
tts_running = True

def tts_worker(q):
    while True:
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            if not tts_running:
                break
            continue
        if msg is None:
            break
        tts_engine.say(msg)
        tts_engine.runAndWait()
        q.task_done()

tts_thread = threading.Thread(target=tts_worker, args=(tts_queue,), daemon=True)
tts_thread.start()

# helper: speak without blocking
def speak(message):
    try:
        # make sure queue won't block main thread
        tts_queue.put_nowait(message)
    except queue.Full:
        # drop old messages if queue is full (prevents blocking)
        try:
            _ = tts_queue.get_nowait()
            tts_queue.put_nowait(message)
        except Exception:
            pass

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

# Optionally set camera resolution (helps model if camera returns huge frames)
# Uncomment if you want specific capture resolution:
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

names = model.names

# smoothing / persistence trackers
zone_ema = {'left': 0.0, 'center': 0.0, 'right': 0.0}      # EMA of largest area seen per zone
zone_presence = {'left': deque(maxlen=PERSISTENCE_FRAMES),
                 'center': deque(maxlen=PERSISTENCE_FRAMES),
                 'right': deque(maxlen=PERSISTENCE_FRAMES)}

last_spoken = ""
last_alert_time = 0.0
frame_idx = 0

def draw_debug(frame, preds, frame_w, frame_h):
    """Draw bounding boxes (preds expected as Nx6: x1,y1,x2,y2,conf,cls)."""
    for x1, y1, x2, y2, conf, cls in preds:
        cv2.rectangle(frame, (int(x1*frame_w), int(y1*frame_h)),
                             (int(x2*frame_w), int(y2*frame_h)), (0,255,0), 2)
        cls_name = names[int(cls)]
        cv2.putText(frame, f"{cls_name} {conf:.2f}",
                    (int(x1*frame_w), int(y1*frame_h)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    return frame

print("Tuned detection started. Press 'q' to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed")
            break

        frame_idx += 1

        # Resize for faster inference (maintain aspect)
        small = cv2.resize(frame, FRAME_RESIZE)
        # convert BGR->RGB and optionally to half precision tensor on device
        img = small[:, :, ::-1]  # BGR->RGB
        # run model (yolov5 handles numpy input directly)
        # specify size param to model for internal resizing if necessary
        results = model(img, size=FRAME_RESIZE[0])  # size is width in ultralytics API
        preds = results.xyxyn[0].cpu().numpy()  # x1,y1,x2,y2,conf,cls

        # prepare arrays for zones
        zones_found = {'left': [], 'center': [], 'right': []}
        debug_preds = []

        for *box, conf, cls in preds:
            if conf < CONF_THRESHOLD:
                continue
            cls_name = names[int(cls)]
            if cls_name not in CRITICAL:
                continue

            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1) * (y2 - y1))   # normalized area
            x_center = (x1 + x2) / 2
            if x_center < 0.33:
                z = 'left'
            elif x_center > 0.66:
                z = 'right'
            else:
                z = 'center'
            zones_found[z].append(area)
            debug_preds.append([x1, y1, x2, y2, conf, cls])

        # For each zone, compute "largest area" this frame (or 0)
        largest_by_zone = {z: (max(v) if v else 0.0) for z, v in zones_found.items()}

        # Persistence: mark presence for last few frames (True/False)
        for z in ['left','center','right']:
            zone_presence[z].append(1 if largest_by_zone[z] > 0 else 0)

        # EMA smoothing: update only if object present, otherwise slowly decay
        for z in ['left','center','right']:
            observed = largest_by_zone[z]
            # decay behavior if observed==0: multiply previous by (1-alpha) to slowly decay
            zone_ema[z] = SMOOTH_ALPHA * observed + (1 - SMOOTH_ALPHA) * zone_ema[z]

        # Decide message (respect cooldown, and require persistence)
        now = time.time()
        message = ""
        if now - last_alert_time >= ALERT_COOLDOWN:
            # Only consider zones that had presence in >= half of persistence frames
            valid_zones = {z for z in ['left','center','right']
                           if sum(zone_presence[z]) >= (PERSISTENCE_FRAMES // 2 + 1)}

            # Compose decision based on EMA values (more stable than raw area)
            # Find zone with max ema
            primary_zone = max(['left','center','right'], key=lambda z: zone_ema[z])
            primary_area = zone_ema[primary_zone]

            if primary_area > AREA_CLOSE:
                if primary_zone == 'center':
                    message = "Warning! Stop. Object very close."
                else:
                    message = f"Warning! Object very close on the {primary_zone}."
            elif primary_area > AREA_NEAR:
                if primary_zone == 'center':
                    message = "Obstacle ahead. Please go around."
                elif primary_zone == 'left':
                    message = "Obstacle on the left. Please go right."
                else:
                    message = "Obstacle on the right. Please go left."
            else:
                # If no strong primary, build summary from valid zones
                if not valid_zones:
                    message = "Path is clear."
                else:
                    if 'center' in valid_zones:
                        message = "Obstacle ahead."
                    elif valid_zones == {'left'}:
                        message = "Obstacle on the left."
                    elif valid_zones == {'right'}:
                        message = "Obstacle on the right."
                    elif 'left' in valid_zones and 'right' in valid_zones:
                        message = "Obstacles on both sides."

            # speak if message changed (dedupe) and not empty
            if message and message != last_spoken:
                # dynamic adjustment: if message is critical, reduce cooldown a bit
                if "Warning! Stop" in message:
                    # immediate urgent message; shorter cooldown so follow-ups possible
                    last_alert_time = now
                else:
                    last_alert_time = now
                last_spoken = message
                print(f"[{time.strftime('%H:%M:%S')}] Speak:", message)
                speak(message)

        # Draw debug less often to save CPU
        if frame_idx % DRAW_EVERY_N_FRAMES == 0:
            # draw on full-size frame for display clarity
            frame_h, frame_w = frame.shape[:2]
            vis = draw_debug(frame, debug_preds, frame_w, frame_h)
            # overlay simple EMA gauges
            cv2.putText(vis, f"EMA L:{zone_ema['left']:.3f} C:{zone_ema['center']:.3f} R:{zone_ema['right']:.3f}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
            cv2.imshow("Smart Glass (tuned)", vis)

        # keyboard quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # stop TTS thread
    tts_running = False
    try:
        # flush queue then join
        tts_queue.put(None)
    except Exception:
        pass
    tts_thread.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
