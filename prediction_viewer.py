import cv2
import pandas as pd
import numpy as np
import colorsys
import tkinter as tk
from tkinter import filedialog

# === Toggle console output ===
VERBOSE = False  # Set to False to silence all print statements

# === File selection GUI ===
root = tk.Tk()
root.withdraw()

if VERBOSE:
    print("CBAS Prediction Viewer")

video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
)
if not video_path:
    if VERBOSE:
        print("No video file selected. Exiting.")
    exit()

csv_path = filedialog.askopenfilename(
    title="Select Prediction CSV File",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not csv_path:
    if VERBOSE:
        print("No CSV file selected. Exiting.")
    exit()

if VERBOSE:
    print(f"Video: {video_path}")
    print(f"CSV:   {csv_path}")

# === Config ===
CONFIDENCE_THRESHOLD = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_scale_secondary = 0.6
thickness = 2
side_margin = 400

# === Load predictions ===
df = pd.read_csv(csv_path)
behavior_names = df.columns[1:].tolist()
predictions = df.iloc[:, 1:].values
n_behaviors = len(behavior_names)
frame_count = len(df)

def generate_colors(n):
    hsv_colors = [(i / n, 1, 1) for i in range(n)]
    return [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]

colors = generate_colors(n_behaviors)

# === Load video ===
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0
playing = False
playback_delay = 100
default_delay = 100
min_delay = 5
max_delay = 1000

cv2.namedWindow('CBAS Prediction Viewer', cv2.WINDOW_NORMAL)
cv2.resizeWindow('CBAS Prediction Viewer', 1280, 720)
cv2.setWindowProperty('CBAS Prediction Viewer', cv2.WND_PROP_TOPMOST, 1)
cv2.createTrackbar('Frame', 'CBAS Prediction Viewer', 0, frame_count - 1, lambda x: None)

def fade_color(color, alpha):
    r, g, b = color
    return tuple(int((1 - alpha) * 255 + alpha * c) for c in (r, g, b))

def pad_frame(frame, pad_width=side_margin, pad_height=60):
    height, width = frame.shape[:2]
    padded = np.zeros((height + pad_height, width + pad_width, 3), dtype=np.uint8)
    padded[pad_height:, :width] = frame
    return padded

def draw_overlay(frame, idx1, prob1, idx2, prob2, frame_idx):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w - side_margin + 10, 20, w - 10, 100
    y1 += 30
    y2 += 30
    label_origin = (x1 + 10, y1 + 30)
    secondary_offset = (x1 + 10, y1 + 70)

    strong = prob1 >= CONFIDENCE_THRESHOLD
    color1 = colors[idx1] if strong else fade_color(colors[idx1], 0.4)
    label1 = f"{behavior_names[idx1]} ({prob1:.2f})" if strong else f"? {behavior_names[idx1]} ({prob1:.2f})"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color1, 2 if strong else 1)
    cv2.putText(frame, label1, label_origin, font, font_scale, color1, thickness if strong else 1)

    color2 = fade_color(colors[idx2], 0.3)
    label2 = f"{behavior_names[idx2]} ({prob2:.2f})"
    cv2.rectangle(frame, (x1+5, y1+5), (x2-5, y2-5), color2, 1)
    cv2.putText(frame, label2, secondary_offset, font, font_scale_secondary, color2, 1)

    if playing:
        speed_mult = default_delay / playback_delay
        frame_label = f"Frame {frame_idx + 1} / {frame_count}  |  {speed_mult:.1f}x"
    else:
        frame_label = f"Frame {frame_idx + 1} / {frame_count}"

    cv2.putText(frame, frame_label, (20, 30), font, font_scale_secondary, (200, 200, 200), 1)

    help_lines = [
        "[SPACE] Play/Pause",
        "[A]/[D] Prev/Next Frame",
        "[W]/[S] -/+10 Frames / Speed",
        "[Q] Quit"
    ]
    for i, text in enumerate(help_lines):
        y = h - 10 - (len(help_lines) - i - 1) * 20
        cv2.putText(frame, text, (w - side_margin + 15, y),
                    font, 0.5, (180, 180, 180), 1)

last_frame_idx = -1

while cap.isOpened():
    trackbar_pos = cv2.getTrackbarPos('Frame', 'CBAS Prediction Viewer')
    if trackbar_pos != frame_idx:
        frame_idx = trackbar_pos

    key = cv2.waitKey(1 if not playing else playback_delay)
    if key != -1 and VERBOSE:
        print("Key:", key)

    if key == ord('q'):
        break
    elif key == ord(' '):
        playing = not playing
    elif key == ord('d'):
        if not playing:
            frame_idx = min(frame_idx + 1, frame_count - 1)
    elif key == ord('a'):
        if not playing:
            frame_idx = max(frame_idx - 1, 0)
    elif key == ord('w'):
        if not playing:
            frame_idx = min(frame_idx + 10, frame_count - 1)
        else:
            playback_delay = max(min_delay, playback_delay // 2)
            if VERBOSE:
                print(f"Speed up: {default_delay / playback_delay:.1f}×")
    elif key == ord('s'):
        if not playing:
            frame_idx = max(frame_idx - 10, 0)
        else:
            playback_delay = min(max_delay, playback_delay * 2)
            if VERBOSE:
                print(f"Slow down: {default_delay / playback_delay:.1f}×")

    if playing:
        frame_idx = (frame_idx + 1) % frame_count

    if frame_idx != last_frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame_idx >= frame_count:
            break

        row = predictions[frame_idx]
        sorted_indices = np.argsort(row)[::-1]
        top_idx, second_idx = sorted_indices[0], sorted_indices[1]
        top_prob, second_prob = row[top_idx], row[second_idx]

        padded_frame = pad_frame(frame)
        draw_overlay(padded_frame, top_idx, top_prob, second_idx, second_prob, frame_idx)

        cv2.imshow('CBAS Prediction Viewer', padded_frame)
        cv2.setTrackbarPos('Frame', 'CBAS Prediction Viewer', frame_idx)

        last_frame_idx = frame_idx

cap.release()
cv2.destroyAllWindows()
