import os
import h5py
import io # Required for in-memory image saving

import time
import cairo
import base64
import math

from sklearn.metrics import classification_report, confusion_matrix

import shutil

import cv2
import torch
import transformers

import subprocess

from datetime import datetime

import numpy as np
import pandas as pd

import random

import yaml

import decord

import classifier_head  # The LSTM + linear classifier head

from torch import nn
import torch.optim as optim

# Add these for the new plotting function
import matplotlib
matplotlib.use('Agg') # Essential for running in a non-GUI thread
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def encode_file(encoder: nn.Module, path: str) -> str | None:
    """
    Given a video file path, use the provided DINOv2 encoder to extract one embedding per frame,
    and save them into an HDF5 file (path ending in "_cls.h5").
    Returns the path to the new HDF5 file, or None if encoding failed.
    """
    try:
        # Load the video frames into a NumPy array via decord (efficient decoding)
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
    except Exception as e:
        print(f"Error reading video {path} with decord: {e}")
        # If the video cannot be read (e.g., still being written), bail out.
        return None

    # Grab all frames, convert to NumPy, then to a Torch tensor (we take only green channel for B/W)
    frames_np = reader.get_batch(range(0, len(reader), 1)).asnumpy()  # shape: (T, H, W, 3)
    if frames_np.size == 0:
        print(f"Warning: Video {path} contains no frames or could not be decoded properly.")
        return None
    # Convert to single-channel (take green channel) and normalize to [0,1], then half-precision
    frames = torch.from_numpy(frames_np[:, :, :, 1] / 255).half()  # shape: (T, H, W)

    batch_size = 256  # chunk size for batching frames through DINOv2
    clss = []  # will store each chunk of embeddings

    # Process frames in batches to avoid GPU OOM
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]  # shape: (B, H, W)

        with torch.no_grad(), torch.amp.autocast(device_type=encoder.device.type if encoder.device.type != 'mps' else 'cpu'): # MPS doesn't support autocast well for all ops
            # DINOv2 expects a (B, S, H, W) input where S=1 for single frame processing
            # but its forward pass handles the unsqueeze for channel internally for (B,1,H,W)
            # The DinoEncoder class takes (B_outer, S_inner, H, W) where S_inner is sequence elements per sample
            # Here, we are passing (batch_size_from_frames, 1, H, W) to the encoder
            out = encoder(batch.unsqueeze(1).to(encoder.device))
            # encoder.forward returns shape: (B_chunk, 1, 768) → we squeeze → (B_chunk, 768)

        out = out.squeeze(1).to("cpu")  # bring embeddings back to CPU
        clss.extend(out)

    # Save all embeddings into a new HDF5 file alongside the video path
    out_file_path = os.path.splitext(path)[0] + "_cls.h5"
    with h5py.File(out_file_path, "w") as file:
        # "cls" dataset shape: (T, 768)
        file.create_dataset("cls", data=torch.stack(clss).numpy())

    return out_file_path


def infer_file(
    file_path: str,
    model: nn.Module,
    dataset_name: str,
    behaviors: list[str],
    seq_len: int,
    device=None,
) -> str | None:
    """
    Given a path to a _cls.h5 file (sequence of DINOv2 embeddings), run the classifier head
    on sliding windows of length seq_len to produce frame-by-frame behavior probabilities.
    Outputs a CSV of shape (T, len(behaviors)) with per-frame softmax probabilities.
    Returns the path to the output CSV, or None if there's insufficient length.
    """
    # Build output filename
    output_file = file_path.replace("_cls.h5", f"_{dataset_name}_outputs.csv")

    # Load embeddings (shape: (T, 768))
    try:
        with h5py.File(file_path, "r") as f:
            cls_np = np.array(f["cls"][:])
    except Exception as e:
        print(f"Error reading HDF5 file {file_path}: {e}")
        return None
    
    if cls_np.ndim < 2 or cls_np.shape[0] == 0:
        print(f"Warning: HDF5 file {file_path} has empty or malformed 'cls' dataset for inference.")
        return None


    # Center embeddings: subtract mean over time (T, 768) → (T, 768) in float16
    cls = torch.from_numpy(cls_np - np.mean(cls_np, axis=0)).half()

    predictions = []
    batch_windows = [] # Renamed for clarity

    # “cls” here is all the embeddings for this video file.
    # If this video has fewer than seq_len frames, there’s no valid seq_len‐sized window → skip it.
    if len(cls) < seq_len:
        print(f"Skipping {file_path}, length {len(cls)} is less than seq_len {seq_len}.")
        return None

    # Move model to correct device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    half_seqlen = seq_len // 2

    # Build sliding windows centered at each possible frame index
    for ind in range(half_seqlen, len(cls) - half_seqlen):
        # Window: frames [ind - half_seqlen, ..., ind + half_seqlen]
        window = cls[ind - half_seqlen : ind + half_seqlen + 1]
        batch_windows.append(window)  # window shape: (seq_len, 768)

        # Every 4096 windows (or at the end), push them through the classifier in one go
        if len(batch_windows) >= 4096 or ind == len(cls) - half_seqlen - 1:
            if not batch_windows: # Should not happen if loop condition is met
                continue
            batch_tensor = torch.stack(batch_windows)  # shape: (B', seq_len, 768)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
                # forward_nodrop → shape: (B', len(behaviors)), i.e. one logit vector per window
                logits = model.forward_nodrop(batch_tensor.to(device))
                probs = torch.softmax(logits, dim=1)
                predictions.extend(probs.detach().cpu().numpy())
            batch_windows = []

    if not predictions: # If no predictions were made (e.g. loop didn't run)
        print(f"No predictions generated for {file_path}, possibly due to windowing issues.")
        return None

    # Now we have a list of predictions, one per valid window (length = len(cls) - (seq_len - 1))
    # We need to re-expand to length T by "padding" the first/last values:
    total_predictions = []
    num_actual_predictions = len(predictions)

    for ind_frame in range(len(cls)):
        if ind_frame < half_seqlen:
            # For the first half_seqlen frames: use the first valid window's prediction
            total_predictions.append(predictions[0])
        elif ind_frame >= len(cls) - half_seqlen:
            # For the last half_seqlen frames: use the last valid window's prediction
            total_predictions.append(predictions[num_actual_predictions-1]) # Corrected to use num_actual_predictions
        else:
            # Otherwise: align to predictions list (offset by half_seqlen)
            total_predictions.append(predictions[ind_frame - half_seqlen])

    total_predictions_np = np.array(total_predictions)  # shape: (T, len(behaviors))

    # Save to CSV (no index column; columns = behavior names)
    dataframe = pd.DataFrame(total_predictions_np, columns=behaviors)
    dataframe.to_csv(output_file, index=False)

    return output_file


class DinoEncoder(nn.Module):
    """
    Wraps DINOv2 as a frozen feature encoder.
    Input: x (B, S, H, W) where S = number of frames in a segment
    Output: cls embeddings shape (B, S, 768)
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device) # Ensure it's a torch.device
        # Load pretrained DINOv2
        self.model = transformers.AutoModel.from_pretrained("facebook/dinov2-base").to(
            self.device
        )
        self.model.eval()

        # Freeze all parameters (no training of DINOv2 itself)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        B: number of batches; S: input elements per sample (sequence length); H: height, W: width
        
        We infer black and white videos but DINOv2 takes color. We add a color channel by adding a
        third dimension before H and W and then reshaping the tensor to copy along this dimension
        three times.
        
        x: (B, S, H, W) → convert to (B*S, 3, H, W), run through transformer,
        then reshape to (B, S, 768).
        """
        B, S, H, W = x.shape
        x = x.to(self.device)

        # Expand to 3 channels: (B, S, 1, H, W) → (B, S, 3, H, W) → (B*S, 3, H, W)
        x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1).reshape(B * S, 3, H, W)

        with torch.no_grad():
            out = self.model(x)  # out.last_hidden_state: (B*S, seq_len_patch, hidden_dim)

        # Take [CLS] token at index 0 → (B*S, 768), then reshape to (B, S, 768)
        cls = out.last_hidden_state[:, 0, :].reshape(B, S, 768)
        return cls


class Recording:
    """
    Represents a folder of video segments for one camera capture.
    - video_files: all .mp4 files in the folder
    - encoding_files: all existing .h5 (DINOv2 embeddings) in the folder
    - unencoded_files: those .mp4 files without a corresponding _cls.h5
    - classifications: maps model_name → list of CSV outputs for that model
    """

    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(path)

        self.path = path
        self.name = os.path.basename(self.path)

        # Gather all .mp4 files, sorted by numeric index in filename
        self.video_files = sorted(
            [f.path for f in os.scandir(self.path) if f.is_file() and f.path.endswith(".mp4")], # Check f.is_file()
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        # Gather all existing .h5 files
        self.encoding_files = [f.path for f in os.scandir(self.path) if f.is_file() and f.path.endswith("_cls.h5")] # More specific

        # Identify unencoded .mp4 files (no corresponding _cls.h5)
        self.unencoded_files = []
        for video_file in self.video_files:
            h5_file_path = video_file.replace(".mp4", "_cls.h5")
            if not os.path.exists(h5_file_path): # Simpler check
                self.unencoded_files.append(video_file)
            elif h5_file_path not in self.encoding_files : # Add if exists but not caught by scandir (rare)
                 self.encoding_files.append(h5_file_path)


        # Build classification_files mapping: model_name → list of CSV outputs
        classification_files = [f.path for f in os.scandir(self.path) if f.is_file() and f.path.endswith(".csv")]
        self.classifications = {}
        for file_path_csv in classification_files: # Renamed for clarity
            # The model name is the token just before "_outputs.csv"
            try:
                model_name = os.path.basename(file_path_csv).split("_")[-2]
                self.classifications.setdefault(model_name, []).append(file_path_csv)
            except IndexError:
                print(f"Warning: Could not parse model name from CSV file: {file_path_csv}")


        print(f"[Recording] {self.path}  → videos: {len(self.video_files)}, unencoded: {len(self.unencoded_files)}, classifications: {self.classifications.keys()}")

    def encode(self, encoder: nn.Module):
        """
        For each unencoded .mp4 in this recording, generate a _cls.h5 with encode_file.
        """
        # Iterate over a copy if modifying the list during iteration
        for file_to_encode in list(self.unencoded_files): 
            out_file_path = encode_file(encoder, file_to_encode)
            if out_file_path: # If encoding was successful
                if out_file_path not in self.encoding_files:
                    self.encoding_files.append(out_file_path)
                if file_to_encode in self.unencoded_files: # Re-check as it might have been processed by another thread
                    self.unencoded_files.remove(file_to_encode)


    def infer(
        self,
        model: nn.Module,
        dataset_name: str,
        behaviors: list[str],
        seq_len: int,
        device=None,
    ):
        """
        For each existing _cls.h5 in this recording, run infer_file → produce a CSV.
        """
        for file_path_h5 in self.encoding_files: # Renamed for clarity
            # Check if output already exists (optional, can be useful for large datasets)
            # output_csv_path = file_path_h5.replace("_cls.h5", f"_{dataset_name}_outputs.csv")
            # if os.path.exists(output_csv_path):
            #     print(f"Output CSV {output_csv_path} already exists. Skipping inference.")
            #     continue
            infer_file(file_path_h5, model, dataset_name, behaviors, seq_len, device=device)


class Camera:
    """
    Wraps camera configuration (RTSP, cropping, resolution, framerate).
    Responsible for starting/stopping ffmpeg-based video segment recording.
    """

    def __init__(self, config: dict[str, str], project: "Project"):
        self.config = config
        self.project = project

        self.recording = False

        self.name = config["name"]
        self.rtsp_url = config.get("rtsp_url", "") # Use .get for safety
        self.framerate = config.get("framerate", 10)
        self.resolution = config.get("resolution", 256)
        self.crop_left_x = config.get("crop_left_x", 0.0)
        self.crop_top_y = config.get("crop_top_y", 0.0)
        self.crop_width = config.get("crop_width", 1.0)
        self.crop_height = config.get("crop_height", 1.0)

        self.path = os.path.join(self.project.cameras_dir, self.name)

    def settings_to_dict(self) -> dict[str, str | int | float]: # Updated type hint
        return {
            "name": self.name, # Also include name
            "rtsp_url": self.rtsp_url,
            "framerate": self.framerate,
            "resolution": self.resolution,
            "crop_left_x": self.crop_left_x,
            "crop_top_y": self.crop_top_y,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
        }

    def update_settings(self, settings: dict[str, str | int | float]): # Updated type hint
        """
        Update camera parameters and rewrite config.yaml in the camera folder.
        """
        self.rtsp_url = str(settings.get("rtsp_url", self.rtsp_url))
        self.framerate = int(settings.get("framerate", self.framerate))
        self.resolution = int(settings.get("resolution", self.resolution))
        self.crop_left_x = float(settings.get("crop_left_x", self.crop_left_x))
        self.crop_top_y = float(settings.get("crop_top_y", self.crop_top_y))
        self.crop_width = float(settings.get("crop_width", self.crop_width))
        self.crop_height = float(settings.get("crop_height", self.crop_height))
        self.write_settings_to_config()

    def create_recording_dir(self) -> str | bool:
        """
        Create a subfolder for a new recording session:
        1) Under recordings/YYYYMMDD/
        2) Named as cameraName-HHMMSS-AM/PM
        Returns the new folder path, or False if it already existed.
        """
        date_dir = datetime.now().strftime("%Y%m%d")
        date_path = os.path.join(self.project.recordings_dir, date_dir)
        os.makedirs(date_path, exist_ok=True) # Use makedirs with exist_ok=True

        cam_dir = self.name + "-" + datetime.now().strftime("%I%M%S-%p")
        cam_path = os.path.join(date_path, cam_dir)
        if not os.path.exists(cam_path):
            os.makedirs(cam_path) # Use makedirs
            return cam_path
        print(f"Warning: Recording directory {cam_path} already exists.")
        return False

    def start_recording(self, destination: str, segment_time: int) -> bool:
        """
        Spawn an ffmpeg subprocess that connects to the RTSP, crops & scales the video,
        then segments into rolling .mp4 files every segment_time seconds (HLS style).
        """
        if self.name in self.project.active_recordings:
            print(f"Camera {self.name} is already recording.")
            return False
            
        if not os.path.exists(destination):
            print(f"Destination directory {destination} does not exist. Creating.")
            os.makedirs(destination, exist_ok=True)

        self.recording = True # Set recording status before starting process

        # Pattern: cameraName_00000.mp4, cameraName_00001.mp4, ...
        destination_pattern = os.path.join(destination, f"{self.name}_%05d.mp4")

        # Ensure numeric types for ffmpeg command
        framerate_val = int(self.framerate)
        resolution_val = int(self.resolution)
        crop_w_val = float(self.crop_width)
        crop_h_val = float(self.crop_height)
        crop_x_val = float(self.crop_left_x)
        crop_y_val = float(self.crop_top_y)


        command = [
            "ffmpeg",
            "-hide_banner", # Quieter output
            "-loglevel", "error", # Show only errors
            "-rtsp_transport", "tcp",
            "-i", str(self.rtsp_url),
            "-r", str(framerate_val),
            "-filter_complex",
            f"[0:v]crop=iw*{crop_w_val}:ih*{crop_h_val}:iw*{crop_x_val}:ih*{crop_y_val},scale={resolution_val}:{resolution_val}[cropped]",
            "-map", "[cropped]",
            "-c:v", "libx264", # Specify video codec
            "-preset", "fast", # Encoding preset
            "-f", "segment",
            "-segment_time", str(segment_time),
            "-segment_format_options", "movflags=+faststart", # For better web playback if needed
            "-reset_timestamps", "1",
            "-strftime", "1", # Enable strftime in segment filename (though not used here)
            "-sc_threshold", "0", # Disable scene change detection for segments
            "-force_key_frames", "expr:gte(t,n_forced*"+str(segment_time)+")", # Force keyframes for clean segments
            "-y", # Overwrite output files without asking
            destination_pattern,
        ]
        
        print(f"Starting FFMPEG for {self.name}: {' '.join(command)}")

        # Launch ffmpeg in its own process
        # Using shell=False is safer if command parts are fully controlled.
        # If shell=True is needed due to complex filter_complex, ensure inputs are sanitized.
        # For this structure, shell=False should be fine.
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.project.active_recordings[self.name] = process
            print(f"Recording started for {self.name} with PID {process.pid}")
            return True
        except Exception as e:
            print(f"Failed to start ffmpeg for {self.name}: {e}")
            self.recording = False
            return False


    def write_settings_to_config(self):
        """
        Persist current camera settings into config.yaml inside camera folder.
        """
        cam_settings = self.settings_to_dict() # Use the method
        with open(os.path.join(self.path, "config.yaml"), "w+") as file:
            yaml.dump(cam_settings, file, allow_unicode=True)

    def stop_recording(self) -> bool:
        """
        Stop ffmpeg subprocess for this camera by sending 'q' or terminating.
        """
        self.recording = False # Set status immediately
        active_recordings = self.project.active_recordings
        if self.name in active_recordings:
            process = active_recordings.pop(self.name) # Remove before trying to stop
            print(f"Stopping recording for {self.name} (PID: {process.pid})...")
            try:
                process.stdin.write(b'q\n') # Send 'q' to ffmpeg to gracefully stop
                process.stdin.flush()
                process.communicate(timeout=10) # Wait for graceful exit
                print(f"FFMPEG for {self.name} exited with code {process.returncode}")
            except subprocess.TimeoutExpired:
                print(f"FFMPEG for {self.name} did not respond to 'q', terminating...")
                process.terminate() # Force terminate if 'q' doesn't work
                try:
                    process.wait(timeout=5)
                    print(f"FFMPEG for {self.name} terminated.")
                except subprocess.TimeoutExpired:
                    print(f"FFMPEG for {self.name} did not terminate, killing...")
                    process.kill() # Force kill if terminate doesn't work
                    process.wait()
                    print(f"FFMPEG for {self.name} killed.")
            except Exception as e:
                 print(f"Error stopping FFMPEG for {self.name}: {e}. Attempting to terminate.")
                 process.terminate() # Fallback
            return True
        print(f"No active recording found for {self.name} to stop.")
        return False


class Model:
    """
    Wraps loading of a saved classification head.
    Expects a folder containing:
      - model.pth (state_dict for classifier)
      - config.yaml (seq_len, behaviors)
    """

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path) # Get name first
        self.config_path = os.path.join(path, "config.yaml")
        self.weights_path = os.path.join(path, "model.pth")
        

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Model config not found: {self.config_path}")
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")


class Dataset:
    """
    Wraps a dataset folder, which contains:
      - config.yaml (metadata: name, behaviors, whitelist, metrics, etc.)
      - labels.yaml (user-labeled behavior instances)
    Provides an update_metric() to modify metrics after training.
    """

    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path) # Add name attribute
        self.config_path = os.path.join(path, "config.yaml")
        self.labels_path = os.path.join(path, "labels.yaml")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Dataset config not found: {self.config_path}")
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        if not os.path.exists(self.labels_path):
            # Create a default empty labels file if it doesn't exist
            print(f"Labels file not found for dataset {self.name} at {self.labels_path}. Creating default.")
            default_behaviors = self.config.get("behaviors", [])
            default_labels = {"behaviors": default_behaviors, "labels": {b: [] for b in default_behaviors}}
            with open(self.labels_path, "w") as f_label:
                yaml.dump(default_labels, f_label, allow_unicode=True)
            self.labels = default_labels
        else:
            with open(self.labels_path) as f:
                self.labels = yaml.safe_load(f)

    def update_metric(self, behavior: str, group: str, value):
        """
        Update one metric (e.g., 'F1 Score', 'Recall', 'Precision') for a given behavior.
        """
        # Ensure metrics structure exists
        if "metrics" not in self.config:
            self.config["metrics"] = {}
        if behavior not in self.config["metrics"]:
            self.config["metrics"][behavior] = {}
            
        self.config["metrics"][behavior][group] = value
        # Write back to the file
        with open(self.config_path, "w") as file: # Use "w" not "w+" for overwriting
            yaml.dump(self.config, file, allow_unicode=True)


def remove_leading_zeros(num: str) -> int:
    """
    Utility: convert a string like '00023' → integer 23.
    """
    for i in range(len(num)):
        if num[i] != "0":
            return int(num[i:])
    return 0


class Actogram:
    """
    Builds an actogram image (time-series raster) for a particular behavior + model.
    """
    def __init__(
        self,
        directory: str, model: str, behavior: str,
        framerate: float, start: float, binsize_minutes: int,
        threshold: float, lightcycle: str, plot_acrophase: bool = False
    ):
        # --- 1. Store all parameters ---
        self.directory = directory
        self.model = model
        self.behavior = behavior
        self.framerate = float(framerate)
        self.start_hour_on_plot = float(start)
        self.threshold = float(threshold)
        self.bin_size_minutes = int(binsize_minutes)
        self.plot_acrophase = plot_acrophase
        if lightcycle == "LL": self.lightcycle_str = "1" * 24
        elif lightcycle == "DD": self.lightcycle_str = "0" * 24
        else: self.lightcycle_str = "1" * 12 + "0" * 12

        self.blob = None
        self.file_path_png = None
        
        if self.framerate <= 0 or self.bin_size_minutes <= 0:
            print("Error: Framerate and Bin Size must be positive.")
            return
            
        self.binsize_frames = int(self.bin_size_minutes * self.framerate * 60)
        if self.binsize_frames <= 0: return

        # --- 2. Calculate the raw per-frame activity ---
        activity_per_frame = []
        try:
            all_files_in_dir = os.listdir(self.directory)
            valid_csv_files = [fname for fname in all_files_in_dir if fname.endswith(f"_{self.model}_outputs.csv")]
            
            if not valid_csv_files:
                print(f"Actogram not generated: No CSV files found for model '{self.model}'.")
                return

            parsed_files = []
            for fname in valid_csv_files:
                try:
                    parts = fname.split('_')
                    if len(parts) >= 3:
                        segment_idx = remove_leading_zeros(parts[-3])
                        parsed_files.append((os.path.join(self.directory, fname), segment_idx))
                except (ValueError, IndexError):
                    continue
            
            if not parsed_files: return
            parsed_files.sort(key=lambda pair: pair[1])

            for file_path, seg_idx in parsed_files:
                try:
                    df = pd.read_csv(file_path)
                    if df.empty or self.behavior not in df.columns: continue
                    
                    prob_columns = [c for c in df.columns if c not in ['frame', 'unnamed: 0', 'index']]
                    if not prob_columns: continue

                    current_col_behavior_index_in_probs = prob_columns.index(self.behavior)
                    arr_probs = df[prob_columns].to_numpy()
                    if arr_probs.size == 0: continue

                    top_one_hot = (np.argmax(arr_probs, axis=1) == current_col_behavior_index_in_probs)
                    max_vals_probs = df[self.behavior].to_numpy()
                    values_for_thresholding = top_one_hot * max_vals_probs
                    frames_activity = (values_for_thresholding >= self.threshold).astype(float)
                    activity_per_frame.extend(frames_activity.tolist())
                except Exception as e:
                    print(f"Error processing CSV {file_path}: {e}")
                    continue
        except FileNotFoundError:
            print(f"Actogram not generated: Directory not found '{self.directory}'.")
            return
        
        if not activity_per_frame:
            print(f"Actogram not generated for {model} - {behavior} due to empty timeseries.")
            return

        # --- 3. Bin the activity data ---
        binned_activity = []
        for i in range(0, len(activity_per_frame), self.binsize_frames):
            binned_activity.append(np.sum(activity_per_frame[i:i + self.binsize_frames]))
        
        if not binned_activity:
            print(f"Actogram not generated for {model} - {behavior} due to empty binned data.")
            return
        
        # --- 4. Call the plotting function with the prepared data ---
        self.draw(binned_activity)


    def draw(self, binned_activity):
        """
        Gathers parameters and calls the main plotting function to generate the actogram.
        """
        light_cycle_booleans = [c == "1" for c in self.lightcycle_str]
        plot_title = f"{self.model} - {self.behavior}"

        fig = _create_matplotlib_actogram(
            binned_activity=binned_activity,
            light_cycle_booleans=light_cycle_booleans,
            tau=24.0,
            bin_size_minutes=self.bin_size_minutes,
            plot_title=plot_title,
            start_hour_offset=self.start_hour_on_plot,
            plot_acrophase=self.plot_acrophase
        )

        if fig is None:
            print("Matplotlib figure generation failed.")
            self.blob = None
            return

        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#343a40')
            buf.seek(0)
            self.blob = base64.b64encode(buf.read()).decode('utf-8')
            
            self.file_path_png = os.path.join(self.directory, f"{self.model}-{self.behavior}-actogram.png")
            with open(self.file_path_png, "wb") as f:
                buf.seek(0)
                f.write(buf.read())
            print(f"Generated Matplotlib Actogram PNG: {self.file_path_png}")

        except Exception as e:
            print(f"Error encoding matplotlib actogram to blob: {e}")
            self.blob = None
        finally:
            if 'buf' in locals():
                buf.close()
            plt.close('all')
            
# This function should be OUTSIDE the class, at the top level of the file.
def _create_matplotlib_actogram(
    binned_activity,
    light_cycle_booleans,
    tau,
    bin_size_minutes,
    plot_title,
    start_hour_offset,
    plot_acrophase=False
):
    """
    Generates a double-plotted actogram using Matplotlib, styled as a heatmap/raster plot.
    This function contains all data preparation and calculation logic.
    """
    bins_per_period = int((tau * 60) / bin_size_minutes)
    if bins_per_period == 0: return None

    # 1. Pad the raw binned activity based on the start time offset
    padding_bins = int(start_hour_offset * 60 / bin_size_minutes)
    padded_activity = np.pad(binned_activity, (padding_bins, 0), 'constant', constant_values=0)

    # 2. Reshape the padded data into a 2D array of days, padding the end with NaNs
    num_days = math.ceil(len(padded_activity) / bins_per_period)
    if num_days < 1: return None
    num_rows_to_plot = num_days
    
    required_len = num_days * bins_per_period
    padded_for_reshape = np.pad(padded_activity, (0, required_len - len(padded_activity)), 'constant', constant_values=np.nan)
    daily_data = padded_for_reshape.reshape(num_days, bins_per_period)

    # 3. Calculate Acrophase on the final, shaped daily_data array
    acrophase_points = []
    if plot_acrophase:
        t = np.linspace(0, 2 * np.pi, bins_per_period, endpoint=False)
        for day_idx, day_activity_with_nan in enumerate(daily_data):
            # Skip partial days (which contain NaNs) or days with no activity
            if np.isnan(day_activity_with_nan).any() or np.sum(np.nan_to_num(day_activity_with_nan)) == 0:
                continue

            day_activity = np.nan_to_num(day_activity_with_nan)
            
            sum_y_sin_t = np.sum(day_activity * np.sin(t))
            sum_y_cos_t = np.sum(day_activity * np.cos(t))
            phase_rad = math.atan2(sum_y_sin_t, sum_y_cos_t)
            
            acrophase_hour_relative_to_start = (phase_rad / (2 * np.pi)) * 24
            if acrophase_hour_relative_to_start < 0:
                acrophase_hour_relative_to_start += 24
        
            absolute_acrophase_hour = (acrophase_hour_relative_to_start + start_hour_offset) % 24
            acrophase_points.append((day_idx, absolute_acrophase_hour))
    
    # 4. Create the double-plotted array for visualization
    right_half = np.full_like(daily_data, np.nan)
    if num_days > 1:
        right_half[:-1, :] = daily_data[1:, :]
    double_plotted_events = np.concatenate([daily_data, right_half], axis=1)

    # --- The rest of the function is for plotting and is unchanged ---
    
    # Setup Background Colormap
    # Define all our potential colors
    light_yellow = '#FEFDE3'
    dark_yellow = '#E8D570' 
    light_grey = '#D3D3D3'
    dark_grey = '#A9A9A9'

    # Determine which pattern and colors to use
    if all(light_cycle_booleans): # LL condition
        # Subjective day/night with shades of yellow
        daily_light_pattern = np.array([1]*int(12*60/bin_size_minutes) + [0]*int(12*60/bin_size_minutes))
        light_cmap = LinearSegmentedColormap.from_list("light_cmap", [dark_yellow, light_yellow])
    elif not any(light_cycle_booleans): # DD condition
        # Subjective day/night with shades of grey
        daily_light_pattern = np.array([1]*int(12*60/bin_size_minutes) + [0]*int(12*60/bin_size_minutes))
        light_cmap = LinearSegmentedColormap.from_list("light_cmap", [dark_grey, light_grey])
    else: # LD condition
        # Actual day/night based on the light cycle string
        hourly_light_dark = [light_cycle_booleans[i % len(light_cycle_booleans)] for i in range(24)]
        daily_light_pattern = np.repeat(hourly_light_dark, (60 / bin_size_minutes))
        light_cmap = LinearSegmentedColormap.from_list("light_cmap", [dark_grey, light_yellow])

    # Create the final background array using the selected pattern
    double_plotted_light = np.array([np.concatenate([daily_light_pattern, daily_light_pattern]) for _ in range(num_rows_to_plot)])
    # --- END OF FINAL, CORRECT BACKGROUND SETUP ---

    # Setup Activity Colormap
    cmap_viridis = plt.cm.get_cmap('viridis')
    activity_colors = cmap_viridis(np.arange(cmap_viridis.N))
    activity_colors[0, 3] = 0
    transparent_viridis = LinearSegmentedColormap.from_list('transparent_viridis', activity_colors)
    transparent_viridis.set_bad(color=(0.0, 0.0, 0.0, 0.0))

    # Setup Figure
    fig_height = max(4, num_rows_to_plot * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=120)
    fig.patch.set_facecolor('#343a40')
    ax.set_facecolor('#343a40')
    
    plot_extent = [0, 2 * tau, num_rows_to_plot, 0]
    
    # Plotting
    ax.imshow(double_plotted_light, aspect='auto', cmap=light_cmap, interpolation='none', extent=plot_extent, vmin=0, vmax=1)
    
    non_zero_activity = [v for v in binned_activity if v > 0]
    vmax = np.percentile(non_zero_activity, 90) + 1e-6 if non_zero_activity else 1
    
    cax = ax.imshow(double_plotted_events, aspect='auto', cmap=transparent_viridis, interpolation='none', extent=plot_extent, vmin=0, vmax=vmax)
    
    if acrophase_points:
        for day_idx, acrophase_hour in acrophase_points:
            ax.plot(acrophase_hour, day_idx + 0.5, 'o', color='red', markersize=8, markeredgecolor='black')
            ax.plot(acrophase_hour + tau, day_idx + 0.5, 'o', color='red', markersize=8, markeredgecolor='black')

    # Styling and Colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Event Count', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(plot_title, color='white', pad=20)
    ax.set_xlabel('Time of Day (Double Plotted)', color='white')
    ax.set_ylabel('Day', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    ax.set_xlim(0, 2*tau)
    ax.set_xticks(np.arange(0, 2 * tau + 1, 4))
    ax.set_xticklabels([f"{int(tick % 24):02d}" for tick in np.arange(0, 2 * tau + 1, 4)])

    ax.set_ylim(num_rows_to_plot, 0)
    ax.set_yticks(np.arange(0.5, num_rows_to_plot, 1))
    ax.set_yticklabels([f"{i+1}" for i in range(num_rows_to_plot)])
    
    fig.tight_layout()
    return fig


class InvalidProject(Exception):
    def __init__(self, path):
        super().__init__(f"{path} is not a valid project")

class StandardDataset(torch.utils.data.Dataset):
    """A standard PyTorch dataset that does not perform balancing."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class BalancedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that implements class balancing for batch creation.
    It ensures that when iterated through (via a DataLoader), samples from
    each behavior class are drawn in a roughly balanced manner.
    """
    def __init__(self, sequences: list[torch.Tensor], labels: list[torch.Tensor], behaviors_list: list[str]):
        """
        Args:
            sequences (list[torch.Tensor]): A flat list of all sequence tensors.
            labels (list[torch.Tensor]): A flat list of corresponding label tensors (scalar, integer indices).
            behaviors_list (list[str]): The canonical list of behavior names, where the order
                                       corresponds to the integer label indices.
        """
        self.behaviors_list = behaviors_list
        self.num_behaviors = len(self.behaviors_list)
        
        self.buckets = {b_name: [] for b_name in self.behaviors_list}
        
        # Populate buckets
        for seq_tensor, label_tensor in zip(sequences, labels):
            label_index = label_tensor.item() # Assumes label_tensor is a scalar tensor
            if 0 <= label_index < self.num_behaviors:
                behavior_name = self.behaviors_list[label_index]
                self.buckets[behavior_name].append(seq_tensor)
            else:
                # This case should ideally not happen if data preparation is correct
                print(f"Warning: Label index {label_index} is out of bounds for behaviors list "
                      f"(len={self.num_behaviors}). Skipping this sample during BalancedDataset init.")

        self.internal_counter = 0

        # Calculate the original total number of sequences across all valid buckets
        self.original_total_sequences = 0
        for behavior_name in self.behaviors_list:
            # Ensure all behaviors in behaviors_list are keys in buckets, even if list is empty
            if behavior_name not in self.buckets: # Should have been created above
                 self.buckets[behavior_name] = []
            self.original_total_sequences += len(self.buckets[behavior_name])

        # Warn about empty behavior buckets as they will cause issues during __getitem__
        for behavior_name in self.behaviors_list:
            if not self.buckets[behavior_name]:
                print(f"Warning: Behavior '{behavior_name}' has 0 samples after bucketing. "
                      "It cannot be sampled by BalancedDataset, which may lead to errors if it's chosen by the cyclic counter.")

    def __len__(self):
        if self.num_behaviors == 0 or self.original_total_sequences == 0:
            return 0
        
        # Adjust length to be a multiple of num_behaviors.
        # This ensures that over one "epoch" (defined by this length), the cyclic counter
        # in __getitem__ will attempt to sample from each behavior class an equal number of times.
        # The modulo operation at the end handles the case where original_total_sequences is already a multiple.
        return self.original_total_sequences + \
               (self.num_behaviors - self.original_total_sequences % self.num_behaviors) % self.num_behaviors

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): The index provided by the DataLoader's sampler. This index is used
                       to select a sample *within* the cyclically chosen behavior's bucket.
        """
        if self.num_behaviors == 0:
            raise IndexError("Cannot get item from BalancedDataset with zero behaviors defined.")

        # Determine which behavior to sample from based on the internal_counter (cyclic)
        # This ensures that successive calls (as DataLoader would make for a batch)
        # cycle through the behavior classes.
        behavior_to_sample_idx = self.internal_counter % self.num_behaviors
        self.internal_counter = (self.internal_counter + 1) # No modulo here, let __len__ define epoch boundary

        chosen_behavior_name = self.behaviors_list[behavior_to_sample_idx]
        bucket_for_chosen_behavior = self.buckets[chosen_behavior_name]

        if not bucket_for_chosen_behavior:
            # This is a critical error if reached, means a behavior class selected for sampling has no samples.
            # Should be caught by warnings in __init__, but good to have a runtime check.
            raise IndexError(f"Critical: Attempted to sample from behavior '{chosen_behavior_name}', "
                             "which has no samples. Check data and BalancedDataset initialization.")

        # Use the DataLoader's idx to pick a sample from this behavior's bucket (with wrap-around/modulo)
        # This allows oversampling of minority classes if their buckets are smaller than others.
        sample_idx_within_bucket = idx % len(bucket_for_chosen_behavior)
        
        sequence_sample = bucket_for_chosen_behavior[sample_idx_within_bucket]
        label_for_sample = torch.tensor(behavior_to_sample_idx).long() # Label is the index in self.behaviors_list

        return sequence_sample, label_for_sample

class Project:
    """
    Top-level project structure:
      - cameras/      (each camera has its own folder & config)
      - recordings/   (date folders → camera session folders → .mp4 & .h5 & .csv)
      - models/       (each trained classifier head has a folder)
      - data_sets/    (each dataset has config.yaml & labels.yaml)
    Automatically builds:
      - self.recordings: dict[date_str] -> dict[cameraName] -> Recording
      - self.cameras: dict[cameraName] -> Camera
      - self.models: dict[modelName] -> Model
      - self.datasets: dict[datasetName] -> Dataset
    """

    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise InvalidProject(path)
        self.path = path

        self.cameras_dir = os.path.join(self.path, "cameras")
        self.recordings_dir = os.path.join(self.path, "recordings")
        self.models_dir = os.path.join(self.path, "models")
        self.datasets_dir = os.path.join(self.path, "data_sets")

        for subdir in [self.cameras_dir, self.recordings_dir, self.models_dir, self.datasets_dir]:
            if not os.path.isdir(subdir):
                os.makedirs(subdir, exist_ok=True) # Create if missing
                print(f"Created missing subdirectory: {subdir}")
                # raise InvalidProject(subdir) # Original behavior

        self.active_recordings: dict[str, subprocess.Popen] = {}

        # Build out recordings by date → cameraName → Recording
        self.recordings: dict[str, dict[str, Recording]] = {}
        if os.path.exists(self.recordings_dir): # Check if recordings_dir exists
            days = [f.path for f in os.scandir(self.recordings_dir) if f.is_dir()]
            for day in days:
                day_str = os.path.basename(day)
                self.recordings[day_str] = {}
                # Each subfolder is a camera session
                subfolders = [f.path for f in os.scandir(day) if f.is_dir()]
                for folder in subfolders:
                    try:
                        rec = Recording(folder)
                        self.recordings[day_str][rec.name] = rec
                    except Exception as e:
                        print(f"Error initializing Recording from folder {folder}: {e}")


        # Build out cameras: name → Camera instance
        self.cameras = {}
        if os.path.exists(self.cameras_dir): # Check if cameras_dir exists
            for cam_path_entry in os.scandir(self.cameras_dir):
                if cam_path_entry.is_dir():
                    cam_path = cam_path_entry.path
                    config_file = os.path.join(cam_path, "config.yaml")
                    if os.path.exists(config_file):
                        try:
                            with open(config_file) as file:
                                config_data = yaml.safe_load(file)
                            if "name" in config_data: # Ensure name is in config
                                cam = Camera(config_data, self)
                                self.cameras[config_data["name"]] = cam
                            else:
                                print(f"Warning: 'name' not found in config {config_file}. Skipping camera.")
                        except Exception as e:
                             print(f"Error loading camera config {config_file}: {e}")
                    else:
                        print(f"Warning: config.yaml not found in camera directory {cam_path}")


        # Build out models: each folder in models/ → Model
        self.models = {}
        if os.path.exists(self.models_dir): # Check if models_dir exists
            for model_path_entry in os.scandir(self.models_dir):
                if model_path_entry.is_dir():
                    model_path = model_path_entry.path
                    try:
                        self.models[os.path.basename(model_path)] = Model(model_path)
                    except Exception as e:
                        print(f"Error loading model from {model_path}: {e}")

        # Include built-in "JonesLabModel" if present under "./models/JonesLabModel" relative to script
        # This might need adjustment based on actual deployment structure
        # For robustness, consider making this path configurable or checking relative to project.path
        script_dir_models_path = os.path.join(os.path.dirname(__file__), "models", "JonesLabModel")
        project_models_path = os.path.join(self.models_dir, "JonesLabModel")

        jlpath_to_check = None
        if os.path.isdir(project_models_path):
            jlpath_to_check = project_models_path
        elif os.path.isdir(script_dir_models_path) and "JonesLabModel" not in self.models: # Check if already loaded from project
             jlpath_to_check = script_dir_models_path
        
        if jlpath_to_check:
            try:
                self.models["JonesLabModel"] = Model(jlpath_to_check)
                print(f"Loaded JonesLabModel from {jlpath_to_check}")
            except Exception as e:
                print(f"Could not load JonesLabModel from {jlpath_to_check}: {e}")


        # Build out datasets: each folder in data_sets/ → Dataset
        self.datasets = {}
        if os.path.exists(self.datasets_dir): # Check if datasets_dir exists
            for ds_path_entry in os.scandir(self.datasets_dir):
                if ds_path_entry.is_dir():
                    ds_path = ds_path_entry.path
                    ds_name = os.path.basename(ds_path)
                    try:
                        self.datasets[ds_name] = Dataset(ds_path)
                    except Exception as e:
                        print(f"Error loading dataset {ds_name} from {ds_path}: {e}")


    @staticmethod
    def create_project(parent_directory: str, project_name: str) -> "Project | None": # Updated type hint
        """
        Create a new project folder structure under parent_directory/project_name
        with required subfolders, then return a Project instance.
        """
        project_path = os.path.join(parent_directory, project_name)
        if os.path.exists(project_path):
            print(f"Project '{project_name}' already exists at {parent_directory}. Please choose a different name or location.")
            return None

        # Make main directories
        try:
            os.makedirs(project_path, exist_ok=True) # Create parent first
            for sub in ["cameras", "recordings", "models", "data_sets"]:
                os.makedirs(os.path.join(project_path, sub), exist_ok=True)
        except OSError as e:
            print(f"Error creating project directories for {project_name}: {e}")
            return None


        project = Project(project_path) # Initialize after directories are made
        print(f"Project '{project_name}' creation successful at {project_path}")
        return project

    def encode_recordings(self):
        """
        Loop over all Recording objects and run .encode() to generate missing _cls.h5.
        """
        encoder = DinoEncoder()  # load one DINOv2 encoder for all
        for day_recordings in self.recordings.values(): # Iterate through dict values
            for rec in day_recordings.values(): # Iterate through inner dict values
                rec.encode(encoder)

    def infer_recordings(self, device_str: str | None = None): # Changed device to device_str for clarity
        """
        Loop over all Recording objects and run .infer() to generate CSV outputs using
        the built-in JonesLabModel (if present).
        """
        if "JonesLabModel" not in self.models:
            print("JonesLabModel not found. Cannot run default inference.")
            return

        jones_model_obj = self.models["JonesLabModel"]
        model_config = jones_model_obj.config
        
        if device_str is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)


        try:
            # It's safer to load the state_dict and instantiate the model
            state_dict = torch.load(jones_model_obj.weights_path, map_location=device)
            cls_head = classifier_head.classifier(
                in_features=768, # Standard DINOv2 feature size
                out_features=len(model_config["behaviors"]),
                seq_len=model_config["seq_len"],
            ).to(device)
            cls_head.load_state_dict(state_dict)
            cls_head.eval()
        except Exception as e:
            print(f"Error loading JonesLabModel: {e}")
            return


        for day_recordings in self.recordings.values():
            for rec in day_recordings.values():
                print(f"Inferring on recording: {rec.name} using JonesLabModel")
                rec.infer(cls_head, "JonesLabModel", model_config["behaviors"], model_config["seq_len"], device=device)

    def create_camera(self, name: str, settings: dict[str, str | int | float]) -> Camera | None: # Updated type hint
        """
        Create a new camera folder under cameras/, write its config.yaml, and return the Camera.
        """
        camera_path = os.path.join(self.cameras_dir, name)
        if os.path.exists(camera_path):
            print(f"Camera '{name}' already exists at {camera_path}.")
            return None # Or return existing self.cameras.get(name)

        os.makedirs(camera_path, exist_ok=True) # Create dir if not exists
        
        # Ensure 'name' is in settings and matches
        settings_with_name = settings.copy() # Avoid modifying input dict
        settings_with_name["name"] = name 

        with open(os.path.join(camera_path, "config.yaml"), "w+") as file:
            yaml.dump(settings_with_name, file, allow_unicode=True)

        cam = Camera(settings_with_name, self)
        self.cameras[name] = cam
        return cam

    def remove_camera(self, camera_name: str):
        """
        Delete a camera folder and remove from self.cameras.
        """
        if camera_name in self.cameras:
            cam = self.cameras[camera_name]
            try:
                shutil.rmtree(cam.path)
                print(f"Removed camera directory: {cam.path}")
            except OSError as e:
                print(f"Error removing camera directory {cam.path}: {e}")
            del self.cameras[camera_name]
        else:
            print(f"Camera '{camera_name}' not found in project.")

    def convert_instances(
        self, insts: list[dict], seq_len: int, behaviors: list[str],
        progress_callback=None  # <<< THIS IS THE KEY FIX
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Given a list of labeled instances, build training sequences.
        Returns flat lists of sequences and their corresponding labels (as tensors).
        The sequences and labels are globally shuffled.
        """
        seqs = []
        labels = []
        half_seqlen = seq_len // 2

        total_instances = len(insts)  # Get total number for progress calculation

        for inst_idx, inst in enumerate(insts):
            # --- Report progress back to the TrainingThread ---
            if progress_callback and total_instances > 0:
                if inst_idx % 5 == 0 or inst_idx == total_instances - 1:
                    percent_done = (inst_idx + 1) / total_instances * 100
                    progress_callback(percent_done)
            # ---

            if not all(k in inst for k in ["start", "end", "video", "label"]):
                print(f"Warning: Instance {inst_idx} is malformed. Data: {inst}. Skipping.")
                continue

            start_val = inst.get("start")
            end_val = inst.get("end")

            if start_val is None or end_val is None:
                print(f"Warning: Instance {inst_idx} has null start/end. Data: {inst}. Skipping.")
                continue
            
            start_frame = int(start_val)
            end_frame = int(end_val)
            video_path = inst["video"]
            behavior_label_name = inst["label"]
            
            cls_path = video_path.replace(".mp4", "_cls.h5")
            if not os.path.exists(cls_path):
                # This is a common warning, can be noisy, but good for debugging.
                # print(f"Warning: HDF5 file not found for instance video {video_path}. Skipping.")
                continue

            try:
                with h5py.File(cls_path, "r") as cls_file:
                    cls_arr = cls_file["cls"][:]
            except Exception as e:
                print(f"Warning: Could not load HDF5 file {cls_path}. Skipping. Error: {e}")
                continue 
            
            if cls_arr.ndim < 2 or cls_arr.shape[0] == 0 or cls_arr.shape[1] != 768:
                print(f"Warning: HDF5 file {cls_path} has malformed 'cls' dataset. Skipping.")
                continue

            video_mean = np.mean(cls_arr, axis=0) 
            cls_arr_centered = cls_arr - video_mean

            T = cls_arr_centered.shape[0]
            if T < seq_len:
                continue

            valid_start = max(half_seqlen, start_frame)
            valid_end = min(T - half_seqlen - 1, end_frame)

            if valid_start > valid_end:
                continue

            for ind in range(valid_start, valid_end + 1):
                window = cls_arr_centered[ind - half_seqlen : ind + half_seqlen + 1]
                if window.shape[0] != seq_len:
                    continue
                seqs.append(torch.from_numpy(window).half())
                try:
                    label_idx = behaviors.index(behavior_label_name)
                    labels.append(torch.tensor(label_idx).long())
                except ValueError:
                    if seqs: seqs.pop() # Remove the corresponding sequence if label is bad

        if not seqs: 
            return [], []

        paired = list(zip(seqs, labels))
        random.shuffle(paired)
        shuffled_seqs, shuffled_labels = zip(*paired)
        
        return list(shuffled_seqs), list(shuffled_labels)

    def create_dataset(self, name: str, behaviors: list[str], recordings_whitelist: list[str]) -> Dataset | None: # Renamed arg
        """
        Create a new dataset folder under data_sets/, initialize config & labels YAML.
        'recordings_whitelist' should be paths relative to the project's recordings_dir,
        e.g., ["YYYYMMDD/Camera1-HHMMSS-AMPM", "YYYYMMDD/Camera2-HHMMSS-AMPM"]
        or just ["YYYYMMDD"] to include all sessions from that day.
        """
        directory = os.path.join(self.datasets_dir, name)
        if os.path.exists(directory):
            print(f"Dataset '{name}' already exists at {directory}.")
            return self.datasets.get(name) # Return existing if found

        os.makedirs(directory, exist_ok=True)

        dataset_config_path = os.path.join(directory, "config.yaml") # Renamed for clarity
        label_file_path = os.path.join(directory, "labels.yaml")     # Renamed for clarity

        # Whitelist entries are now expected to be cleaner, like "YYYYMMDD/CameraName-SessionID"
        # The old r"\\" suffix was Windows-specific and might not be robust.
        # The Project class usually stores full paths or paths relative to a base.
        # For dataset whitelists, it's often better to store partial paths that can be matched.
        metrics = {b: {"Train #": 0, "Test #": 0, "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A"} for b in behaviors}

        dconfig = {
            "name": name,
            "behaviors": behaviors,
            "whitelist": recordings_whitelist, # Store the provided whitelist directly
            "model": None, # Path to the trained model associated with this dataset
            "metrics": metrics,
        }
        # Default empty labels structure
        labelconfig = {"behaviors": behaviors, "labels": {b: [] for b in behaviors}}

        with open(dataset_config_path, "w") as file:
            yaml.dump(dconfig, file, allow_unicode=True)
        with open(label_file_path, "w") as file:
            yaml.dump(labelconfig, file, allow_unicode=True)

        ds = Dataset(directory)
        self.datasets[name] = ds
        return ds

    def load_dataset(
        self, name: str, seed: int = 42, split: float = 0.2, seq_len: int = 15,
        progress_callback=None
    ) -> tuple[BalancedDataset | None, BalancedDataset | None]:
        dataset_path = os.path.join(self.datasets_dir, name)
        if seq_len % 2 == 0:
            seq_len += 1 
        if not os.path.isdir(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            raise FileNotFoundError(dataset_path)

        config_file = os.path.join(dataset_path, "config.yaml")
        labels_file = os.path.join(dataset_path, "labels.yaml")

        if not os.path.exists(config_file) or not os.path.exists(labels_file):
            print(f"Config or labels file missing in {dataset_path}")
            return None, None

        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)
        with open(labels_file, "r") as file:
            label_config_data = yaml.safe_load(file)

        behaviors_list = label_config_data.get("behaviors", [])
        if not behaviors_list:
            print(f"No behaviors defined in labels file for dataset {name}.")
            return None, None
            
        all_labeled_instances = []
        for beh_name in behaviors_list:
            instances_for_beh = label_config_data.get("labels", {}).get(beh_name, [])
            for inst in instances_for_beh:
                if 'label' not in inst or inst['label'] != beh_name:
                    inst['label'] = beh_name
                all_labeled_instances.append(inst)
        
        if not all_labeled_instances:
            print(f"No labeled instances found in labels.yaml for dataset {name}.")
            return BalancedDataset([], [], behaviors_list), BalancedDataset([], [], behaviors_list)

        random.seed(seed)
        random.shuffle(all_labeled_instances)

        # <<< START OF LIKELY MISSING LOGIC >>>
        # This block groups instances by video to ensure a clean train/test split.
        inst_groups: dict[str, list[dict]] = {} 
        for inst in all_labeled_instances:
            if 'video' not in inst or not inst['video']: continue
            try:
                video_segment_dir = os.path.dirname(inst["video"])
                group_key = video_segment_dir 
            except Exception:
                group_key = inst["video"]
            inst_groups.setdefault(group_key, []).append(inst)
        
        video_group_keys = list(inst_groups.keys())
        random.shuffle(video_group_keys)

        num_groups = len(video_group_keys)
        split_idx_group = int((1 - split) * num_groups)
        
        train_groups_keys = video_group_keys[:split_idx_group]
        test_groups_keys = video_group_keys[split_idx_group:]

        # These two lines define the variables that were missing.
        train_insts_flat_list = [inst for key in train_groups_keys for inst in inst_groups[key]]
        test_insts_flat_list = [inst for key in test_groups_keys for inst in inst_groups[key]]
            
        random.shuffle(train_insts_flat_list)
        random.shuffle(test_insts_flat_list)
        # <<< END OF LIKELY MISSING LOGIC >>>

        def train_progress_updater(percent):
            if progress_callback: progress_callback(percent * 0.5)

        def test_progress_updater(percent):
            if progress_callback: progress_callback(50.0 + (percent * 0.5))

        train_seqs_flat, train_labels_flat = self.convert_instances(train_insts_flat_list, seq_len, behaviors_list, progress_callback=train_progress_updater)
        test_seqs_flat, test_labels_flat = self.convert_instances(test_insts_flat_list, seq_len, behaviors_list, progress_callback=test_progress_updater)
        
        train_balanced_dataset = BalancedDataset(train_seqs_flat, train_labels_flat, behaviors_list)
        test_balanced_dataset = BalancedDataset(test_seqs_flat, test_labels_flat, behaviors_list)
        
        return train_balanced_dataset, test_balanced_dataset
        
    def load_dataset_for_weighted_loss(
        self, name: str, seed: int = 42, split: float = 0.2, seq_len: int = 15,
        progress_callback=None
    ):
        dataset_path = os.path.join(self.datasets_dir, name)
        if seq_len % 2 == 0:
            seq_len += 1 
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(dataset_path)

        config_file = os.path.join(dataset_path, "config.yaml")
        labels_file = os.path.join(dataset_path, "labels.yaml")

        if not os.path.exists(config_file) or not os.path.exists(labels_file):
            return None, None, None

        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)
        with open(labels_file, "r") as file:
            label_config_data = yaml.safe_load(file)

        behaviors_list = label_config_data.get("behaviors", [])
        if not behaviors_list:
            return None, None, None
            
        all_labeled_instances = []
        for beh_name in behaviors_list:
            instances_for_beh = label_config_data.get("labels", {}).get(beh_name, [])
            for inst in instances_for_beh:
                inst['label'] = beh_name
                all_labeled_instances.append(inst)
        
        if not all_labeled_instances:
            return None, None, None

        random.seed(seed)
        random.shuffle(all_labeled_instances)
        
        # <<< START OF LIKELY MISSING LOGIC >>>
        inst_groups: dict[str, list[dict]] = {} 
        for inst in all_labeled_instances:
            if 'video' not in inst or not inst['video']: continue
            try:
                video_segment_dir = os.path.dirname(inst["video"])
                group_key = video_segment_dir 
            except Exception:
                group_key = inst["video"]
            inst_groups.setdefault(group_key, []).append(inst)
        
        video_group_keys = list(inst_groups.keys())
        random.shuffle(video_group_keys)

        num_groups = len(video_group_keys)
        split_idx_group = int((1 - split) * num_groups)
        
        train_groups_keys = video_group_keys[:split_idx_group]
        test_groups_keys = video_group_keys[split_idx_group:]

        train_insts_flat_list = [inst for key in train_groups_keys for inst in inst_groups[key]]
        test_insts_flat_list = [inst for key in test_groups_keys for inst in inst_groups[key]]
            
        random.shuffle(train_insts_flat_list)
        random.shuffle(test_insts_flat_list)
        # <<< END OF LIKELY MISSING LOGIC >>>
        
        def train_progress_updater(p):
            if progress_callback: progress_callback(p * 0.5)
        def test_progress_updater(p):
            if progress_callback: progress_callback(50.0 + p * 0.5)

        train_seqs_flat, train_labels_flat = self.convert_instances(train_insts_flat_list, seq_len, behaviors_list, progress_callback=train_progress_updater)
        test_seqs_flat, test_labels_flat = self.convert_instances(test_insts_flat_list, seq_len, behaviors_list, progress_callback=test_progress_updater)
        
        if not train_labels_flat:
             return None, None, None

        class_counts = np.bincount([label.item() for label in train_labels_flat], minlength=len(behaviors_list))
        total_samples = float(sum(class_counts))
        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total_samples / (len(behaviors_list) * count)
                class_weights.append(weight)
            else:
                class_weights.append(0)
        
        train_standard_dataset = StandardDataset(train_seqs_flat, train_labels_flat)
        test_standard_dataset = StandardDataset(test_seqs_flat, test_labels_flat)
        
        return train_standard_dataset, test_standard_dataset, class_weights
    
def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a batch list of (seq_tensor, label_tensor), stack them into batched tensors:
    - dcls: (batch_size, seq_len, feature_dim)
    - lbls: (batch_size,)
    """
    if not batch: # Handle empty batch
        return torch.empty(0), torch.empty(0)
    dcls = [item[0] for item in batch]
    lbls = [item[1] for item in batch]
    if not dcls or not lbls: # If lists became empty after filtering
        return torch.empty(0), torch.empty(0)
    try:
        return torch.stack(dcls), torch.stack(lbls)
    except Exception as e:
        print(f"Error in collate_fn: {e}. Batch items dcls len: {len(dcls)}, lbls len: {len(lbls)}")
        # Potentially print shapes of items if error persists
        # for i, item_d in enumerate(dcls): print(f"dcls[{i}].shape: {item_d.shape}")
        # for i, item_l in enumerate(lbls): print(f"lbls[{i}].shape: {item_l.shape}, value: {item_l}")
        raise e



def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Given a square matrix x, return its off-diagonal elements flattened.
    Used for covariance loss (VICReg-style).
    """
    n, m = x.shape
    assert n == m, f"Matrix must be square, got {n}x{m}"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class PerformanceReport:
    """
    Simple container for storing sklearn classification_report (dict) and confusion_matrix (ndarray).
    """

    def __init__(self, sklearn_report: dict, confusion_matrix_arr: np.ndarray): # Renamed for clarity
        self.sklearn_report = sklearn_report
        self.confusion_matrix = confusion_matrix_arr # Store the array


def train_lstm_model(
    train_set, # Can now be BalancedDataset or StandardDataset
    test_set, 
    seq_len: int,
    behaviors: list[str], # This should be the canonical list of behaviors used for model output
    batch_size: int = 512,
    lr: float = 1e-4, # Initial learning rate for Adam, schedule overrides it
    epochs: int = 10,
    device=None,
    class_weights=None
) -> tuple[nn.Module | None, list[PerformanceReport] | None, int]: 
    """
    Train a joint LSTM + linear classifier head on the provided train/test BalancedDatasets.
    Returns (model_with_best_weights, list_of_reports_for_best_run, best_epoch_index)
    The list_of_reports contains a PerformanceReport for each epoch of the run that produced the best model.
    """

    if len(train_set) == 0:
        print("Error: Training set is empty. Cannot train model.")
        return None, None, -1
    if len(test_set) == 0:
        print("Warning: Test set is empty. Model will train but validation metrics will be absent.")
        # Proceed with training, but validation will be skipped or produce empty reports.

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True # num_workers=0 for main thread
    )
    # Only create test_loader if test_set is not empty
    test_loader = None
    if len(test_set) > 0:
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True
        )


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")


    overall_best_f1 = -1.0 
    overall_best_model_state_dict = None
    overall_best_epoch_idx = -1
    overall_best_reports_list = [] 

    # Single trial as per previous refactoring (Item 3 was about best epoch in single trial)
    # The multi-trial logic (Item 4) is handled in workthreads.py
    trial_num = 0 # For print statements, effectively trial 1
    print(f"--- Starting Training Run ---") # Changed from "Trial" to "Run"
    
    model = classifier_head.classifier(
        in_features=768, out_features=len(behaviors), seq_len=seq_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    current_run_reports = [] 
    current_run_best_f1_local = -1.0 
    current_run_best_model_state_dict_local = None
    current_run_best_epoch_local = -1

    if epochs == 0:
        print("Warning: Number of epochs is 0. No training will occur.")
        # Capture initial model state if no training
        if overall_best_model_state_dict is None: 
            overall_best_model_state_dict = model.state_dict().copy() 
            overall_best_epoch_idx = -1 # No "best" epoch if no training
            # overall_best_reports_list remains empty

    for e in range(epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        num_batches = 0
        for i, (d_batch, l_batch) in enumerate(train_loader): # Renamed d,l for clarity
            if d_batch.nelement() == 0 : continue # Skip empty batches from collate_fn
            d_batch = d_batch.to(device).float()
            l_batch = l_batch.to(device)
            
            optimizer.zero_grad()
            lstm_logits, linear_logits, rawm = model(d_batch) # model.forward()
            
            logits = lstm_logits + linear_logits
            inv_loss = criterion(logits, l_batch)
            
            rawm_centered = rawm - rawm.mean(dim=0)
            # Ensure rawm_centered is 2D and batch size > 1 for meaningful covariance
            if rawm_centered.ndim == 2 and rawm_centered.shape[0] > 1:
                covm = (rawm_centered.T @ rawm_centered) / (rawm_centered.shape[0] -1) # Corrected Covariance calculation (Features x Features)
                # To match VICReg off-diagonal of (Batch x Batch) as before:
                # covm_batch_samples = (rawm_centered @ rawm_centered.T) / (rawm_centered.shape[0] -1) # if N > 1
                # covm_loss = torch.sum(torch.pow(off_diagonal(covm_batch_samples), 2)) / rawm_centered.shape[1] # Denom is feature_dim (128)
                
                # Using original formulation for (Batch x Batch) covariance for off_diagonal
                # This requires N > 1 for off_diagonal to work.
                covm_batch_samples = (rawm_centered @ rawm_centered.T) / rawm_centered.shape[0] # Original denominator for stability if N is small
                if rawm_centered.shape[0] > 1: # off_diagonal needs N > 1
                     covm_loss = torch.sum(torch.pow(off_diagonal(covm_batch_samples), 2)) / rawm_centered.shape[1]
                else:
                     covm_loss = torch.tensor(0.0).to(device) # No covariance loss for batch size 1
            else:
                covm_loss = torch.tensor(0.0).to(device) # No covariance loss if not suitable

            loss = inv_loss + covm_loss # Add scaling factor for covm_loss if needed, e.g., 0.1 * covm_loss
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches +=1
            if i % 50 == 0: 
                print(f"[Run/Epoch {e+1}/{epochs} Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f} (Inv: {inv_loss.item():.4f}, Cov: {covm_loss.item() if isinstance(covm_loss, torch.Tensor) else covm_loss:.4f})")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"--- End of Epoch {e+1}/{epochs} --- Avg Train Loss: {avg_epoch_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s ---")

        # --- Validation pass ---
        wf1score = -1.0 # Default if no validation
        full_report_obj = PerformanceReport({}, np.array([])) # Default empty report

        if test_loader and len(test_set) > 0: # Only validate if there's a test_loader and test_set
            model.eval()
            actuals, predictions_val = [], [] # Renamed predictions
            with torch.no_grad():
                for i, (d_batch, l_batch) in enumerate(test_loader):
                    if d_batch.nelement() == 0 : continue
                    d_batch = d_batch.to(device).float()
                    l_batch = l_batch.to(device) # Labels are already on device from loader
                    logits_val = model.forward_nodrop(d_batch)
                    actuals.extend(l_batch.cpu().numpy())
                    predictions_val.extend(logits_val.argmax(1).cpu().numpy())

            if not actuals or not predictions_val: 
                print(f"Warning: No actuals or predictions in validation for Epoch {e+1}. Using F1= -1.0.")
            else:
                # Ensure target_names matches the model's output dimension length
                report_dict = classification_report(actuals, predictions_val, target_names=behaviors[:len(model.lin2.weight)], output_dict=True, zero_division=0)
                wf1score = report_dict["weighted avg"]["f1-score"]
                # Ensure confusion matrix labels match the actual unique labels present, or use full range
                unique_labels_present = sorted(list(set(actuals) | set(predictions_val)))
                cm_labels = unique_labels_present if unique_labels_present else list(range(len(behaviors)))

                try:
                    cm = confusion_matrix(actuals, predictions_val, labels=cm_labels)
                    full_report_obj = PerformanceReport(report_dict, cm)
                except ValueError as ve: # If labels in cm are problematic
                    print(f"Warning: Could not generate confusion matrix for epoch {e+1}: {ve}. Using empty CM.")
                    full_report_obj = PerformanceReport(report_dict, np.array([]))


        current_run_reports.append(full_report_obj)
        print(f"Run/Epoch {e+1} - Weighted F1: {wf1score:.4f}")

        if wf1score > current_run_best_f1_local:
            current_run_best_f1_local = wf1score
            current_run_best_model_state_dict_local = model.state_dict().copy()
            current_run_best_epoch_local = e
    
    # After all epochs for this single run are done:
    # This run's best becomes the overall best because it's the only run.
    if current_run_best_f1_local > overall_best_f1: # Check if any valid F1 was found
        overall_best_f1 = current_run_best_f1_local
        overall_best_model_state_dict = current_run_best_model_state_dict_local
        overall_best_epoch_idx = current_run_best_epoch_local
        overall_best_reports_list = current_run_reports.copy()
        print(f"--- Training Run Complete. Best F1: {overall_best_f1:.4f} at Epoch {overall_best_epoch_idx + 1} ---")
    elif epochs > 0 and overall_best_f1 == -1.0: # If training ran but no improvement over initial -1.0
        # This means current_run_best_f1_local was always <= -1.0
        # We should still save the last state or the initial state if epochs was 0
        print("Warning: No improvement in F1 score during training. Saving last model state or initial if epochs=0.")
        if current_run_best_model_state_dict_local: # If at least one epoch ran and had a state
             overall_best_model_state_dict = current_run_best_model_state_dict_local
             overall_best_epoch_idx = current_run_best_epoch_local if current_run_best_epoch_local !=-1 else epochs-1
             overall_best_reports_list = current_run_reports.copy()
        elif overall_best_model_state_dict is None and epochs > 0: # Should not happen if epochs > 0
            overall_best_model_state_dict = model.state_dict().copy() # fallback to last state
            overall_best_epoch_idx = epochs -1
            overall_best_reports_list = current_run_reports.copy()
        # If epochs was 0, overall_best_model_state_dict was already set to initial.


    if overall_best_model_state_dict is not None:
        final_best_model = classifier_head.classifier(
            in_features=768, out_features=len(behaviors), seq_len=seq_len
        ).to(device) 
        final_best_model.load_state_dict(overall_best_model_state_dict)
        final_best_model.eval() 
        return final_best_model, overall_best_reports_list, overall_best_epoch_idx
    else:
        print("Error: No model was effectively trained or no best model state was captured.")
        return None, None, -1


"""
# Example Usage (uncomment to run):
print(f"CUDA available? {torch.cuda.is_available()}")

# 1) Create or load a Project:
proj = Project.create_project('../', 'my_lab_project')  # or Project('../my_lab_project')

# 2) Add a camera:
cam_settings = {
    "name": "cam1",
    "rtsp_url": "rtsp://admin:password@192.168.1.10:8554/stream0",
    "framerate": 10,
    "resolution": 256,
    "crop_left_x": 0.1,
    "crop_top_y": 0.1,
    "crop_width": 0.8,
    "crop_height": 0.8,
}
proj.create_camera("cam1", cam_settings)

# 3) Start recording:
cam = proj.cameras["cam1"]
session_folder = cam.create_recording_dir()
cam.start_recording(session_folder, segment_time=300)  # segment every 5 minutes
time.sleep(60)  # record for a bit
cam.stop_recording()

# 4) Encode & classify:
proj.encode_recordings()
proj.infer_recordings(device_str="cuda:0") # Changed to device_str

# 5) Create a dataset & train:
dataset = proj.create_dataset("my_behaviors", ["rest", "rearing", "grooming"], ["20250101\\cam1-120000-AM", ...]) # Fixed example recordings_whitelist
train_ds, test_ds = proj.load_dataset("my_behaviors", seq_len=31)
if train_ds and test_ds: # Check if datasets loaded successfully
    model, reports, best_ep = train_lstm_model(train_ds, test_ds, seq_len=31, behaviors=["rest", "rearing", "grooming"], epochs=10)
    # 6) Save your best_model etc. as needed (now handled by workthreads.py)
else:
    print("Failed to load datasets for training.")
"""