import eel
import time
import ctypes
import yaml
import socket # For finding an available port
import os
import math # Potentially used by other imported modules
import sys
import random # Potentially used by other imported modules
from sys import exit # Though not explicitly used, good to have if needed for forceful exit
import subprocess
import shutil # Potentially used by other imported modules
import cairo # Potentially used by other imported modules (e.g., Actogram)

import cv2 # Used by cbas.py, gui_state.py

import base64 # Used by various pages for image transfer
from datetime import datetime, timezone # Potentially used

import h5py # Used by cbas.py

import torch # Base PyTorch
# Unused specific torch imports in this file, but good for context if they were here
# from torch.cuda.amp import autocast, GradScaler
from torch import nn
# from torch.utils.data import Dataset # Defined in cbas.py
# from torch.utils.data import DataLoader # Used in cbas.py
# from transformers import AutoImageProcessor, AutoModel # Used in cbas.py DINOEncoder
# import torch.optim as optim # Used in cbas.py

# from sklearn.metrics import classification_report # Used in cbas.py

# from decord import VideoReader # Used in cbas.py
# from decord import cpu, gpu # Used in cbas.py

from cmap import Colormap # Used in label_train_page.py

import queue # Standard library queue, though task_queue here is not directly used by workthreads.py

import numpy as np # Used throughout
import pandas as pd # Used by cbas.py Actogram

import threading # For gpu_lock, though workthreads.py uses its own locks

# Import project-specific modules (assuming final names)
import cbas # Should be after cbas.py is finalized
from classifier_head import classifier # Referenced by cbas.py, workthreads.py
import gui_state # Central state management
import startup_page # Eel-exposed functions for startup
import record_page  # Eel-exposed functions for recording
import label_train_page # Eel-exposed functions for labeling and training
import visualize_page # Eel-exposed functions for visualization
import workthreads    # Background processing threads

# --- Global Variables ---
# These globals seem to be remnants from older versions or for direct Eel interaction not fully refactored into gui_state.
# Review if they are still actively used or can be removed/managed via gui_state.py.

# active_streams: dict = {} # Managed by gui_state.proj.active_recordings now
# recordings: str = ""     # Managed by gui_state.proj.recordings_dir now
# stop_threads: bool = False # Thread stopping is handled within workthreads.py via raise_exception

# progresses: list = [] # Seems related to an old inference progress bar, not used by workthreads.py

# label_dict_path: str | None = None # Managed by gui_state.label_dataset now
# label_dict: dict | None = None       # Managed by gui_state.label_dataset now
# col_map: Colormap | None = None       # Managed by gui_state.label_col_map now

# label_capture: cv2.VideoCapture | None = None # Managed by gui_state.label_capture
# label_videos: list[str] = []                 # Managed by gui_state.label_videos
# label_vid_index: int = -1                    # Managed by gui_state.label_vid_index
# label_index: int = -1                        # Managed by gui_state.label_index
# label: int = -1                              # Managed by gui_state.label_type
# start: int = -1                              # Managed by gui_state.label_start

# task_queue: queue.Queue = queue.Queue() # Not directly used by the new workthreads.py (uses internal lists)

# instance_stack: list | None = None # Managed by gui_state.label_history

# gpu_lock: threading.Lock = threading.Lock() # Not used by current workthreads.py (streams are per-thread)

# classification_threads: list = [] # Replaced by gui_state.classify_thread

# actogram: cbas.Actogram | None = None # Managed by gui_state.cur_actogram

# tthread: threading.Thread | None = None # Replaced by gui_state.training_thread

def find_available_port(start_port=8000, max_tries=100):
    """Finds an available network port."""
    for i in range(max_tries):
        port_to_try = start_port + i
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port_to_try)) # Try to bind to the port
            sock.close()
            return port_to_try
        except OSError: # Port already in use
            continue
    raise IOError("No free ports found for Eel.")

def main():
    """Main function to initialize and start the Eel application."""
    
    # Ensure required directories for Eel exist
    os.makedirs("frontend", exist_ok=True) # Assuming 'frontend' is your web root

    # Find an available port for Eel to run on
    try:
        port = find_available_port()
        print(f"Eel will run on port: {port}")
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Set path to Electron executable
    # Ensure this path is correct for your environment or use a more robust way to find it.
    # For example, using a package like `electron_binaries` or environment variables.
    electron_executable_path = os.path.join("node_modules", "electron", "dist", "electron")
    if sys.platform == "win32":
        electron_executable_path += ".exe"
    electron_executable_path = os.path.abspath(electron_executable_path)

    if not os.path.exists(electron_executable_path):
        print(f"Error: Electron executable not found at {electron_executable_path}")
        print("Please ensure Electron is installed correctly in node_modules (e.g., via npm install electron)")
        # sys.exit(1) # Allow to run without Electron for browser mode if desired

    eel.browsers.set_path("electron", electron_executable_path)
    eel.init("frontend")  # Initialize Eel with the web folder

    print("Starting worker threads...")
    workthreads.start_threads() # Initialize and start background threads

    # Define options for starting Eel with Electron
    # '--remote-debugging-port=9222' can be useful for debugging Electron's main process
    # '--trace-warnings' can help identify deprecations or issues
    electron_options = [
        ".", # Path to the Eel app directory (current directory)
        str(port), # Port for Eel to communicate on
        "--trace-warnings"
        # "--remote-debugging-port=9222" # Uncomment for debugging
    ]
    
    eel_options = {
        'mode': 'electron', # Specify Electron mode
        'host': 'localhost',
        'port': port,
        'app_mode': True, # Important for Electron packaging
        # 'electron_flags': ["--enable-logging"] # Example flag
    }

    print(f"Starting Eel with Electron: {electron_executable_path} {' '.join(electron_options)}")
    
    try:
        # Attempt to start with Electron first
        if os.path.exists(electron_executable_path):
             eel.start("index.html", options=eel_options, block=False, suppress_error=True) # Start non-blocking
        else:
            print("Electron not found, attempting to start in default browser mode.")
            eel.start("index.html", mode=None, port=port, block=False, suppress_error=True) # Fallback to browser
    except Exception as e:
        print(f"Could not start Eel: {e}")
        print("Ensure your 'frontend/index.html' exists and Eel is configured correctly.")
        workthreads.stop_threads() # Attempt to stop threads if Eel fails to start
        sys.exit(1)


    # Main application loop (keeps Python script alive while Eel/Electron runs)
    try:
        while True:
            eel.sleep(1.0)  # eel.sleep allows Eel to process events
    except (KeyboardInterrupt, SystemExit):
        print("Application exiting...")
    finally:
        print("Cleaning up: Stopping worker threads...")
        workthreads.stop_threads()
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    # It's good practice to ensure gui_state variables are initialized if threads access them early.
    # Threads in workthreads.py are started after gui_state.proj might be set by startup_page.
    # gui_state.py itself initializes them to None or empty lists.
    main()