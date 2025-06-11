import eel
import time
import ctypes
import yaml
import socket
import os
import math
import sys
import random
from sys import exit
import subprocess
import shutil
import cairo

import cv2

import base64
from datetime import datetime, timezone

import h5py

import torch
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
import torch.optim as optim

from sklearn.metrics import classification_report

from decord import VideoReader
from decord import cpu, gpu

from cmap import Colormap

import queue

import numpy as np
import pandas as pd

import ctypes

import threading

from classifier_head import classifier

import startup_page
import record_page
import label_train_page
import visualize_page
import workthreads

active_streams = {}

recordings = ""

stop_threads = False

progresses = []

label_dict_path = None
label_dict = None
col_map = None

label_capture = None
label_videos = []
label_vid_index = -1
label_index = -1
label = -1
start = -1

task_queue = queue.Queue()

instance_stack = None

gpu_lock = threading.Lock()

classification_threads = []

actogram = None


tthread = None

sock = socket.socket()
sock.bind(('', 0))
port = sock.getsockname()[1]
sock.close()

eel.browsers.set_path("electron", "node_modules/electron/dist/electron")

eel.init("frontend")

electron_path = os.path.abspath("node_modules/electron/dist/electron")

# Start the Eel server in a background thread
eel.start("frontend/index.html", mode=None, block=False, port=port)

# Give the server thread a moment to start and bind to the port
time.sleep(2)

# Launch the Electron front-end, which will now connect successfully
# Using Popen without shell=True is safer and allows it to be managed as a child process
subprocess.Popen([electron_path, ".", str(port), "--trace-warnings"])


workthreads.start_threads()

while True:
    eel.sleep(1.0)
