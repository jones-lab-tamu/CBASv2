import eel
import time
import ctypes
import yaml
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


def tab20_map(val):

    if val < 10:
        return val * 2
    else:
        return (val - 10) * 2 + 1


def fill_colors(frame):
    global label_videos
    global label_vid_index
    global label_capture
    global label
    global label_index
    global start

    global label_dict
    global col_map

    behaviors = label_dict["behaviors"]

    labels = label_dict["labels"]

    cur_video = label_videos[label_vid_index]
    amount_of_frames = label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    for i, b in enumerate(behaviors):

        sub_labels = labels[b]

        color = str(col_map(tab20_map(i))).lstrip("#")
        color = np.flip(np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)]))

        for l in sub_labels:
            if l["video"] != cur_video:
                continue
            sInd = l["start"]
            eInd = l["end"]

            marker_posS = int(frame.shape[1] * sInd / amount_of_frames)
            marker_posE = int(frame.shape[1] * eInd / amount_of_frames)

            frame[
                -49:,
                marker_posS : marker_posE + 1,
            ] = color

    if label != -1:
        color = str(col_map(tab20_map(label))).lstrip("#")
        color = np.flip(np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)]))

        stemp = min(start, label_index)
        eInd = max(start, label_index)

        sInd = stemp

        marker_posS = int(frame.shape[1] * sInd / amount_of_frames)
        marker_posE = int(frame.shape[1] * eInd / amount_of_frames)

        frame[
            -49:,
            marker_posS : marker_posE + 1,
        ] = color

    return frame


eel.init("frontend")
eel.browsers.set_path("electron", "node_modules/electron/dist/electron")


@eel.expose
def get_progress_update():
    if len(progresses) == 0:
        eel.inferLoadBar(False)()
    else:
        eel.inferLoadBar(progresses)()

@eel.expose
def kill_streams():
    global active_streams
    global stop_threads
    global tthread

    workthreads.stop_threads()

    for stream in active_streams.keys():
        active_streams[stream].communicate(input=b"q")

    stop_threads = True

    if tthread:
        tthread.raise_exception()
        tthread.join()

eel.start("frontend/index.html", mode="electron", block=False)

workthreads.start_threads()

while True:
    eel.sleep(1.0)
