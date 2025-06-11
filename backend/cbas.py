import os
import h5py

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

import classifier_head

from torch import nn
import torch.optim as optim

def encode_file(encoder: nn.Module, path: str) -> str:
    # We use decord to efficiently load a video as a Torch tensor.
    try:
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
    except:
        return None

    frames = reader.get_batch(range(0, len(reader), 1)).asnumpy()
    frames = torch.from_numpy(frames[:, :, :, 1] / 255).half()

    # TODO: I am not fully convinced this is most efficient? How can we increase throughput.
    batch_size = 256

    clss = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]

        with torch.no_grad() and torch.amp.autocast(device_type=encoder.device.type):
            out = encoder(batch.unsqueeze(1).to(encoder.device))

        out = out.squeeze(1).to("cpu")
        clss.extend(out)

    # Build the h5 file.
    out_file_path = os.path.splitext(path)[0] + "_cls.h5"
    with h5py.File(out_file_path, "w") as file:
        file.create_dataset("cls", data=torch.stack(clss).numpy())

    return out_file_path

def infer_file(
    file_path: str,
    model: nn.Module,
    dataset_name: str,
    behaviors: str,
    seq_len: int,
    device=None,
):
    output_file = file_path.replace(
        "_cls.h5", "_" + dataset_name + "_outputs.csv"
    )
    with h5py.File(file_path, "r") as file:
        cls = np.array(file["cls"][:])

    cls = torch.from_numpy(cls - np.mean(cls, axis=0)).half()

    predictions = []
    batch = []

    if len(cls) < seq_len:
        return None

    model = model.to(device)

    for ind in range(seq_len // 2, len(cls) - seq_len // 2):
        batch.append(cls[ind - seq_len // 2 : ind + seq_len // 2 + 1])

        if len(batch) >= 4096 or ind == len(cls) - seq_len // 2 - 1:
            batch = torch.stack(batch)
            with torch.no_grad() and torch.amp.autocast(device):
                logits = model.forward_nodrop(batch.to(device))
                probs = torch.softmax(logits, dim=1)

                predictions.extend(probs.detach().cpu().numpy())

            batch = []

    total_predictions = []

    for ind in range(len(cls)):

        if ind < seq_len // 2:
            total_predictions.append(predictions[0])
        elif ind >= len(cls) - seq_len // 2:
            total_predictions.append(predictions[-1])
        else:
            total_predictions.append(predictions[ind - seq_len // 2])

    total_predictions = np.array(total_predictions)

    dataframe = pd.DataFrame(total_predictions, columns=behaviors)

    dataframe.to_csv(output_file)

    return output_file


class DinoEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super(DinoEncoder, self).__init__()
        self.device = device
        self.model = None # <-- CHANGE: Initialize model as None

    def _initialize_model(self):
        """Lazy initializer for the model."""
        if self.model is None:
            print("Initializing DINOv2 model (this may download on first run)...")
            # This is the slow, blocking call
            self.model = transformers.AutoModel.from_pretrained("facebook/dinov2-base").to(
                self.device
            )
            self.model.eval()

            # We obviously dont want to train DINO ourselves.
            for param in self.model.parameters():
                param.requires_grad = False
            print("DINOv2 model initialized.")

    def forward(self, x):
        self._initialize_model()

        # B: number of batches
        # S: input elements per sample
        # H: height
        # W: width
        B, S, H, W = x.shape

        x = x.to(self.device)

        # We inference black and white images, but DINOv2 takes color.  We need to add a color channel then.
        # Add a third dimension for color channel before width and height.
        # Then reshape tensor to copy along this dimension three times.
        x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        x = x.reshape(B * S, 3, H, W)

        # Run it through DINOv2
        with torch.no_grad():
            out = self.model(x)

        # Each image is converted to a 768 element wide hidden state.  We can reshape to to make this output:
        cls = out.last_hidden_state[:, 0, :].reshape(B, S, 768)

        return cls



class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, behaviors):
        self.seqs = seqs
        self.labels = labels
        self.behaviors = behaviors

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


class Recording:
    # Takes a path to the folder.
    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise FileNotFoundError(path)

        self.path = path
        self.name = os.path.basename(self.path)

        # We need to check all the files we have encodeded.
        self.unencoded_files = []

        self.video_files = [
            f.path for f in os.scandir(self.path) if f.path.endswith(".mp4")
        ]
        self.video_files = sorted(
            self.video_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        self.encoding_files = [
            f.path for f in os.scandir(self.path) if f.path.endswith(".h5")
        ]

        for video_file in self.video_files:
            h5_file_path = video_file.replace(".mp4", "_cls.h5")
            try:
                f = h5py.File(h5_file_path, "r")
            except:
                self.unencoded_files.append(video_file)

        for video_file in self.video_files:
            h5_file_path = video_file.replace(".mp4", "_cls.h5")
            try:
                f = h5py.File(h5_file_path, "r")
            except:
                self.unencoded_files.append(video_file)

        classification_files = [
            f.path for f in os.scandir(self.path) if f.path.endswith(".csv")
        ]

        self.classifications = {}

        for file in classification_files:
            model = file.split('_')[-2]

            if model not in self.classifications:
                self.classifications[model] = []
            
            self.classifications[model].append(file)
        
        print(self.path, self.classifications)

    def encode(self, encoder: nn.Module):
        for file in self.unencoded_files:
            out_file_path = encode_file(encoder, file)
            if out_file_path not in self.encoding_files:
                self.encoding_files.append(out_file_path)

    def infer(
        self,
        model: nn.Module,
        dataset_name: str,
        behaviors: str,
        seq_len: int,
        device=None,
    ):
        for file_path in self.encoding_files:
            infer_file(file_path, model, dataset_name, behaviors, seq_len, device=None)

class Camera:
    def __init__(self, config: dict[str, str], project: "Project"):
        self.config = config
        self.project = project

        self.recording = False

        self.name = config["name"]
        self.rtsp_url = config["rtsp_url"]
        self.framerate = config["framerate"]
        self.resolution = config["resolution"]
        self.crop_left_x = config["crop_left_x"]
        self.crop_top_y = config["crop_top_y"]
        self.crop_width = config["crop_width"]
        self.crop_height = config["crop_height"]

        self.path = os.path.join(self.project.cameras_dir, self.name)

    def settings_to_dict(self):
        return {
            "rtsp_url": self.rtsp_url,
            "framerate": self.framerate,
            "resolution": self.resolution,
            "crop_left_x": self.crop_left_x,
            "crop_top_y": self.crop_top_y,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
        }

    def update_settings(self, settings):
        self.rtsp_url = settings["rtsp_url"]
        self.framerate = settings["framerate"]
        self.resolution = settings["resolution"]
        self.crop_left_x = settings["crop_left_x"]
        self.crop_top_y = settings["crop_top_y"]
        self.crop_width = settings["crop_width"]
        self.crop_height = settings["crop_height"]

        self.write_settings_to_config()

    def create_recording_dir(self):
        date_dir = datetime.now().strftime("%Y%m%d")
        date_path = os.path.join(self.project.recordings_dir, date_dir)

        if not os.path.exists(date_path):
            os.mkdir(date_path)

        cam_dir = self.name + "-" + datetime.now().strftime("%I%M%S-%p")
        cam_path = os.path.join(date_path, cam_dir)

        if not os.path.exists(cam_path):
            os.mkdir(cam_path)

            return cam_path

        return False

    def start_recording(self, destination: str, segment_time: int):
        self.recording = True

        if self.name in self.project.active_recordings.keys():
            return False

        if not os.path.exists(destination):
            os.mkdir(destination)

        destination = os.path.join(destination, f"{self.name}_%05d.mp4")

        command = [
            "ffmpeg",
            "-loglevel",
            "panic",
            "-rtsp_transport",
            "tcp",
            "-i",
            str(self.rtsp_url),
            "-r",
            str(self.framerate),
            "-filter_complex",
            f"[0:v]crop=(iw*{self.crop_width}):(ih*{self.crop_height}):(iw*{self.crop_left_x}):(ih*{self.crop_top_y}),scale={self.resolution}:{self.resolution}[cropped]",
            "-map",
            "[cropped]",
            "-f",
            "segment",
            "-segment_time",
            str(segment_time),
            "-reset_timestamps",
            "1",
            "-hls_flags",
            "temp_file",
            "-y",
            destination,
        ]

        process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)

        self.project.active_recordings[self.name] = process

        return True

    def write_settings_to_config(self):
        cam_settings = {
            "name": self.name,
            "rtsp_url": self.rtsp_url,
            "framerate": self.framerate,
            "resolution": self.resolution,
            "crop_left_x": self.crop_left_x,
            "crop_top_y": self.crop_top_y,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
        }

        with open(os.path.join(self.path, "config.yaml"), "w+") as file:
            yaml.dump(cam_settings, file, allow_unicode=True)

    def stop_recording(self):
        self.recording = False

        active_recordings = self.project.active_recordings
        if self.name in active_recordings.keys():
            active_recordings[self.name].communicate(input=b"q")
            active_recordings.pop(self.name)

            return True

        return False

class Model:
    def __init__(self, path: str):
        self.path = path

        self.config_path = os.path.join(path, "config.yaml")
        self.weights_path = os.path.join(path, "model.pth")

        self.name = os.path.basename(path)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(self.config_path)

        with open(self.config_path) as file:
            self.config = yaml.safe_load(file)

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(self.weights_path)

class Dataset:
    def __init__(self, path: str):
        self.path = path

        self.config_path = os.path.join(path, "config.yaml")
        self.labels_path = os.path.join(path, "labels.yaml")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(self.config_path)

        with open(self.config_path) as file:
            self.config = yaml.safe_load(file)

        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(self.labels_path)

        with open(self.labels_path) as file:
            self.labels = yaml.safe_load(file)

    def update_metric(self, behavior, group, value):
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)

        config["metrics"][behavior][group] = value

        with open(self.config_path, "w+") as file:
            yaml.dump(config, file, allow_unicode=True)

def remove_leading_zeros(num):
    for i in range(0, len(num)):
        if num[i] != "0":
            return int(num[i:])
    return 0

class Actogram:
    def __init__(
        self,
        directory,
        model,
        behavior,
        framerate,
        start,
        binsize,
        color,
        threshold,
        norm,
        lightcycle,
        width=500,
        height=500,
    ):

        self.directory = directory
        self.model = model
        self.behavior = behavior
        self.framerate = framerate
        self.start = start
        self.color = color
        self.threshold = threshold
        self.norm = norm
        self.lightcycle = lightcycle

        self.width = width
        self.height = height

        self.binsize = binsize

        self.timeseries()

        self.draw()

    def draw(self):

        self.cycles = []
        for c in self.lightcycle:
            if c == "1":
                self.cycles.append(True)
            else:
                self.cycles.append(False)

        clocklab_time = (
            "{:02d}".format(int(self.start))
            + ":"
            + "{:02d}".format(int(60 * (self.start - int(self.start))))
        )

        clocklab_file = [
            self.behavior,
            "01-jan-2024",
            clocklab_time,
            self.binsize / self.framerate / 60 * 4,
            0,
            0,
            0,
        ]

        bins = []

        for b in range(0, len(self.totalts), self.binsize):
            bins.append(
                (
                    sum(np.array(self.totalts[b : b + self.binsize]) >= self.threshold),
                    b / self.framerate / 3600,
                )
            )
            clocklab_file.append(
                sum(np.array(self.totalts[b : b + self.binsize]) >= self.threshold)
            )

        df = pd.DataFrame(data=np.array(clocklab_file))

        self.clfile = os.path.join(
            self.directory, self.model + "-" + self.behavior + "-" + "clocklab.csv"
        )

        df.to_csv(self.clfile, header=False, index=False)

        awdfile = self.clfile.replace(".csv", ".awd")

        if os.path.exists(awdfile):
            os.remove(awdfile)

        os.rename(self.clfile, self.clfile.replace(".csv", ".awd"))

        self.timeseries_data = bins

        if len(bins) < 2:
            print("Not enough videos to make an actogram.")
            return

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(surface)

        width = self.width
        height = self.height

        ctx.scale(width, height)

        self.align_draw(ctx)

        self.file = os.path.join(
            self.directory, self.model + "-" + self.behavior + "-" + "actogram.png"
        )

        surface.write_to_png(self.file)

        frame = cv2.imread(self.file)

        ret, frame = cv2.imencode(".jpg", frame)

        frame = frame.tobytes()

        blob = base64.b64encode(frame)
        self.blob = blob.decode("utf-8")

    def align_draw(self, ctx):

        padding = 0.01

        actogram_width = 1 - 2 * padding
        actogram_height = 1 - 2 * padding

        cx = padding
        cy = padding

        self.draw_actogram(ctx, cx, cy, actogram_width, actogram_height, padding)

    def draw_actogram(self, ctx, tlx, tly, width, height, padding):

        ctx.set_line_width(0.005)

        ctx.rectangle(
            tlx - padding / 4,
            tly - padding / 4,
            width + padding / 2,
            height + padding / 2,
        )
        ctx.set_source_rgba(0.1, 0.1, 0.1, 1)
        ctx.fill()

        tsdata = self.timeseries_data

        total_days = math.ceil((tsdata[-1][1] + self.start) / 24)

        if total_days % 2 == 0:
            total_days -= 1

        day_height = height / total_days

        bin_width = 1 / 48 * self.binsize / 36000

        ts = np.array([a[0] for a in tsdata])
        times = np.array([a[1] + self.start for a in tsdata])

        for d in range(total_days):

            by = tly + (d + 1) * day_height

            d1 = d

            valid = np.logical_and(times > (d1 * 24), times <= ((d1 + 2) * 24))

            if d1 % 2 != 0:
                adj_times = times[valid] + 24
            else:
                adj_times = times[valid]

            adj_times = adj_times % 48

            series = ts[valid]

            if len(series) == 0:
                continue

            # normalize the series
            series = np.array(series)
            series = series / self.norm

            series = series * 0.90

            if self.cycles is not None and d < len(self.cycles):
                LD = self.cycles[d]
            else:
                LD = True

            if LD:
                ctx.set_source_rgb(223 / 255, 223 / 255, 223 / 255)
                ctx.rectangle(
                    tlx + 0 / 48 * width, by - day_height, 6 / 48 * width, day_height
                )
                ctx.fill()
                ctx.rectangle(
                    tlx + 18 / 48 * width, by - day_height, 12 / 48 * width, day_height
                )
                ctx.fill()
                ctx.rectangle(
                    tlx + 42 / 48 * width, by - day_height, 6 / 48 * width, day_height
                )
                ctx.fill()

                ctx.set_source_rgb(255 / 255, 239 / 255, 191 / 255)
                ctx.rectangle(
                    tlx + 6 / 48 * width, by - day_height, 12 / 48 * width, day_height
                )
                ctx.fill()
                ctx.rectangle(
                    tlx + 30 / 48 * width, by - day_height, 12 / 48 * width, day_height
                )
                ctx.fill()

            else:
                ctx.set_source_rgb(223 / 255, 223 / 255, 223 / 255)
                ctx.rectangle(
                    tlx + 0 / 48 * width, by - day_height, 6 / 48 * width, day_height
                )
                ctx.fill()
                ctx.rectangle(
                    tlx + 18 / 48 * width, by - day_height, 12 / 48 * width, day_height
                )
                ctx.fill()
                ctx.rectangle(
                    tlx + 42 / 48 * width, by - day_height, 6 / 48 * width, day_height
                )
                ctx.fill()

                ctx.set_source_rgb(246 / 255, 246 / 255, 246 / 255)
                ctx.rectangle(
                    tlx + 6 / 48 * width, by - day_height, 12 / 48 * width, day_height
                )
                ctx.fill()
                ctx.rectangle(
                    tlx + 30 / 48 * width, by - day_height, 12 / 48 * width, day_height
                )
                ctx.fill()

            for t in range(len(adj_times)):
                timepoint = adj_times[t]
                value = series[t]

                a_time = timepoint / 48

                ctx.rectangle(
                    tlx + a_time * width,
                    by - value * day_height,
                    bin_width * width,
                    value * day_height,
                )
                ctx.set_source_rgba(
                    self.color[0] / 255, self.color[1] / 255, self.color[2] / 255, 1
                )
                ctx.fill()

        for d in range(total_days):

            by = tly + (d + 1) * day_height

            ctx.set_line_width(0.002)
            ctx.set_source_rgb(0, 0, 0)
            ctx.move_to(tlx, by)
            ctx.line_to(tlx + 48 / 48 * width, by)
            ctx.stroke()

    def timeseries(self):

        behavior = self.behavior

        valid_files = [
            file
            for file in os.listdir(self.directory)
            if file.endswith(".csv") and "_" + self.model + "_" in file
        ]

        # Split the files into path and camera number for use later
        split_files = [
            (
                os.path.join(self.directory, file),
                remove_leading_zeros(file.split("_")[-3]),
            )
            for file in valid_files
        ]

        split_files.sort(key=lambda vf: vf[1])

        first_num = split_files[0][1]
        last_num = split_files[-1][1]

        if len(split_files) != (last_num - first_num) + 1:

            prev_num = -1
            for vf, num in split_files:
                if num != prev_num + 1:
                    raise Exception(f"Missing number - {prev_num+1}")

        self.totalts = []

        col_index = -1

        continuous = False

        for vf, num in split_files:

            dataframe = pd.read_csv(vf)

            if col_index == -1:
                behaviors = dataframe.columns.to_list()[1:]

                col_index = behaviors.index(behavior)

            top = np.argmax(dataframe[behaviors].to_numpy(), axis=1) == col_index
            values = np.max(dataframe[behaviors].to_numpy(), axis=1)

            values = top * values

            if continuous:
                frames = dataframe[behavior].to_list()
            else:
                frames = values

            self.totalts.extend(frames)

class InvalidProject(Exception):
    def __init__(self, path):
        self.path = path
        super().__init__(f"{self.path} is not a valid project")

class Project:
    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise InvalidProject(path)
        self.path = path

        self.cameras_dir = os.path.join(self.path, "cameras")
        self.recordings_dir = os.path.join(self.path, "recordings")
        self.models_dir = os.path.join(self.path, "models")
        self.datasets_dir = os.path.join(self.path, "data_sets")

        if not os.path.isdir(self.cameras_dir):
            raise InvalidProject(self.cameras_dir)
        if not os.path.isdir(self.recordings_dir):
            raise InvalidProject(self.recordings_dir)
        if not os.path.isdir(self.models_dir):
            raise InvalidProject(self.models_dir)
        if not os.path.isdir(self.datasets_dir):
            raise InvalidProject(self.datasets_dir)

        self.active_recordings = {}

        # Build out our list of recordings:
        self.recordings: dict[str, dict[str, Recording]] = {}
        days = [f.path for f in os.scandir(self.recordings_dir) if f.is_dir()]

        for day in days:
            day_str = os.path.basename(day)

            subrecording_folders = [f.path for f in os.scandir(day)]
            self.recordings[day_str] = {}
            for folder in subrecording_folders:
                rec = Recording(folder)
                self.recordings[day_str][rec.name] = rec

        # Build out our list of cameras
        self.cameras = {}
        for cam_path in [f.path for f in os.scandir(self.cameras_dir) if f.is_dir()]:
            with open(os.path.join(cam_path, "config.yaml")) as file:
                config = yaml.safe_load(file)

            self.cameras[config["name"]] = Camera(config, self)

        self.models = {}
        for model_path in [f.path for f in os.scandir(self.models_dir) if f.is_dir()]:
            self.models[os.path.basename(model_path)] = Model(model_path)
        
        self.models["JonesLabModel"] = Model(os.path.join("models", "JonesLabModel"))

        # Build out list of datasets
        self.datasets = {}
        for dataset_path in [
            f.path for f in os.scandir(self.datasets_dir) if f.is_dir()
        ]:
            dataset_name = os.path.basename(dataset_path)

            self.datasets[dataset_name] = Dataset(dataset_path)

    @staticmethod
    def create_project(parent_directory: str, project_name: str) -> "Project":
        # main project directory
        project_path = os.path.join(parent_directory, project_name)
        # check to see if the project already exists
        if os.path.exists(project_path):
            print("Project already exists. Please choose a different name or location.")
            return None
        else:
            print("Creating project...")

        cameras_dir = os.path.join(project_path, "cameras")
        recordings_dir = os.path.join(project_path, "recordings")
        models_dir = os.path.join(project_path, "models")
        datasets_dir = os.path.join(project_path, "data_sets")

        os.mkdir(project_path)
        os.mkdir(cameras_dir)
        os.mkdir(recordings_dir)
        os.mkdir(models_dir)
        os.mkdir(datasets_dir)

        project = Project(project_path)

        # make all those directories

        print(f"Project creation successful!")

        return project

    def encode_recordings(self):
        # Loading the model weights takes a long time, lets just do it once, then
        # use that for all of our encoding.
        encoder = DinoEncoder()

        for day in self.recordings:
            for recording in self.recordings[day]:
                self.recordings[day][recording].encode(encoder)

    def infer_recordings(self, device=None):
        model = torch.load("CBASv2/models/JonesLabModel/model.pth")

        with open("CBASv2/models/JonesLabModel/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        state_dict = model
        model = classifier_head.classifier(
            in_features=768,
            out_features=len(config["behaviors"]),
            seq_len=config["seq_len"],
        )
        model.load_state_dict(state_dict=state_dict)
        model.eval()

        model = model.to(device)

        for day in self.recordings:
            for recording in self.recordings[day]:
                self.recordings[day][recording].infer(
                    model, "JonesLabModel", config["behaviors"], 31, "cuda"
                )

    #    def create_camera(self, name, rtsp_url, framerate=10, resolution=256, crop_left_x=0, crop_top_y=0, crop_width=1, crop_height=1):
    def create_camera(self, name, settings):
        # set up a folder for the camera
        camera_path = os.path.join(self.cameras_dir, name)

        if os.path.exists(camera_path):
            return None

        print("Creating camera...")
        os.mkdir(camera_path)

        # set up the camera config
        camera_config = {
            "name": name,
            "rtsp_url": settings["rtsp_url"],
            "framerate": settings["framerate"],
            "resolution": settings["resolution"],
            "crop_left_x": settings["crop_left_x"],
            "crop_top_y": settings["crop_top_y"],
            "crop_width": settings["crop_width"],
            "crop_height": settings["crop_height"],
        }

        # save the camera config
        with open(os.path.join(camera_path, "config.yaml"), "w+") as file:
            yaml.dump(camera_config, file, allow_unicode=True)

        camera = Camera(camera_config, self)
        self.cameras[name] = camera

        return camera

    def remove_camera(self, camera_name):
        camera = self.cameras[camera_name]

        shutil.rmtree(camera.path)

        del self.cameras[camera_name]

    # Takes a list of instances and converts it into sequences tensors and indices for training.
    def convert_instances(
        self, insts: list, seq_len: int, behaviors: list[str]
    ) -> tuple[list[torch.tensor], list[int]]:
        seqs = []
        labels = []

        for inst in insts:
            start = int(inst["start"])
            end = int(inst["end"])
            video_path = inst["video"]
            cls_path = video_path.replace(".mp4", "_cls.h5")

            cls_file = h5py.File(cls_path, "r")

            cls = cls_file["cls"][:]

            video_mean = np.mean(cls)

            half_seqlen = seq_len // 2
            if cls.shape[0] - half_seqlen < start or half_seqlen > end:
                continue

            # Pick the 'true'  start and end points, all sequences inside will be used.
            valid_start = max((seq_len // 2) + 1, start) - 1
            valid_end = min(cls.shape[0] - (half_seqlen + 1), end) - 1

            inds = list(range(valid_start, valid_end))

            for t in inds:
                ind = t

                clss = cls_file["cls"][ind - half_seqlen : ind + (half_seqlen + 1)]

                clss = torch.from_numpy(clss - video_mean).half()

                if clss.shape[0] != seq_len:
                    continue

                seqs.append(clss)
                label = behaviors.index(inst["label"])
                labels.append(torch.tensor(label).long())

        all = list(zip(seqs, labels))
        random.shuffle(all)
        seqs, labels = map(list, zip(*all))

        return torch.stack(seqs), torch.stack(labels)

    def create_dataset(self, name: str, behaviors: list[str], recordings: list[str]):
        directory = os.path.join(self.datasets_dir, name)
        os.mkdir(directory)

        dataset_config = os.path.join(directory, "config.yaml")
        label_file = os.path.join(directory, "labels.yaml")

        whitelist = []

        for r in recordings:
            whitelist.append(r + "\\")

        metrics = {
            b: {
                "Train #": 0,
                "Test #": 0,
                "Precision": "N/A",
                "Recall": "N/A",
                "F1 Score": "N/A",
            }
            for b in behaviors
        }

        dconfig = {
            "name": name,
            "behaviors": behaviors,
            "whitelist": whitelist,
            "model": None,
            "metrics": metrics,
        }

        labelconfig = {"behaviors": behaviors, "labels": {b: [] for b in behaviors}}

        with open(dataset_config, "w+") as file:
            yaml.dump(dconfig, file, allow_unicode=True)

        with open(label_file, "w+") as file:
            yaml.dump(labelconfig, file, allow_unicode=True)

        dataset = Dataset(directory)
        self.datasets[name] = dataset
        return dataset

    def load_dataset(
        self, name: str, seed=42, split=0.2, seq_len=15
    ) -> tuple[DatasetLoader, DatasetLoader]:
        dataset_path = os.path.join(self.path, "data_sets", name)

        if seq_len % 2 == 0:
            seq_len += 1

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(dataset_path)

        with open(os.path.join(dataset_path, "config.yaml"), "r") as file:
            config = yaml.safe_load(file)

        with open(os.path.join(dataset_path, "labels.yaml"), "r") as file:
            label_config = yaml.safe_load(file)

        behaviors = label_config["behaviors"]

        # Split whitelisted recordings into tuples of date and time.
        whitelist = [
            (path.split(os.sep)[0], path.split(os.sep)[1])
            for path in config["whitelist"]
        ]

        insts = label_config["labels"]
        random.seed(seed)

        train_insts = []
        test_insts = []

        # TODO: I really think this could be a lot simpler, but it works for now.

        # An 'inst_group' is an individual video file.
        # Each one has a number of labels for that file, each with a different behavior.
        # This dictionary stores for each group an index of all labels associated with a given behavior.
        inst_groups = {}

        total_insts = {b: 0 for b in behaviors}

        for b in behaviors:
            for inst in insts[b]:
                vid_name = os.path.split(inst["video"])[1]
                group = vid_name.split("_")[1]

                if group not in inst_groups:
                    inst_groups[group] = {b: [] for b in behaviors}

                inst_groups[group][b].append(inst)
                total_insts[b] += 1

        # This will store all the insts for a given behavior.
        behavior_insts = {b: [] for b in behaviors}

        # We don't want them in video file order, so we shuffle the order.
        video_files = list(inst_groups.keys())
        random.shuffle(video_files)

        for video_file in video_files:
            insts = inst_groups[video_file]
            for b in behaviors:
                behavior_insts[b].extend(insts[b])

        for b in behaviors:
            splt = int((1 - split) * len(behavior_insts[b]))
            train_insts.extend(behavior_insts[b][:splt])
            test_insts.extend(behavior_insts[b][splt:])

        random.shuffle(train_insts)
        random.shuffle(test_insts)

        train_seqs, train_labels = self.convert_instances(
            train_insts, seq_len=seq_len, behaviors=behaviors
        )
        test_seqs, test_labels = self.convert_instances(
            test_insts, seq_len=seq_len, behaviors=behaviors
        )

        return DatasetLoader(train_seqs, train_labels, behaviors), DatasetLoader(
            test_seqs, test_labels, behaviors
        )


def collate_fn(batch):
    dcls = [item[0] for item in batch]
    lbls = [item[1] for item in batch]

    dcls = torch.stack(dcls)
    lbls = torch.stack(lbls)

    return dcls, lbls


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class PerformanceReport:
    def __init__(self, sklearn_report, confusion_matrix):
        self.sklearn_report = sklearn_report
        self.confusion_matrix = confusion_matrix

def train_lstm_model(
    train_set,
    test_set,
    seq_len,
    behaviors,
    batch_size=512,
    lr=1e-4,
    epochs=10,
    device=None,
):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_f1 = 0 
    best_report = None
    best_epoch = 0
    best_model = None
    best_report_list = None

    for _ in range(1):
        model = classifier_head.classifier(
            in_features=768, out_features=len(behaviors), seq_len=seq_len
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        reports = []

        for e in range(epochs):
            for t, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = (epochs - e) / epochs * 0.0005 + (e / epochs) * 0.00001

            for i, (d, l) in enumerate(train_loader):

                d = d.to(device).float()
                l = l.to(device)

                optimizer.zero_grad()

                B = d.shape[0]

                lstm_logits, linear_logits, rawm = model(d)

                logits = lstm_logits + linear_logits

                inv_loss = criterion(logits, l)

                rawm = rawm - rawm.mean(dim=0)

                covm = (rawm @ rawm.T) / rawm.shape[0]
                covm_loss = torch.sum(torch.pow(off_diagonal(covm), 2)) / rawm.shape[1]

                loss = inv_loss + covm_loss

                loss.backward()
                optimizer.step()

                print(f"Epoch: {e} Batch: {i} Total Loss: {loss.item()}")
            
            actuals = []
            predictions = []
            
            for i, (d, l) in enumerate(test_loader):
                d = d.to(device).float()
                l = l.to(device)

                with torch.no_grad():
                    logits = model.forward_nodrop(d)

                actuals.extend(l.cpu().numpy())
                predictions.extend(logits.argmax(1).cpu().numpy())

            report_dict = classification_report(actuals, predictions, target_names=behaviors, output_dict=True)

            wf1score = report_dict['weighted avg']['f1-score']

            full_report = PerformanceReport(report_dict, confusion_matrix(actuals, predictions))
            reports.append(full_report)

            if best_f1 < wf1score:
                best_f1 = wf1score
                best_report = full_report
                best_report_list = reports
                best_epoch = e
                best_model = model
                
    return best_model, best_report_list, best_epoch

"""
# Library usage
print(f'Do we have CUDA: {torch.cuda.is_available()}')

#proj = Project.create_project('../', 'create_test')
proj = Project('../create_test')
proj.create_camera('cam1', 'rtsp://admin:scnscn@192.168.1.108:8554/profile0')
cam = proj.cameras[0]

print(cam)
cam.start_recording("test/", 600)
time.sleep(30)
cam.stop_recording()

print(cam)
#proj.encode_recordings()
#proj.infer_recordings()

#train_ds1, test_ds1 = proj.load_dataset('ds1')

#model, report= train_lstm_model(train_ds1, test_ds1, 15, train_ds1.behaviors)
"""

# encoder = DinoEncoder()
# print(encode_file(encoder, '../example_project/recordings/20241110\cam1-035459-PM\cam1_00000.mp4'))
