import os
import h5py

import time

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
    reader = decord.VideoReader(path, ctx=decord.cpu(0))
    frames = reader.get_batch(range(0, len(reader), 1)).asnumpy()
    frames = torch.from_numpy(frames[:, :, :, 1] / 255).half()

    # TODO: I am not fully convinced this is most efficient? How can we increase throughput.
    batch_size = 1024

    clss = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]

        with torch.no_grad() and torch.amp.autocast(encoder.device):
            out = encoder(batch.unsqueeze(1).to(encoder.device))

        out = out.squeeze(1).to("cpu")
        clss.extend(out)

    # Build the h5 file.
    out_file_path = os.path.splitext(path)[0] + "_cls.h5"
    with h5py.File(out_file_path, "w") as file:
        file.create_dataset("cls", data=torch.stack(clss).numpy())

    return out_file_path


class DinoEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super(DinoEncoder, self).__init__()
        self.device = device

        self.model = transformers.AutoModel.from_pretrained("facebook/dinov2-base").to(
            device
        )
        self.model.eval()

        # We obviously dont want to train DINO ourselves.
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
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


class Dataset(torch.utils.data.Dataset):
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
        self.unencodeded_files = []

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
                self.unencodeded_files.append(video_file)

    def encode(self, encoder: nn.Module):
        for file in self.unencodeded_files:
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
            output_file = file_path.replace(
                "_cls.h5", "_" + dataset_name + "_outputs.csv"
            )
            with h5py.File(file_path, "r") as file:
                cls = np.array(file["cls"][:])

            cls = torch.from_numpy(cls - np.mean(cls, axis=0)).half()

            predictions = []
            batch = []

            if len(cls) < seq_len:
                continue

            model = model.to(device)
            print(seq_len)

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


class Dataset:
    def __init__(self, path: str):
        self.path = path

        with open(os.path.join(path, "config.yaml")) as file:
            config = yaml.safe_load(file)


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
        self.recordings: dict[str, list[Recording]] = {}
        days = [f.path for f in os.scandir(self.recordings_dir) if f.is_dir()]

        for day in days:
            day_str = os.path.basename(day)

            subrecording_folders = [f.path for f in os.scandir(day)]
            self.recordings[day_str] = {}
            for folder in subrecording_folders:
                rec = Recording(folder)
                self.recordings[day_str][rec.name] = rec
            break

        # Build out our list of cameras
        self.cameras = {}
        for cam_path in [f.path for f in os.scandir(self.cameras_dir) if f.is_dir()]:
            with open(os.path.join(cam_path, "config.yaml")) as file:
                config = yaml.safe_load(file)

            self.cameras[config["name"]] = Camera(config, self)

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

    def load_dataset(
        self, name: str, seed=42, split=0.2, seq_len=15
    ) -> tuple[Dataset, Dataset]:
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

        return Dataset(train_seqs, train_labels, behaviors), Dataset(
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

    model = classifier_head.classifier(
        in_features=768, out_features=len(behaviors), seq_len=seq_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
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

    return model


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

#model = train_lstm_model(train_ds1, test_ds1, 15, train_ds1.behaviors)
"""

# encoder = DinoEncoder()
# print(encode_file(encoder, '../example_project/recordings/20241110\cam1-035459-PM\cam1_00000.mp4'))
