import cbas
import gui_state

import threading
import ctypes

import eel

from cmap import Colormap

import yaml
import base64
import os
import cv2
import numpy as np

import workthreads

@eel.expose
def load_dataset_configs():
    return {name: gui_state.proj.datasets[name].config for name in gui_state.proj.datasets }

@eel.expose
def handle_click_on_label_image(x, y):
    amount_of_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    gui_state.label_index = int(x * amount_of_frames / 500)

    render_image()

def tab20_map(val):
    if val < 10:
        return val * 2
    else:
        return (val - 10) * 2 + 1

@eel.expose
def next_video(shift):
    gui_state.label_start = -1
    gui_state.label_vid_index = gui_state.label_vid_index + shift
    gui_state.label_type = -1
    gui_state.label_index = -1

    video = gui_state.label_videos[gui_state.label_vid_index % len(gui_state.label_videos)]

    capture = cv2.VideoCapture(video)

    if not capture.isOpened():

        recovered = False

        for i in range(len(gui_state.label_videos)):
            gui_state.label_vid_index += shift
            video = gui_state.label_videos[gui_state.label_vid_index % len(gui_state.label_videos)]
            capture = cv2.VideoCapture(video)

            if capture.isOpened():
                recovered = True
                break
        
        if not recovered:
            raise Exception("No valid videos in the dataset.")

    gui_state.label_capture = capture
    next_frame(1)

@eel.expose
def next_frame(shift):
    if shift <= 0:
        shift -= 1

    if gui_state.label_capture.isOpened():
        amount_of_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        gui_state.label_index += shift
        gui_state.label_index %= amount_of_frames

        render_image()

@eel.expose
def start_labeling(name):
    gui_state.label_capture = None
    gui_state.label_index = -1
    gui_state.label_videos = []
    gui_state.label_vid_index = -1
    gui_state.label_type = -1
    gui_state.label_start = -1
    gui_state.label_history = []

    dataset: cbas.Dataset = gui_state.proj.datasets[name]

    gui_state.label_dataset = dataset

    whitelist = dataset.config["whitelist"]

    gui_state.label_col_map = Colormap("seaborn:tab20")

    # Make a list of all the videos, we will then filter for whitelist
    all_videos = []

    sub_dirs = [
        os.path.join(gui_state.proj.recordings_dir, d) 
        for d in os.listdir(gui_state.proj.recordings_dir) 
        if os.path.isdir(os.path.join(gui_state.proj.recordings_dir, d))
    ]

    if len(sub_dirs) == 0:
        return False

    for sub_dir in sub_dirs:
        sub_sub_dirs = [
            os.path.join(sub_dir, d) 
            for d in os.listdir(sub_dir) 
            if os.path.isdir(os.path.join(sub_dir, d))
        ]

        for sub_sub_dir in sub_sub_dirs:
            all_videos.extend(
                [
                    os.path.join(sub_sub_dir, v)
                    for v in os.listdir(sub_sub_dir)
                    if v.endswith(".mp4")
                ]
            )
    
    # Filter for whitelist videos
    valid_videos = []
    for v in all_videos:
        for wl in whitelist:
            if wl in v:
                valid_videos.append(v)
    
    if len(valid_videos) == 0:
        return False

    gui_state.label_videos = valid_videos 

    # Set the video to be the first.
    next_video(1)

    update_counts()

    return dataset.labels["behaviors"], [
        str(gui_state.label_col_map(tab20_map(i))) for i in range(len(dataset.labels["behaviors"]))
    ]


@eel.expose
def label_frame(value):
    behaviors = gui_state.label_dataset.labels["behaviors"]

    beh = None
    ind = None

    for b in behaviors:
        if beh != None or ind != None:
            break

        for i, inst in enumerate(gui_state.label_dataset.labels["labels"][b]):
            if inst["start"] <= gui_state.label_index <= inst["end"] and inst["video"] == gui_state.label_videos[gui_state.label_vid_index]:
                beh = b
                ind = i
                break

    # Change the behavior type of an instance
    if beh is not None and gui_state.label_type == -1:
        old_instance = gui_state.label_dataset.labels["labels"][beh][ind]

        new_beh = behaviors[value]

        new_instance = {
            "video": gui_state.label_videos[gui_state.label_vid_index],
            "start": old_instance["start"],
            "end": old_instance["end"],
            "label": new_beh,
        }

        gui_state.label_dataset.labels["labels"][new_beh].append(new_instance)

        del gui_state.label_dataset.labels["labels"][beh][ind]

        # save the label dictionary
        with open(gui_state.label_dataset.labels_path, "w+") as file:
            yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)

        update_counts()
        render_image()

        return

    if value >= len(behaviors):
        return False

    if value == gui_state.label_type:
        add_instance()
        gui_state.label_type = -1
    elif gui_state.label_type == -1:
        gui_state.label_type = value
        gui_state.label_start = gui_state.label_index
    else:
        gui_state.label_type = -1

def add_instance():
    stemp = min(gui_state.label_start, gui_state.label_index)
    eInd = max(gui_state.label_start, gui_state.label_index)
    sInd = stemp

    # check for collisions
    labels = gui_state.label_dataset.labels["labels"]
    behaviors = gui_state.label_dataset.labels["behaviors"]

    for i, b in enumerate(behaviors):
        sub_labels = labels[b]

        for l in sub_labels:
            if l["video"] == gui_state.label_videos[gui_state.label_vid_index]:
                if sInd < l["end"] and sInd > l["start"]:
                    gui_state.label_type = -1
                    raise Exception(
                        "Overlapping behavior region! Behavior not recorded."
                    )
                elif eInd < l["end"] and eInd > l["start"]:
                    gui_state.label_type = -1
                    raise Exception(
                        "Overlapping behavior region! Behavior not recorded."
                    )

    behavior = gui_state.label_dataset.labels["behaviors"][gui_state.label_type]

    instance = {
        "video": gui_state.label_videos[gui_state.label_vid_index],
        "start": sInd,
        "end": eInd,
        "label": behavior,
    }

    gui_state.label_dataset.labels["labels"][behavior].append(instance)

    gui_state.label_history.append(instance)

    # save the label dictionary
    with open(gui_state.label_dataset.labels_path, "w+") as file:
        yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)

@eel.expose
def update_counts():
    config_path = os.path.join(os.path.split(gui_state.label_dataset.labels_path)[0], "config.yaml")

    for b in gui_state.label_dataset.labels["behaviors"]:
        total_frames = 0
        insts = gui_state.label_dataset.labels["labels"][b]
        for inst in insts:
            total_frames += inst["end"] - inst["start"]

        gui_state.label_dataset.update_metric(b, "Train #", int(round(len(insts) * 0.75)))
        gui_state.label_dataset.update_metric(b, "Test #", int(round(len(insts) * 0.25)))
        gui_state.label_dataset.update_metric(b, "Train Frames", total_frames * 0.75)
        gui_state.label_dataset.update_metric(b, "Test Frames", total_frames * 0.25)

        eel.updateCount(b, len(insts), total_frames)()

    filename = gui_state.label_videos[gui_state.label_vid_index]

    eel.updateFileInfo(os.path.relpath(filename, start=gui_state.proj.path))()

@eel.expose
def delete_instance():
    beh = None
    ind = None

    for b in gui_state.label_dataset.labels["behaviors"]:
        if beh != None or ind != None:
            break

        for i, inst in enumerate(gui_state.label_dataset.labels["labels"][b]):
            if inst["start"] <= gui_state.label_index <= inst["end"]:
                beh = b
                ind = i
                break

    if beh == None or ind == None:
        return
    else:
        del gui_state.label_dataset.labels["labels"][beh][ind]

    # save the label dictionary
    with open(gui_state.label_dataset.labels_path, "w+") as file:
        yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)

    update_counts()

    render_image()


@eel.expose
def pop_instance():

    if len(gui_state.label_history) == 0:
        return

    last_inst = gui_state.label_history[-1]

    beh = None
    ind = None

    for b in gui_state.label_dataset.labels["behaviors"]:
        for i, inst in enumerate(gui_state.label_dataset.labels["labels"][b]):
            if (
                inst["video"] == last_inst["video"]
                and inst["start"] == last_inst["start"]
                and inst["end"] == last_inst["end"]
            ):
                beh = b
                ind = i
    if beh == None or ind == None:
        return
    else:
        del gui_state.label_dataset.labels["labels"][beh][ind]

    # save the label dictionary
    with open(gui_state.label_dataset.labels_path, "w+") as file:
        yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)

    update_counts()

    render_image()

def render_image():
    amount_of_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    gui_state.label_capture.set(cv2.CAP_PROP_POS_FRAMES, gui_state.label_index)

    ret, frame = gui_state.label_capture.read()

    if ret:
        frame = cv2.resize(frame, (500, 500))

        temp = np.zeros((frame.shape[0] + 50, frame.shape[1], frame.shape[2]))

        temp[:-50, :, :] = frame

        temp[-50, :, :] = 0

        temp[-49:, :, :] = 100

        temp = fill_colors(temp)

        marker_pos = int(frame.shape[1] * gui_state.label_index / amount_of_frames)

        if marker_pos != 0 and marker_pos != frame.shape[1] - 1:
            temp[-45:-5, marker_pos - 1 : marker_pos + 2, :] = 255
        else:
            if marker_pos == 0:
                temp[-45:-5, marker_pos : marker_pos + 2, :] = 255
            else:
                temp[-45:-5, marker_pos - 1 : marker_pos + 1, :] = 255

        ret, frame = cv2.imencode(".jpg", temp)

        frame = frame.tobytes()

        blob = base64.b64encode(frame)
        blob = blob.decode("utf-8")

        eel.updateLabelImageSrc(blob)()

def fill_colors(frame):
    behaviors = gui_state.label_dataset.labels["behaviors"]

    labels = gui_state.label_dataset.labels["labels"]

    cur_video = gui_state.label_videos[gui_state.label_vid_index]
    amount_of_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    for i, b in enumerate(behaviors):

        sub_labels = labels[b]

        color = str(gui_state.label_col_map(tab20_map(i))).lstrip("#")
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

    if gui_state.label_type != -1:
        color = str(gui_state.label_col_map(tab20_map(gui_state.label_type))).lstrip("#")
        color = np.flip(np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)]))

        stemp = min(gui_state.label_start, gui_state.label_index)
        eInd = max(gui_state.label_start, gui_state.label_index)

        sInd = stemp

        marker_posS = int(frame.shape[1] * sInd / amount_of_frames)
        marker_posE = int(frame.shape[1] * eInd / amount_of_frames)

        frame[
            -49:,
            marker_posS : marker_posE + 1,
        ] = color

    return frame

@eel.expose
def get_record_tree():
    rt = {}

    sub_dirs = [
        d
        for d in os.listdir(gui_state.proj.recordings_dir)
        if os.path.isdir(os.path.join(gui_state.proj.recordings_dir, d))
    ]

    if len(sub_dirs) == 0:
        return False
    else:
        rt = {sd: [] for sd in sub_dirs}

        for sd in sub_dirs:

            sub_sub_dirs = [
                d
                for d in os.listdir(os.path.join(gui_state.proj.recordings_dir, sd))
                if os.path.isdir(os.path.join(gui_state.proj.recordings_dir, sd, d))
            ]

            rt[sd] = sub_sub_dirs

    return rt

@eel.expose
def create_dataset(name, behaviors, recordings):
    rt = get_record_tree()

    if not rt:
        return False

    dataset = gui_state.proj.create_dataset(name, behaviors, recordings)
    gui_state.label_dataset = dataset

    return True

@eel.expose
def train_model(name, batch_size, learning_rate, epochs, sequence_length):
    dataset = gui_state.proj.datasets[name]

    task = workthreads.TrainingTask(name, dataset, dataset.config['behaviors'], int(batch_size), float(learning_rate), int(epochs), int(sequence_length))
    gui_state.training_thread.queue_task(task)

@eel.expose
def start_classification(dataset, whitelist):
    if not gui_state.classify_thread.running:
        gui_state.classify_thread.start_inferring(gui_state.proj.models[dataset], whitelist)

        gui_state.classify_lock.acquire()
        for path in whitelist:
            if len(path.split(os.path.sep)) == 1:
                for subdir in os.listdir(os.path.join(gui_state.proj.recordings_dir, path)):
                    for video_file in os.listdir(os.path.join(gui_state.proj.recordings_dir, path, subdir)):
                        if ".h5" in video_file:
                            gui_state.classify_tasks.append(os.path.join(gui_state.proj.recordings_dir, path, subdir, video_file))
            else:
                for video_file in os.listdir(os.path.join(gui_state.proj.recordings_dir, path)):
                    if ".h5" in video_file:
                        gui_state.classify_tasks.append(os.path.join(gui_state.proj.recordings_dir, path, video_file))
        gui_state.classify_lock.release()

    #if model_path and os.path.exists(model_path):
    if True:
        """
        path = gui_state.proj.recordings_dir  # Replace with the path you want to watch
        event_handler = VideoFileWatcher(whitelist)
        observer = Observer()

        gui_state.active_classifications.append((whitelist, observer))

        observer.schedule(event_handler, path, recursive=True)
        observer.start()
        existing_files =[]
        for path in whitelist:
            paths = path.split(os.path.sep)
            if len(paths) == 1:
                for subdir in os.listdir(os.path.join(gui_state.proj.recordings_dir, path)):
                    for video_file in os.listdir(os.path.join(gui_state.proj.recordings_dir, path, subdir)):
                        existing_files.append(os.path.join(gui_state.proj.recordings_dir, path, subdir, video_file))
            elif len(paths) == 2:
                for video_file in os.listdir(os.path.join(gui_state.proj.recordings_dir, path)):
                    existing_files.append(os.path.join(gui_state.proj.recordings_dir, path, video_file))

        gui_state.class_lock.acquire()
        for file in existing_files:
            print(file)
        gui_state.class_lock.release()
        """