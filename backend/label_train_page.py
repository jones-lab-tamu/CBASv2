"""
Manages all backend logic for the 'Label/Train' page of the CBAS application.

This includes:
- Handling the labeling interface state (loading videos, drawing frames).
- Managing the session buffer for in-memory label editing.
- Saving corrected labels back to file.
- Providing data for the UI (dataset configs, video lists, etc.).
- Launching and managing training and inference tasks via the workthreads module.
"""

import os
import base64
import traceback
import yaml
import cv2
import numpy as np
import torch
import pandas as pd

# Project-specific imports
import cbas
import classifier_head
import gui_state
import workthreads
from cmap import Colormap

import eel


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def color_distance(rgb1, rgb2):
    """Calculates the perceived distance between two RGB colors for contrast checking."""
    rmean = (rgb1[0] + rgb2[0]) // 2
    r = rgb1[0] - rgb2[0]
    g = rgb1[1] - rgb2[1]
    b = rgb1[2] - rgb2[2]
    return (((512 + rmean) * r * r) >> 8) + (4 * g * g) + (((767 - rmean) * b * b) >> 8)


def hex_to_rgb(hex_color):
    """Converts a hex color string (e.g., '#RRGGBB') to an (R, G, B) tuple."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def tab20_map(val: int) -> int:
    """
    Maps an integer to an index for the 'seaborn:tab20' colormap.
    This version includes manual overrides for known problematic (grey/brown) colors
    to ensure good UI contrast against the timeline background.
    """
    remap = {7: 6, 14: 2, 15: 4}  # Remap grey/brown colors to vibrant ones
    if val in remap:
        return remap[val]
    # Default distinct color mapping logic
    return (val * 2) if val < 10 else ((val - 10) * 2 + 1)


# =================================================================
# EEL-EXPOSED FUNCTIONS: DATA & CONFIGURATION
# =================================================================

@eel.expose
def model_exists(model_name: str) -> bool:
    """Checks if a model with the given name exists in the current project."""
    return gui_state.proj and model_name in gui_state.proj.models


@eel.expose
def load_dataset_configs() -> dict:
    """Loads configurations for all available datasets."""
    if not gui_state.proj: return {}
    return {name: dataset.config for name, dataset in gui_state.proj.datasets.items()}


@eel.expose
def get_available_models() -> list[str]:
    """Returns a sorted list of all model names available in the project."""
    if not gui_state.proj: return []
    return sorted(list(gui_state.proj.models.keys()))


@eel.expose
def get_record_tree() -> dict:
    """Fetches the recording directory tree structure for modal dialogs."""
    if not gui_state.proj or not os.path.exists(gui_state.proj.recordings_dir):
        return {}
    tree = {}
    for date_dir in os.scandir(gui_state.proj.recordings_dir):
        if date_dir.is_dir():
            tree[date_dir.name] = [
                session.name for session in os.scandir(date_dir.path) if session.is_dir()
            ]
    return tree


@eel.expose
def get_videos_for_dataset(dataset_name: str) -> list[tuple[str, str]]:
    """Finds all .mp4 files within a dataset's whitelist for 'Label from Scratch' mode."""
    if not gui_state.proj: return []
    dataset = gui_state.proj.datasets.get(dataset_name)
    if not dataset: return []
    
    whitelist = dataset.config.get("whitelist", [])
    if not whitelist: return []

    video_list = []
    # This logic can be simplified in the future, but works for now.
    if gui_state.proj.recordings_dir and os.path.exists(gui_state.proj.recordings_dir):
        for root, _, files in os.walk(gui_state.proj.recordings_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    normalized_path = os.path.normpath(video_path)
                    if any(os.path.normpath(p) in normalized_path for p in whitelist):
                        display_name = os.path.relpath(video_path, gui_state.proj.recordings_dir)
                        video_list.append((video_path, display_name))
    
    return sorted(video_list, key=lambda x: x[1])


@eel.expose
def get_inferred_session_dirs(dataset_name: str, model_name: str) -> list[str]:
    """Finds unique sub-directories that contain videos inferred by a specific model."""
    if not gui_state.proj: return []
    dataset = gui_state.proj.datasets.get(dataset_name)
    if not dataset: return []
    
    whitelist_paths = [os.path.abspath(os.path.join(gui_state.proj.recordings_dir, p)) for p in dataset.config.get("whitelist", [])]
    if not whitelist_paths: return []

    inferred_dirs = set()
    for root, _, files in os.walk(gui_state.proj.recordings_dir):
        for file in files:
            if file.endswith(f"_{model_name}_outputs.csv"):
                csv_abs_path = os.path.abspath(os.path.join(root, file))
                if any(csv_abs_path.startswith(wl_path) for wl_path in whitelist_paths):
                    inferred_dirs.add(os.path.relpath(root, gui_state.proj.recordings_dir))
    
    return sorted(list(inferred_dirs))


@eel.expose
def get_inferred_videos_for_session(session_dir_rel: str, model_name: str) -> list[tuple[str, str]]:
    """Gets a list of inferred videos from a single specified session directory."""
    if not gui_state.proj: return []
    session_abs_path = os.path.join(gui_state.proj.recordings_dir, session_dir_rel)
    if not os.path.isdir(session_abs_path): return []
    
    ready_videos = []
    for file in os.listdir(session_abs_path):
        if file.endswith(f"_{model_name}_outputs.csv"):
            csv_path = os.path.join(session_abs_path, file)
            mp4_path = csv_path.replace(f"_{model_name}_outputs.csv", ".mp4")
            if os.path.exists(mp4_path):
                ready_videos.append((mp4_path, os.path.basename(mp4_path)))
                
    return sorted(ready_videos, key=lambda x: x[1])


# =================================================================
# EEL-EXPOSED FUNCTIONS: LABELING WORKFLOW & ACTIONS
# =================================================================

def _start_labeling_worker(name: str, video_to_open: str = None, preloaded_instances: list = None, probability_df: pd.DataFrame = None):
    """
    (WORKER) This is the background task that prepares the labeling session. It handles
    all state setup and then calls back to the JavaScript UI when ready.
    """
    try:
        # 1. Validate State and Arguments
        print("Labeling worker started.")
        if gui_state.proj is None: raise ValueError("Project is not loaded.")
        if name not in gui_state.proj.datasets: raise ValueError(f"Dataset '{name}' not found.")
        if not video_to_open or not os.path.exists(video_to_open):
            raise FileNotFoundError(f"Video to label does not exist: {video_to_open}")

        # 2. Reset Global Labeling State
        print("Resetting labeling state for new session.")
        if gui_state.label_capture and gui_state.label_capture.isOpened():
            gui_state.label_capture.release()
        
        gui_state.label_capture, gui_state.label_index = None, -1
        gui_state.label_videos, gui_state.label_vid_index = [], -1
        gui_state.label_type, gui_state.label_start = -1, -1
        gui_state.label_history, gui_state.label_behavior_colors = [], []
        gui_state.label_session_buffer, gui_state.selected_instance_index = [], -1
        gui_state.label_probability_df = probability_df
        # Ensure confirmation mode is always turned off when starting a new session.
        gui_state.label_confirmation_mode = False
        
        # 3. Set Up Dataset and Colormap
        dataset: cbas.Dataset = gui_state.proj.datasets[name]
        gui_state.label_dataset = dataset
        gui_state.label_col_map = Colormap("seaborn:tab20")

        # 4. Prepare the In-Memory Session Buffer
        print("Loading session buffer...")
        gui_state.label_videos = [video_to_open]
        current_video_path = gui_state.label_videos[0]

        for b_name, b_insts in gui_state.label_dataset.labels["labels"].items():
            for inst in b_insts:
                if inst.get("video") == current_video_path:
                    gui_state.label_session_buffer.append(inst)
        print(f"Loaded {len(gui_state.label_session_buffer)} existing human labels into buffer.")

        labeling_mode = 'scratch'
        model_name_for_ui = ''
        if preloaded_instances:
            labeling_mode = 'review'
            model_name_for_ui = name
            print(f"Applying {len(preloaded_instances)} pre-loaded instances.")
            for pred_inst in preloaded_instances:
                is_overlapping = any(max(pred_inst['start'], h['start']) <= min(pred_inst['end'], h['end']) for h in gui_state.label_session_buffer)
                if not is_overlapping:
                    gui_state.label_session_buffer.append(pred_inst)
        
        # 5. Generate and Store Corrected Colors
        dataset_behaviors = gui_state.label_dataset.labels.get("behaviors", [])
        behavior_colors = [str(gui_state.label_col_map(tab20_map(i))) for i in range(len(dataset_behaviors))]
        gui_state.label_behavior_colors = behavior_colors

        # 6. Call Frontend to Build the UI
        print("Setup complete. Calling frontend to build UI.")
        eel.buildLabelingUI(dataset_behaviors, behavior_colors)()
        eel.setLabelingModeUI(labeling_mode, model_name_for_ui)()
        eel.setConfirmationModeUI(False)()

        # 7. Load Video and Push Initial Frame
        if not gui_state.label_videos: raise ValueError("Video list is empty after setup.")
        print("Loading video and pushing initial frame...")
        next_video(0)

    except Exception as e:
        print(f"FATAL ERROR in labeling worker: {e}")
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(f"Failed to start labeling session: {e}")()


@eel.expose
def start_labeling(name: str, video_to_open: str = None, preloaded_instances: list = None) -> bool:
    """
    (LAUNCHER) Lightweight function to spawn the labeling worker in the background.
    Returns True immediately to unblock the JavaScript UI.
    """
    try:
        eel.spawn(_start_labeling_worker, name, video_to_open, preloaded_instances, None)
        print(f"Spawned labeling worker for dataset '{name}'.")
        return True
    except Exception as e:
        print(f"Failed to spawn labeling worker: {e}")
        return False


@eel.expose
def start_labeling_with_preload(dataset_name: str, model_name: str, video_path_to_label: str) -> bool:
    """
    (LAUNCHER) Runs a quick inference step and then spawns the labeling worker.
    """
    try:
        print(f"Request to pre-label '{os.path.basename(video_path_to_label)}' with model '{model_name}'...")
        if gui_state.proj is None: raise ValueError("Project not loaded")
        dataset = gui_state.proj.datasets.get(dataset_name)
        model_obj = gui_state.proj.models.get(model_name)
        if not dataset or not model_obj: raise ValueError("Dataset or Model not found.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = classifier_head.classifier(
            in_features=768,
            out_features=len(model_obj.config["behaviors"]),
            seq_len=model_obj.config["seq_len"],
        ).to(device)
        torch_model.load_state_dict(torch.load(model_obj.weights_path, map_location=device, weights_only=True))
        torch_model.eval()

        h5_path = video_path_to_label.replace(".mp4", "_cls.h5")
        if not os.path.exists(h5_path): raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        csv_path = cbas.infer_file(
            file_path=h5_path, model=torch_model, dataset_name=model_obj.name,
            behaviors=model_obj.config["behaviors"], seq_len=model_obj.config["seq_len"], device=device
        )
        if not csv_path or not os.path.exists(csv_path):
            raise RuntimeError("Inference failed to produce a CSV output file.")

        preloaded_instances, probability_df = dataset.predictions_to_instances_with_confidence(csv_path, model_obj.name)
        
        eel.spawn(_start_labeling_worker, dataset_name, video_path_to_label, preloaded_instances, probability_df)
        
        print(f"Spawned pre-labeling worker for video '{os.path.basename(video_path_to_label)}'.")
        return True

    except Exception as e:
        print(f"ERROR in start_labeling_with_preload: {e}")
        eel.showErrorOnLabelTrainPage(f"Failed to start pre-labeling: {e}")()
        return False


@eel.expose
def save_session_labels():
    """
    Filters for only human-verified/confirmed instances and saves them.
    """
    if not gui_state.label_dataset or not gui_state.label_videos: return

    final_labels_to_save = []
    for inst in gui_state.label_session_buffer:
        # Save an instance if it's human-added OR if it's been confirmed with Enter.
        if 'confidence' not in inst or inst.get('_confirmed', False):
            final_inst = inst.copy()
            # Clean up all temporary keys before saving
            for key in ['confidence', 'confidences', '_original_start', '_original_end', '_confirmed']:
                final_inst.pop(key, None)
            final_labels_to_save.append(final_inst)

    current_video_path = gui_state.label_videos[0]
    all_labels = gui_state.label_dataset.labels["labels"]

    # Remove all old labels for the current video
    for behavior_name in all_labels:
        all_labels[behavior_name][:] = [
            inst for inst in all_labels[behavior_name] if inst.get("video") != current_video_path
        ]
    
    # Add the new, verified-only labels back
    for corrected_inst in final_labels_to_save:
        all_labels.setdefault(corrected_inst['label'], []).append(corrected_inst)
    
    with open(gui_state.label_dataset.labels_path, "w") as file:
        yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)
    
    print(f"Saved {len(final_labels_to_save)} human-verified/corrected labels for video {os.path.basename(current_video_path)}.")
    
    # Reset the confirmation mode after saving
    gui_state.label_confirmation_mode = False
    # Tell the UI to reset the buttons
    eel.setConfirmationModeUI(False)()
    # Trigger a final re-render to show the clean state
    render_image()


# =================================================================
# EEL-EXPOSED FUNCTIONS: IN-SESSION LABELING ACTIONS
# =================================================================

@eel.expose
def confirm_selected_instance():
    """
    Toggles the 'confirmed' state of the currently selected instance.
    If un-confirming, it also reverts any boundary edits to their original state.
    """
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        
        # Check the current state of the instance
        is_currently_confirmed = instance.get('_confirmed', False)

        if is_currently_confirmed:
            # --- UNLOCK LOGIC ---
            # It's already confirmed, so un-confirm it and revert changes.
            instance['_confirmed'] = False
            
            # Revert to original boundaries if they exist
            if '_original_start' in instance:
                instance['start'] = instance['_original_start']
                instance['end'] = instance['_original_end']
            
            print(f"Unlocked instance {gui_state.selected_instance_index} and reverted changes.")
        
        else:
            # --- LOCK/CONFIRM LOGIC ---
            # It's not confirmed, so confirm it.
            instance['_confirmed'] = True
            print(f"Confirmed instance {gui_state.selected_instance_index}.")
            
        # Re-render to show the change in visual state (border color, boundaries)
        render_image()

@eel.expose
def handle_click_on_label_image(x: int, y: int):
    """Handles a click on the timeline to scrub to a specific frame."""
    if gui_state.label_capture and gui_state.label_capture.isOpened():
        total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames > 0:
            gui_state.label_index = int(x * total_frames / 500)
            render_image()


@eel.expose
def next_video(shift: int):
    """Loads the next or previous video in the labeling session's video list."""
    if not gui_state.label_videos:
        eel.updateLabelImageSrc(None, None, None); eel.updateFileInfo("No videos available."); return

    gui_state.label_start, gui_state.label_type = -1, -1
    gui_state.label_vid_index = (gui_state.label_vid_index + shift) % len(gui_state.label_videos)
    current_video_path = gui_state.label_videos[gui_state.label_vid_index]

    if gui_state.label_capture and gui_state.label_capture.isOpened():
        gui_state.label_capture.release()
    
    print(f"Loading video for labeling: {current_video_path}")
    capture = cv2.VideoCapture(current_video_path)

    if not capture.isOpened():
        eel.updateLabelImageSrc(None, None, None); eel.updateFileInfo(f"Error loading video."); gui_state.label_capture = None; return

    gui_state.label_capture = capture
    gui_state.label_index = 0
    render_image()
    update_counts()


@eel.expose
def next_frame(shift: int):
    """
    Moves forward or backward by a number of frames. If an instance is selected,
    this movement is constrained to within the boundaries of that instance.
    """
    if not (gui_state.label_capture and gui_state.label_capture.isOpened()):
        return

    new_index = gui_state.label_index + shift
    
    # Check if we are in "Inspection Mode" (an instance is selected)
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        start_frame = instance.get("start", 0)
        end_frame = instance.get("end", 0)
        
        # Constrain the new index to the boundaries of the selected instance
        gui_state.label_index = max(start_frame, min(new_index, end_frame))

    else:
        # Otherwise, behave as normal (free scrolling)
        total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames > 0:
            gui_state.label_index = new_index % total_frames
            
    render_image()


@eel.expose
def jump_to_instance(direction: int):
    """Finds the next/previous instance and jumps the playhead. Clears any active draft."""
    if not gui_state.label_session_buffer:
        eel.highlightBehaviorRow(None)()
        eel.updateConfidenceBadge(None, None)() # Clear badge if no instances
        return

    sorted_instances = sorted(enumerate(gui_state.label_session_buffer), key=lambda item: item[1]['start'])
    if not sorted_instances:
        eel.highlightBehaviorRow(None)()
        eel.updateConfidenceBadge(None, None)() # Clear badge
        return

    current_frame = gui_state.label_index
    target_item = None

    if direction > 0:
        found = next((item for item in sorted_instances if item[1]['start'] > current_frame), None)
        target_item = found or sorted_instances[0]
    else:
        found = next((item for item in reversed(sorted_instances) if item[1]['start'] < current_frame), None)
        target_item = found or sorted_instances[-1]
            
    if target_item:
        original_index, instance_data = target_item
        gui_state.label_index = instance_data['start']
        gui_state.selected_instance_index = original_index
        
        confidence = instance_data.get('confidence')
        eel.updateConfidenceBadge(instance_data['label'], confidence)()

        eel.highlightBehaviorRow(instance_data['label'])()
        render_image()
    else:
        eel.highlightBehaviorRow(None)()
        eel.updateConfidenceBadge(None, None)()


@eel.expose
def update_instance_boundary(boundary_type: str):
    """
    Directly updates the start or end frame of the currently selected instance
    after validating the change. Stores original boundaries on first edit.
    """
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        current_frame = gui_state.label_index

        # On first edit, store the original boundaries permanently.
        if '_original_start' not in instance:
            instance['_original_start'] = instance['start']
            instance['_original_end'] = instance['end']
            instance['_confirmed'] = False # Mark as edited but not yet confirmed

        if boundary_type == 'start' and current_frame < instance['end']:
            instance['start'] = current_frame
        elif boundary_type == 'end' and current_frame > instance['start']:
            instance['end'] = current_frame
        
        render_image()


@eel.expose
def get_zoom_range_for_click(x_pos: int) -> int:
    """Calculates a new frame index based on a click on the zoom bar."""
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Use original boundaries for context if they exist, otherwise current ones.
        original_start = instance.get('_original_start', instance['start'])
        original_end = instance.get('_original_end', instance['end'])
        
        inst_len = original_end - original_start
        context = inst_len * 2
        zoom_start = max(0, original_start - context)
        zoom_end = min(total_frames, original_end + context)
        
        if zoom_end > zoom_start:
            # Calculate what frame corresponds to the click position (x_pos out of 500)
            new_frame = int(zoom_start + (x_pos / 500.0) * (zoom_end - zoom_start))
            gui_state.label_index = new_frame
            render_image()


def add_instance_to_buffer():
    """Helper function to create a new label instance and add it to the session buffer."""
    if gui_state.label_type == -1 or gui_state.label_start == -1: return

    start_idx = min(gui_state.label_start, gui_state.label_index)
    end_idx = max(gui_state.label_start, gui_state.label_index)
    if start_idx == end_idx: return # Avoid zero-length labels

    # Collision check with existing labels in the buffer
    for inst in gui_state.label_session_buffer:
        if max(start_idx, inst['start']) <= min(end_idx, inst['end']):
            print(f"Collision detected. New label ({start_idx}-{end_idx}) overlaps with existing.")
            eel.showErrorOnLabelTrainPage("Overlapping behavior region! Behavior not recorded.")
            return

    behavior_name = gui_state.label_dataset.labels["behaviors"][gui_state.label_type]
    new_instance = {
        "video": gui_state.label_videos[gui_state.label_vid_index],
        "start": start_idx, "end": end_idx, "label": behavior_name,
    }
    gui_state.label_session_buffer.append(new_instance)
    gui_state.label_history.append(new_instance)
    update_counts()


@eel.expose
def label_frame(value: int):
    """Handles user keypresses to start, end, or change labels."""
    if gui_state.label_dataset is None or not gui_state.label_videos: return
    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    if not 0 <= value < len(behaviors): return

    # Check if we're clicking on an existing label to change it
    clicked_instance_index = -1
    for i, inst in enumerate(gui_state.label_session_buffer):
        if inst.get("start", -1) <= gui_state.label_index <= inst.get("end", -1):
            clicked_instance_index = i
            break
            
    if clicked_instance_index != -1 and gui_state.label_type == -1:
        # Change the type of an existing label in the buffer
        new_behavior_name = behaviors[value]
        gui_state.label_session_buffer[clicked_instance_index]['label'] = new_behavior_name
        print(f"Changed instance {clicked_instance_index} to '{new_behavior_name}'.")
    else:
        # Standard start/end labeling logic
        if value == gui_state.label_type: # End current labeling
            add_instance_to_buffer()
            gui_state.label_type = -1; gui_state.label_start = -1
        elif gui_state.label_type == -1: # Start new labeling
            gui_state.label_type, gui_state.label_start = value, gui_state.label_index
            gui_state.selected_instance_index = -1
            eel.updateConfidenceBadge(None, None)()
        else: # Switch active label type
            gui_state.label_type, gui_state.label_start = value, gui_state.label_index
            eel.updateConfidenceBadge(None, None)()
            
    render_image()


@eel.expose
def delete_instance_from_buffer():
    """Finds and removes an instance from the session buffer at the current frame."""
    if not gui_state.label_session_buffer: return
    current_frame = gui_state.label_index
    idx_to_remove = -1
    for i, inst in enumerate(gui_state.label_session_buffer):
        if inst.get("start", -1) <= current_frame <= inst.get("end", -1):
            idx_to_remove = i
            break
    if idx_to_remove != -1:
        removed_inst = gui_state.label_session_buffer.pop(idx_to_remove)
        if removed_inst in gui_state.label_history:
            gui_state.label_history.remove(removed_inst)
        gui_state.selected_instance_index = -1
        eel.updateConfidenceBadge(None, None)()
        render_image(); update_counts()


@eel.expose
def pop_instance_from_buffer():
    """Undoes the last-added instance from the session buffer."""
    if not gui_state.label_history: return
    last_added = gui_state.label_history.pop()
    try:
        gui_state.label_session_buffer.remove(last_added)
        gui_state.selected_instance_index = -1
        render_image(); update_counts()
    except ValueError:
        print(f"Could not pop {last_added}, not found in buffer.")

@eel.expose
def stage_for_commit():
    """Enters confirmation mode and triggers a re-render showing only staged labels."""
    gui_state.label_confirmation_mode = True
    eel.setConfirmationModeUI(True)
    render_image()

@eel.expose
def cancel_commit_stage():
    """Exits confirmation mode and triggers a re-render of the normal view."""
    gui_state.label_confirmation_mode = False
    eel.setConfirmationModeUI(False)
    render_image()

# =================================================================
# EEL-EXPOSED FUNCTIONS: DATASET & MODEL MANAGEMENT
# =================================================================

@eel.expose
def create_dataset(name: str, behaviors: list[str], recordings_whitelist: list[str]) -> bool:
    """Creates a new dataset via the project interface."""
    if not gui_state.proj: return False
    dataset = gui_state.proj.create_dataset(name, behaviors, recordings_whitelist)
    if dataset:
        gui_state.label_dataset = dataset
        return True
    return False


@eel.expose
def train_model(name: str, batch_size: str, learning_rate: str, epochs: str, sequence_length: str, training_method: str):
    """Queues a training task for the specified dataset."""
    if not gui_state.proj or name not in gui_state.proj.datasets: return
    if not gui_state.training_thread: return

    try:
        task = workthreads.TrainingTask(
            name=name, dataset=gui_state.proj.datasets[name],
            behaviors=gui_state.proj.datasets[name].config.get('behaviors', []),
            batch_size=int(batch_size), learning_rate=float(learning_rate),
            epochs=int(epochs), sequence_length=int(sequence_length),
            training_method=training_method
        )
        gui_state.training_thread.queue_task(task)
        eel.updateTrainingStatusOnUI(name, "Training task queued...")()
    except ValueError:
        eel.showErrorOnLabelTrainPage("Invalid training parameters provided.")


@eel.expose
def start_classification(dataset_name_for_model: str, recordings_whitelist_paths: list[str]):
    """Queues HDF5 files for classification using a specified model."""
    if not gui_state.proj or not gui_state.classify_thread: return
    model_to_use = gui_state.proj.models.get(dataset_name_for_model)
    if not model_to_use: return

    gui_state.classify_thread.start_inferring(model_to_use, recordings_whitelist_paths)

    h5_files_to_classify = []
    for rel_path in recordings_whitelist_paths:
        search_root = os.path.join(gui_state.proj.recordings_dir, rel_path)
        if os.path.isdir(search_root):
            for dirpath, _, filenames in os.walk(search_root):
                for filename in filenames:
                    if filename.endswith("_cls.h5"):
                        h5_files_to_classify.append(os.path.join(dirpath, filename))
    
    if h5_files_to_classify:
        with gui_state.classify_lock:
            for h5_file in h5_files_to_classify:
                if h5_file not in gui_state.classify_tasks:
                    gui_state.classify_tasks.append(h5_file)
        print(f"Queued {len(h5_files_to_classify)} files for classification with '{model_to_use.name}'.")


# =================================================================
# RENDERING & INTERNAL LOGIC (Not Exposed)
# =================================================================

def render_image():
    """
    Renders the current video frame, timelines, and overlays. Persistently shows
    corrected boundaries on the zoom bar for confirmed instances.
    """
    if not gui_state.label_capture or not gui_state.label_capture.isOpened():
        eel.updateLabelImageSrc(None, None, None)(); return

    total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames == 0:
        eel.updateLabelImageSrc(None, None, None)(); return

    current_frame_idx = max(0, min(int(gui_state.label_index), int(total_frames) - 1))
    gui_state.label_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = gui_state.label_capture.read()

    if not ret or frame is None:
        eel.updateLabelImageSrc(None, None, None)(); return

    main_frame_blob, timeline_blob, zoom_blob = None, None, None

    # 1. Generate Main Video Frame Blob
    frame_resized = cv2.resize(frame, (500, 500))
    _, encoded_main_frame = cv2.imencode(".jpg", frame_resized)
    main_frame_blob = base64.b64encode(encoded_main_frame.tobytes()).decode("utf-8")

    # 2. Generate Full Timeline Blob
    timeline_canvas = np.full((50, 500, 3), 100, dtype=np.uint8)
    timeline_canvas_with_overlays = fill_colors(timeline_canvas, total_frames)
    marker_x = int(500 * current_frame_idx / total_frames)
    cv2.line(timeline_canvas_with_overlays, (marker_x, 0), (marker_x, 50), (0, 0, 255), 2)
    _, encoded_timeline = cv2.imencode(".jpg", timeline_canvas_with_overlays)
    timeline_blob = base64.b64encode(encoded_timeline.tobytes()).decode("utf-8")

    # 3. Generate Zoom Timeline Blob (ALWAYS)
    zoom_canvas = np.full((50, 500, 3), 100, dtype=np.uint8)
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        
        # Use original boundaries for context if they exist, otherwise current ones.
        original_start = instance.get('_original_start', instance['start'])
        original_end = instance.get('_original_end', instance['end'])
        
        inst_len = original_end - original_start
        context = inst_len * 2
        zoom_start = max(0, original_start - context)
        zoom_end = min(total_frames, original_end + context)
        
        if zoom_end > zoom_start:
            try:
                b_idx = gui_state.label_dataset.labels["behaviors"].index(instance['label'])
                color_hex = gui_state.label_behavior_colors[b_idx].lstrip("#")
                bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))

                # If it has been edited, show the original as a faded background.
                if '_original_start' in instance:
                    orig_start_px = int(500 * (instance['_original_start'] - zoom_start) / (zoom_end - zoom_start))
                    orig_end_px = int(500 * (instance['_original_end'] - zoom_start) / (zoom_end - zoom_start))
                    overlay = zoom_canvas.copy()
                    cv2.rectangle(overlay, (orig_start_px, 5), (orig_end_px, 45), bgr_color, -1)
                    cv2.addWeighted(overlay, 0.4, zoom_canvas, 0.6, 0, zoom_canvas)

                # Draw the current boundaries (which may be corrected) as the main bar.
                start_px = int(500 * (instance['start'] - zoom_start) / (zoom_end - zoom_start))
                end_px = int(500 * (instance['end'] - zoom_start) / (zoom_end - zoom_start))
                cv2.rectangle(zoom_canvas, (start_px, 5), (end_px, 45), bgr_color, -1)
                
            except (ValueError, IndexError): pass 

            marker_x_zoom = int(500 * (current_frame_idx - zoom_start) / (zoom_end - zoom_start))
            cv2.line(zoom_canvas, (marker_x_zoom, 0), (marker_x_zoom, 50), (0, 0, 255), 2)

    _, encoded_zoom = cv2.imencode(".jpg", zoom_canvas)
    zoom_blob = base64.b64encode(encoded_zoom.tobytes()).decode("utf-8")

    eel.updateLabelImageSrc(main_frame_blob, timeline_blob, zoom_blob)()


def fill_colors(canvas_img: np.ndarray, total_frames: int) -> np.ndarray:
    """Draws colored bars on a timeline canvas using the in-memory session buffer."""
    if not all([gui_state.label_dataset, gui_state.label_videos, gui_state.label_capture, gui_state.label_capture.isOpened()]):
        return canvas_img

    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    if total_frames == 0: return canvas_img

    timeline_y, timeline_h, timeline_w = 0, canvas_img.shape[0], canvas_img.shape[1]

    for i, inst in enumerate(gui_state.label_session_buffer):
        is_human_added = 'confidence' not in inst
        is_confirmed = inst.get('_confirmed', False)
        
        # If in confirmation mode, ONLY draw labels that will be saved.
        if gui_state.label_confirmation_mode and not (is_human_added or is_confirmed):
            continue # Skip this instance, as it won't be saved

        try:
            b_idx = behaviors.index(inst['label'])
            color_hex = gui_state.label_behavior_colors[b_idx].lstrip("#")
            bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            
            start_px = int(timeline_w * inst.get("start", 0) / total_frames)
            end_px = int(timeline_w * (inst.get("end", 0) + 1) / total_frames)

            if start_px < end_px:
                confidence = inst.get('confidence', 1.0)
                overlay = canvas_img.copy()
                cv2.rectangle(overlay, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), bgr_color, -1)
                # Blend the overlay with the original canvas based on confidence
                # In confirmation mode, everything is solid
                alpha = 1.0 if gui_state.label_confirmation_mode else confidence
                cv2.addWeighted(overlay, alpha, canvas_img, 1 - alpha, 0, canvas_img)
                
                # Use the 'is_confirmed' flag for the border
                border_color = (255, 255, 255) if is_confirmed else bgr_color
                border_thickness = 2 if is_confirmed else 1
                cv2.rectangle(canvas_img, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), border_color, border_thickness)

                if i == gui_state.selected_instance_index:
                    # Keep the selection highlight distinct
                    cv2.rectangle(canvas_img, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), (0, 255, 255), 2) # Cyan highlight

        except (ValueError, IndexError):
            continue

    if gui_state.label_type != -1 and gui_state.label_start != -1:
        color_hex = gui_state.label_behavior_colors[gui_state.label_type].lstrip("#")
        bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        start_f, end_f = min(gui_state.label_start, gui_state.label_index), max(gui_state.label_start, gui_state.label_index)
        start_px, end_px = int(timeline_w * start_f / total_frames), int(timeline_w * (end_f + 1) / total_frames)
        if start_px < end_px:
            cv2.rectangle(canvas_img, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), bgr_color, -1)
            cv2.rectangle(canvas_img, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), (255, 255, 255), 1)

    return canvas_img


def update_counts():
    """Updates instance/frame counts in the UI based on the current session buffer."""
    if not gui_state.label_dataset: return

    if gui_state.label_videos and gui_state.label_vid_index >= 0:
        rel_path = os.path.relpath(gui_state.label_videos[gui_state.label_vid_index], start=gui_state.proj.path)
        eel.updateFileInfo(rel_path)()
    else:
        eel.updateFileInfo("No video loaded.")()

    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    counts = {b: {'instances': 0, 'frames': 0} for b in behaviors}
    
    for inst in gui_state.label_session_buffer:
        b_name = inst.get('label')
        if b_name in counts:
            counts[b_name]['instances'] += 1
            counts[b_name]['frames'] += (inst.get("end", 0) - inst.get("start", 0) + 1)
    
    for b_name, data in counts.items():
        eel.updateLabelingStats(b_name, data['instances'], data['frames'])()