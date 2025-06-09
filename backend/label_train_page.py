import cbas
import gui_state

# import threading # Not directly used here, tasks are offloaded to workthreads
# import ctypes   # Not directly used here

import eel

from cmap import Colormap # For coloring labels

import yaml
import base64
import os
import cv2
import numpy as np

import workthreads # For queuing TrainingTask

@eel.expose
def model_exists(model_name: str) -> bool:
    """
    Checks if a model with the given name exists in the current project.
    """
    if gui_state.proj is None:
        return False
    return model_name in gui_state.proj.models

@eel.expose
def load_dataset_configs() -> dict:
    """
    Loads configurations for all available datasets.
    Returns a dictionary mapping dataset names to their config dicts.
    """
    if gui_state.proj is None or not gui_state.proj.datasets:
        return {}
    return {name: dataset.config for name, dataset in gui_state.proj.datasets.items()}

@eel.expose
def handle_click_on_label_image(x: int, y: int):
    """
    Handles a click on the labeling image to set the current frame index.
    x: x-coordinate of the click (0-500 range from JS).
    y: y-coordinate (not used for frame seeking, but passed by JS).
    """
    if gui_state.label_capture is None or not gui_state.label_capture.isOpened():
        print("Label capture not ready for click handling.")
        return

    amount_of_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if amount_of_frames > 0:
        # Assuming image width in JS is 500px
        gui_state.label_index = int(x * amount_of_frames / 500)
        gui_state.label_index = max(0, min(gui_state.label_index, int(amount_of_frames) - 1)) # Clamp index
        render_image()
    else:
        print("Video has no frames to seek to.")


def tab20_map(val: int) -> int:
    """Maps an integer to an index for the tab20 colormap for distinct colors."""
    if val < 10:
        return val * 2
    else:
        return (val - 10) * 2 + 1

@eel.expose
def next_video(shift: int):
    """
    Moves to the next or previous video in the labeling queue.
    shift: 1 for next, -1 for previous.
    """
    if not gui_state.label_videos:
        print("No videos available for labeling.")
        eel.updateLabelImageSrc(None) # Clear image
        eel.updateFileInfo("No video loaded.")
        return

    gui_state.label_start = -1 # Reset current label start
    gui_state.label_vid_index = (gui_state.label_vid_index + shift) % len(gui_state.label_videos)
    gui_state.label_type = -1    # Reset current label type
    gui_state.label_index = 0    # Start at the beginning of the new video

    current_video_path = gui_state.label_videos[gui_state.label_vid_index]
    
    if gui_state.label_capture is not None and gui_state.label_capture.isOpened():
        gui_state.label_capture.release() # Release previous capture

    print(f"Loading video for labeling: {current_video_path}")
    capture = cv2.VideoCapture(current_video_path)

    if not capture.isOpened():
        print(f"Error: Could not open video {current_video_path}. Attempting to find another valid video.")
        # Attempt to recover by finding the next valid video
        for i in range(len(gui_state.label_videos)):
            gui_state.label_vid_index = (gui_state.label_vid_index + (1 if shift >=0 else -1)) % len(gui_state.label_videos)
            current_video_path = gui_state.label_videos[gui_state.label_vid_index]
            capture = cv2.VideoCapture(current_video_path)
            if capture.isOpened():
                print(f"Recovered: Loaded video {current_video_path}")
                break
            else:
                print(f"Skipping invalid video: {current_video_path}")
        else: # Else for 'for' loop: executed if loop finished without break
            eel.updateLabelImageSrc(None)
            eel.updateFileInfo("No valid videos found in dataset.")
            print("Error: No valid videos found in the dataset after attempting recovery.")
            gui_state.label_capture = None
            return

    gui_state.label_capture = capture
    # Set to first frame and render (next_frame will handle initial render)
    # gui_state.label_index = 0 # Already set
    next_frame(0) # Call next_frame with 0 shift to render current frame (next_frame handles shift<=0)
    update_counts() # Update counts for the new video

@eel.expose
def next_frame(shift: int):
    """
    Moves to the next/previous frame in the current labeling video.
    shift: Number of frames to move by. If <=0, it moves by shift-1 (e.g. 0 -> -1, -1 -> -2).
    """
    if gui_state.label_capture is None or not gui_state.label_capture.isOpened():
        # print("No video capture available for next_frame.") # Can be noisy
        return

    # Original logic for shift adjustment seems to be for specific keybinding behavior
    if shift <= 0: 
        shift -= 1 # e.g., arrow left (0) becomes -1, custom key (-1) becomes -2

    amount_of_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if amount_of_frames == 0: return # No frames

    gui_state.label_index += shift
    # Ensure label_index wraps around correctly and stays within bounds
    gui_state.label_index = int(gui_state.label_index % amount_of_frames)
    if gui_state.label_index < 0: # Handle negative wrap-around
        gui_state.label_index += int(amount_of_frames)
    
    render_image()

@eel.expose
def start_labeling(name: str) -> tuple[list[str], list[str]] | bool:
    """
    Initializes the labeling interface for a given dataset.
    name: The name of the dataset to label.
    Returns: Tuple of (behavior_names, color_map_strings) or False on failure.
    """
    if gui_state.proj is None:
        print("Error: Project not loaded. Cannot start labeling.")
        return False
    if name not in gui_state.proj.datasets:
        print(f"Error: Dataset '{name}' not found in project.")
        return False

    # Reset labeling state
    if gui_state.label_capture and gui_state.label_capture.isOpened():
        gui_state.label_capture.release()
    gui_state.label_capture = None
    gui_state.label_index = -1
    gui_state.label_videos = []
    gui_state.label_vid_index = -1
    gui_state.label_type = -1
    gui_state.label_start = -1
    gui_state.label_history = []

    dataset: cbas.Dataset = gui_state.proj.datasets[name]
    gui_state.label_dataset = dataset

    whitelist = dataset.config.get("whitelist", []) # Use .get for safety
    if not isinstance(whitelist, list): whitelist = []


    gui_state.label_col_map = Colormap("seaborn:tab20")

    # Collect all .mp4 video files from the project's recordings directory
    all_project_videos = []
    if gui_state.proj.recordings_dir and os.path.exists(gui_state.proj.recordings_dir):
        for root_dir, _, files in os.walk(gui_state.proj.recordings_dir):
            for file in files:
                if file.endswith(".mp4"):
                    all_project_videos.append(os.path.join(root_dir, file))
    
    if not all_project_videos:
        print("No .mp4 files found in any recording directories.")
        return False
    
    # Filter videos based on the dataset's whitelist
    # A whitelist entry can be a partial path (e.g., "YYYYMMDD/CameraName-Session")
    valid_videos_for_labeling = []
    if not whitelist: # If whitelist is empty, consider all videos
        print("Warning: Dataset whitelist is empty. Considering all project videos for labeling.")
        valid_videos_for_labeling = all_project_videos
    else:
        for video_path in all_project_videos:
            # Normalize paths for consistent matching
            normalized_video_path = os.path.normpath(video_path)
            for wl_pattern in whitelist:
                normalized_wl_pattern = os.path.normpath(wl_pattern)
                if normalized_wl_pattern in normalized_video_path:
                    valid_videos_for_labeling.append(video_path)
                    break # Found a match, no need to check other whitelist patterns for this video
    
    if not valid_videos_for_labeling:
        print(f"No videos found matching the whitelist for dataset '{name}'. Whitelist: {whitelist}")
        return False

    gui_state.label_videos = sorted(list(set(valid_videos_for_labeling))) # Sort and unique

    # Load the first video
    if gui_state.label_videos:
        next_video(0) # Call with 0 to load the first video (index 0 after modulo)
        # update_counts() is called within next_video -> next_frame path
    else: # Should be caught by 'if not valid_videos_for_labeling'
        return False

    dataset_behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    behavior_colors = [
        str(gui_state.label_col_map(tab20_map(i))) for i in range(len(dataset_behaviors))
    ]
    return dataset_behaviors, behavior_colors


@eel.expose
def label_frame(value: int):
    """
    Handles labeling actions based on user input (e.g., key press).
    value: Integer representing the behavior index or action.
    """
    if gui_state.label_dataset is None or not gui_state.label_videos or gui_state.label_vid_index == -1:
        print("Labeling not active or no video loaded.")
        return

    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    current_video_path = gui_state.label_videos[gui_state.label_vid_index]

    # Check if clicked on an existing label to change it
    existing_label_info = None
    for b_idx, b_name in enumerate(behaviors):
        if b_name not in gui_state.label_dataset.labels.get("labels", {}): continue
        for i, inst in enumerate(gui_state.label_dataset.labels["labels"][b_name]):
            if inst.get("video") == current_video_path and \
               inst.get("start", -1) <= gui_state.label_index <= inst.get("end", -1):
                existing_label_info = {"behavior_name": b_name, "instance_index": i, "original_value": b_idx}
                break
        if existing_label_info: break
    
    # Case 1: Change type of an existing label
    if existing_label_info and gui_state.label_type == -1: # Not currently defining a new label
        if 0 <= value < len(behaviors): # Ensure 'value' is a valid new behavior index
            old_behavior_name = existing_label_info["behavior_name"]
            instance_idx_to_move = existing_label_info["instance_index"]
            
            if value == existing_label_info["original_value"]: # Clicked same behavior, do nothing
                print("Clicked on existing label of the same type. No change.")
                return

            new_behavior_name = behaviors[value]
            
            instance_to_move = gui_state.label_dataset.labels["labels"][old_behavior_name].pop(instance_idx_to_move)
            instance_to_move["label"] = new_behavior_name # Update label field in instance
            
            gui_state.label_dataset.labels["labels"].setdefault(new_behavior_name, []).append(instance_to_move)
            print(f"Changed label: {instance_to_move} from '{old_behavior_name}' to '{new_behavior_name}'.")
            
            with open(gui_state.label_dataset.labels_path, "w") as file: # Use "w" to overwrite
                yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)
            update_counts(); render_image()
        else:
            print(f"Invalid new behavior index {value} for changing label.")
        return

    # Case 2 & 3: Start or End a new label
    if value >= len(behaviors): # Invalid behavior index for starting/ending a new label
        print(f"Invalid behavior index {value} for new label.")
        return False # Keep consistent with JS expectation

    if value == gui_state.label_type: # End current labeling
        try:
            add_instance()
        except Exception as e:
            print(f"Error adding instance: {e}") # TODO: Send this error to UI via Eel
            # Optionally, could use eel.showErrorModal(str(e)) if such a JS function exists
        gui_state.label_type = -1 # Reset label type
        gui_state.label_start = -1
    elif gui_state.label_type == -1: # Start new labeling
        gui_state.label_type = value
        gui_state.label_start = gui_state.label_index
        print(f"Started labeling for behavior: {behaviors[value]} at frame {gui_state.label_index}")
    else: # Clicked a different behavior while one is active - cancel current
        print(f"Cancelled labeling for behavior: {behaviors[gui_state.label_type]}. Starting new for {behaviors[value]}.")
        gui_state.label_type = value
        gui_state.label_start = gui_state.label_index
    
    render_image() # Update timeline with current selection


def add_instance():
    """Adds the currently defined (start-end-type) instance to the dataset labels."""
    if gui_state.label_type == -1 or gui_state.label_start == -1:
        print("Cannot add instance: Label type or start not set.")
        return

    start_index = min(gui_state.label_start, gui_state.label_index)
    end_index = max(gui_state.label_start, gui_state.label_index)

    if start_index == end_index : # Single frame label, make it inclusive
        print(f"Warning: Labeling a single frame ({start_index}). This might be too short for some seq_len.")
        # Depending on policy, you might want to enforce a minimum duration or allow it.
        # For now, allow, but `convert_instances` might skip it if seq_len is too large.

    current_video_path = gui_state.label_videos[gui_state.label_vid_index]
    all_labels_dict = gui_state.label_dataset.labels.get("labels", {})
    behaviors_list = gui_state.label_dataset.labels.get("behaviors", [])

    # Check for collisions with existing labels in the current video
    for behavior_name_iter in behaviors_list:
        if behavior_name_iter not in all_labels_dict: continue
        for existing_inst in all_labels_dict[behavior_name_iter]:
            if existing_inst.get("video") == current_video_path:
                existing_start = existing_inst.get("start", -1)
                existing_end = existing_inst.get("end", -1)
                # Check for overlap: (StartA <= EndB) and (EndA >= StartB)
                if start_index <= existing_end and end_index >= existing_start:
                    # More precise overlap check:
                    # Overlap if not (new_end < old_start OR new_start > old_end)
                    # i.e., overlap if (end_index >= existing_start AND start_index <= existing_end)
                    print(f"Collision detected: New label ({start_index}-{end_index}) overlaps with existing "
                          f"'{behavior_name_iter}' label ({existing_start}-{existing_end}).")
                    eel.showError("Overlapping behavior region! Behavior not recorded.") # Example UI error
                    raise Exception("Overlapping behavior region! Behavior not recorded.")


    target_behavior_name = behaviors_list[gui_state.label_type]
    new_instance = {
        "video": current_video_path,
        "start": start_index,
        "end": end_index,
        "label": target_behavior_name, # Store the name, not index
    }

    all_labels_dict.setdefault(target_behavior_name, []).append(new_instance)
    gui_state.label_history.append(new_instance) # For undo functionality

    with open(gui_state.label_dataset.labels_path, "w") as file:
        yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)
    
    print(f"Added instance: {new_instance}")
    update_counts() # Update counts on UI

@eel.expose
def update_counts():
    """
    Updates the display of instance and frame counts.
    The counts sent to the labeling UI are for the CURRENT VIDEO ONLY.
    The metrics saved to the dataset config are for the ENTIRE DATASET.
    """
    if gui_state.label_dataset is None or gui_state.proj is None: return

    dataset_labels = gui_state.label_dataset.labels
    behaviors_list = dataset_labels.get("behaviors", [])
    all_labels_by_behavior = dataset_labels.get("labels", {})

    # Determine the current video path
    current_video_path = None
    if gui_state.label_videos and gui_state.label_vid_index >= 0:
        current_video_path = gui_state.label_videos[gui_state.label_vid_index]

    for b_name in behaviors_list:
        all_instances_for_b = all_labels_by_behavior.get(b_name, [])
        
        # --- PER-VIDEO COUNTS for the UI ---
        instances_in_current_video = 0
        frames_in_current_video = 0
        if current_video_path: # Only count if a video is loaded
            for inst in all_instances_for_b:
                # This is the crucial filter
                if inst.get("video") == current_video_path:
                    instances_in_current_video += 1
                    frames_in_current_video += (inst.get("end", 0) - inst.get("start", 0) + 1)
        
        # Send the correct, filtered counts to the labeling UI
        eel.updateLabelingStats(b_name, instances_in_current_video, frames_in_current_video)()

        # --- TOTAL DATASET COUNTS for the main page metrics ---
        total_instances_for_b = len(all_instances_for_b)
        total_frames_for_b = 0
        for inst in all_instances_for_b:
            total_frames_for_b += (inst.get("end", 0) - inst.get("start", 0) + 1)
            
        # Update the dataset-wide metrics (used on the main Label/Train page)
        gui_state.label_dataset.update_metric(b_name, "Train #", int(round(total_instances_for_b * 0.75)))
        gui_state.label_dataset.update_metric(b_name, "Test #", total_instances_for_b - int(round(total_instances_for_b * 0.75)))
        gui_state.label_dataset.update_metric(b_name, "Train Frames", int(round(total_frames_for_b * 0.75)))
        gui_state.label_dataset.update_metric(b_name, "Test Frames", total_frames_for_b - int(round(total_frames_for_b * 0.75)))

    # Update the file info display at the top of the labeling UI
    if current_video_path:
        try:
            rel_path = os.path.relpath(current_video_path, start=gui_state.proj.path)
        except ValueError:
            rel_path = current_video_path
        eel.updateFileInfo(rel_path)()
    else:
        eel.updateFileInfo("No video loaded.")()


@eel.expose
def delete_instance():
    """Deletes an existing labeled instance that the current frame falls into."""
    if gui_state.label_dataset is None or not gui_state.label_videos or gui_state.label_vid_index == -1: return

    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    current_video_path = gui_state.label_videos[gui_state.label_vid_index]
    labels_dict = gui_state.label_dataset.labels.get("labels", {})

    found_and_deleted = False
    for b_name in behaviors:
        if b_name not in labels_dict: continue
        instances_for_b = labels_dict[b_name]
        for i, inst in enumerate(instances_for_b):
            if inst.get("video") == current_video_path and \
               inst.get("start", -1) <= gui_state.label_index <= inst.get("end", -1):
                print(f"Deleting instance: {inst}")
                del instances_for_b[i]
                found_and_deleted = True
                break # Found and deleted, exit inner loop
        if found_and_deleted: break # Exit outer loop

    if found_and_deleted:
        with open(gui_state.label_dataset.labels_path, "w") as file:
            yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)
        update_counts()
        render_image()
    else:
        print("No label to delete at current frame.")


@eel.expose
def pop_instance():
    """Undoes the last added label instance."""
    if not gui_state.label_history:
        print("No label history to pop (undo).")
        return

    last_added_inst = gui_state.label_history.pop()
    target_behavior_name = last_added_inst["label"]
    labels_dict = gui_state.label_dataset.labels.get("labels", {})

    if target_behavior_name in labels_dict:
        # Find and remove the exact instance (safer than relying on just last in list)
        instances_for_b = labels_dict[target_behavior_name]
        for i, inst in enumerate(instances_for_b):
            if inst["video"] == last_added_inst["video"] and \
               inst["start"] == last_added_inst["start"] and \
               inst["end"] == last_added_inst["end"]:
                del instances_for_b[i]
                print(f"Popped (undid) instance: {last_added_inst}")
                with open(gui_state.label_dataset.labels_path, "w") as file:
                    yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)
                update_counts()
                render_image()
                return
    print(f"Could not find instance {last_added_inst} in labels to pop.")


def render_image():
    """Renders the current video frame with label overlays and sends to UI."""
    if gui_state.label_capture is None or not gui_state.label_capture.isOpened() or gui_state.label_index < 0:
        # print("Render image: capture not ready or index invalid.") # Can be noisy
        eel.updateLabelImageSrc(None) # Send None to clear image if capture not ready
        return

    total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames == 0: 
        eel.updateLabelImageSrc(None); return

    # Clamp label_index again just in case
    current_frame_idx = max(0, min(int(gui_state.label_index), int(total_frames) - 1))
    gui_state.label_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = gui_state.label_capture.read()

    if ret and frame is not None:
        frame_resized = cv2.resize(frame, (500, 500))
        
        # Create canvas for frame + timeline
        canvas_height = frame_resized.shape[0] + 50 # 50px for timeline
        canvas = np.zeros((canvas_height, frame_resized.shape[1], 3), dtype=np.uint8) # Start with black
        
        canvas[:-50, :, :] = frame_resized      # Copy frame
        canvas[-50:-1, :, :] = 100              # Timeline background (grey)
        # canvas[-1, :, :] = 0                  # Bottom border (black) - optional

        canvas_with_overlays = fill_colors(canvas) # Add label overlays to timeline part

        # Draw current frame marker on timeline
        if total_frames > 0: # Avoid division by zero
            marker_pos_x = int(frame_resized.shape[1] * current_frame_idx / total_frames)
            marker_pos_x = np.clip(marker_pos_x, 0, frame_resized.shape[1] - 2) # Ensure within bounds for marker width
            
            # Timeline marker (e.g., red line)
            cv2.line(canvas_with_overlays, 
                     (marker_pos_x, canvas_height - 45), 
                     (marker_pos_x, canvas_height - 5), 
                     (0,0,255), thickness=2) # Red marker line

        _, encoded_frame = cv2.imencode(".jpg", canvas_with_overlays)
        blob = base64.b64encode(encoded_frame.tobytes()).decode("utf-8")
        eel.updateLabelImageSrc(blob)()
    else:
        print(f"Failed to read frame {gui_state.label_index} from video.")
        eel.updateLabelImageSrc(None) # Clear image on error

def fill_colors(canvas_img: np.ndarray) -> np.ndarray:
    """Draws colored bars on the timeline part of the canvas_img for existing labels."""
    if gui_state.label_dataset is None or not gui_state.label_videos or gui_state.label_vid_index < 0:
        return canvas_img

    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    labels_data = gui_state.label_dataset.labels.get("labels", {})
    current_video_path = gui_state.label_videos[gui_state.label_vid_index]
    total_frames_in_video = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if total_frames_in_video == 0: return canvas_img # No frames to draw on

    timeline_y_start = canvas_img.shape[0] - 49 # Start y of the colorable timeline area
    timeline_height = 45 # Height of the colorable area, leaving small borders
    timeline_width_pixels = canvas_img.shape[1]

    # Draw existing saved labels
    for behavior_idx, behavior_name in enumerate(behaviors):
        if behavior_name not in labels_data: continue
        
        # Get color for this behavior
        color_hex = str(gui_state.label_col_map(tab20_map(behavior_idx))).lstrip("#")
        # Convert hex to BGR (OpenCV format)
        bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0)) # BGR order

        for inst in labels_data[behavior_name]:
            if inst.get("video") == current_video_path:
                start_f = inst.get("start", 0)
                end_f = inst.get("end", 0)
                
                # Calculate pixel positions for start and end on the timeline
                marker_pos_start_x = int(timeline_width_pixels * start_f / total_frames_in_video)
                marker_pos_end_x = int(timeline_width_pixels * (end_f + 1) / total_frames_in_video) # +1 to make end inclusive for rect
                marker_pos_start_x = np.clip(marker_pos_start_x, 0, timeline_width_pixels)
                marker_pos_end_x = np.clip(marker_pos_end_x, 0, timeline_width_pixels)

                if marker_pos_start_x < marker_pos_end_x: # Ensure width > 0
                    cv2.rectangle(canvas_img, 
                                  (marker_pos_start_x, timeline_y_start), 
                                  (marker_pos_end_x, timeline_y_start + timeline_height -1), 
                                  bgr_color, thickness=cv2.FILLED)

    # Draw currently active (being defined) label
    if gui_state.label_type != -1 and gui_state.label_start != -1:
        active_color_hex = str(gui_state.label_col_map(tab20_map(gui_state.label_type))).lstrip("#")
        active_bgr_color = tuple(int(active_color_hex[i:i+2], 16) for i in (4, 2, 0))

        temp_start_f = min(gui_state.label_start, gui_state.label_index)
        temp_end_f = max(gui_state.label_start, gui_state.label_index)

        marker_pos_S_active = int(timeline_width_pixels * temp_start_f / total_frames_in_video)
        marker_pos_E_active = int(timeline_width_pixels * (temp_end_f + 1) / total_frames_in_video)
        marker_pos_S_active = np.clip(marker_pos_S_active, 0, timeline_width_pixels)
        marker_pos_E_active = np.clip(marker_pos_E_active, 0, timeline_width_pixels)

        if marker_pos_S_active < marker_pos_E_active:
            cv2.rectangle(canvas_img, 
                          (marker_pos_S_active, timeline_y_start), 
                          (marker_pos_E_active, timeline_y_start + timeline_height -1), 
                          active_bgr_color, thickness=cv2.FILLED)
            # Add a slightly different border or overlay to indicate it's "active"
            cv2.rectangle(canvas_img, 
                          (marker_pos_S_active, timeline_y_start), 
                          (marker_pos_E_active, timeline_y_start + timeline_height -1), 
                          (255,255,255), thickness=1) # White border

    return canvas_img


@eel.expose
def get_record_tree() -> dict | bool:
    """Fetches the recording directory tree structure."""
    if gui_state.proj is None or not gui_state.proj.recordings_dir or \
       not os.path.exists(gui_state.proj.recordings_dir):
        return False # Project not loaded or recordings dir missing

    rt = {} # date_str -> list of session_dir_names
    for date_dir_entry in os.scandir(gui_state.proj.recordings_dir):
        if date_dir_entry.is_dir():
            date_str = date_dir_entry.name
            rt[date_str] = []
            for session_dir_entry in os.scandir(date_dir_entry.path):
                if session_dir_entry.is_dir():
                    rt[date_str].append(session_dir_entry.name)
    return rt if rt else False


@eel.expose
def create_dataset(name: str, behaviors: list[str], recordings_whitelist: list[str]) -> bool:
    """
    Creates a new dataset via the project interface.
    recordings_whitelist: List of strings, each being "DATE_DIR/SESSION_DIR_NAME"
    """
    if gui_state.proj is None: return False
    
    # The cbas.Project.create_dataset expects full relative paths for whitelist
    # The JS seems to send "DATE_DIR\\SESSION_DIR_NAME" or just "DATE_DIR"
    # We need to ensure the format matches what cbas.Project expects or adapt here.
    # Assuming cbas.Project.create_dataset handles the whitelist format appropriately now.
    
    dataset = gui_state.proj.create_dataset(name, behaviors, recordings_whitelist)
    if dataset:
        gui_state.label_dataset = dataset # Set as current for potential immediate labeling
        return True
    return False

@eel.expose
def train_model(name: str, batch_size: str, learning_rate: str, epochs: str, sequence_length: str, training_method = str):
    """Queues a training task."""
    if gui_state.proj is None or name not in gui_state.proj.datasets:
        print(f"Cannot train: Dataset '{name}' not found or project not loaded.")
        return
    if gui_state.training_thread is None:
        print("Cannot train: Training thread not initialized.")
        return

    dataset = gui_state.proj.datasets[name]
    
    try:
        bs = int(batch_size)
        lr = float(learning_rate)
        ep = int(epochs)
        sl = int(sequence_length)
    except ValueError:
        print("Error: Batch size, learning rate, epochs, and sequence length must be valid numbers.")
        # eel.showError("Invalid training parameters.") # Example
        return

    task = workthreads.TrainingTask(
        name=name, 
        dataset=dataset, 
        behaviors=dataset.config.get('behaviors', []), 
        batch_size=bs, 
        learning_rate=lr, 
        epochs=ep, 
        sequence_length=sl,
        training_method=training_method
    )
    gui_state.training_thread.queue_task(task)
    print(f"Training task for '{name}' queued.")
    eel.updateTrainingStatusOnUI(name, "Training task queued...")() # Notify UI

@eel.expose
def start_classification(dataset_name_for_model: str, recordings_whitelist_paths: list[str]):
    """
    Starts classification on specified recordings using the model trained for dataset_name_for_model.
    recordings_whitelist_paths: List of "DATE_DIR/SESSION_DIR_NAME" or "DATE_DIR" to classify.
    """
    if gui_state.proj is None or gui_state.classify_thread is None:
        print("Project or classify thread not ready for start_classification.")
        return
    if dataset_name_for_model not in gui_state.proj.models:
        print(f"Model for dataset '{dataset_name_for_model}' not found.")
        # eel.showError(f"Model '{dataset_name_for_model}' not found.")
        return

    model_to_use = gui_state.proj.models[dataset_name_for_model]
    
    # The ClassificationThread's start_inferring takes a general whitelist (not used for task filtering).
    # The tasks are added here based on recordings_whitelist_paths.
    # This is okay, but start_inferring's whitelist argument becomes somewhat redundant.
    gui_state.classify_thread.start_inferring(model_to_use, recordings_whitelist_paths) # Pass model object

    h5_files_to_classify = []
    for rel_path_pattern in recordings_whitelist_paths:
        # Construct full path to search within. rel_path_pattern can be "DATE" or "DATE/SESSION"
        search_root = os.path.join(gui_state.proj.recordings_dir, rel_path_pattern)
        if os.path.isdir(search_root):
            for dirpath, _, filenames in os.walk(search_root):
                for filename in filenames:
                    if filename.endswith("_cls.h5"): # Ensure it's an embedding file
                        h5_files_to_classify.append(os.path.join(dirpath, filename))
        else:
            print(f"Warning: Whitelisted path for classification not found or not a directory: {search_root}")
    
    if h5_files_to_classify:
        gui_state.classify_lock.acquire()
        # Add only if not already in the queue to avoid duplicates
        for h5_file in h5_files_to_classify:
            if h5_file not in gui_state.classify_tasks:
                gui_state.classify_tasks.append(h5_file)
        gui_state.classify_lock.release()
        print(f"Queued {len(h5_files_to_classify)} HDF5 files for classification using model '{model_to_use.name}'.")
    else:
        print(f"No HDF5 files found to classify for whitelist: {recordings_whitelist_paths}")

# It's good practice to have a single point for eel.expose definitions for functions
# that are called from JS but might be defined elsewhere in Python if not directly in this file.
# However, since these are callbacks from JS, they are fine here.

# eel.expose for updateCount is in this file.
# eel.expose for updateFileInfo is in this file.
# eel.expose for updateLabelImageSrc is in this file.
# eel.expose for updateTrainingStatus (used by workthreads.py) needs to be defined here
# if not already in app.py or another globally accessible place.

# @eel.expose
# def updateTrainingStatus(dataset_name: str, message: str):
    # """JS callback to update training status on the UI."""
    # # This is a Python function callable from JS.
    # # The actual call from Python to JS would be eel.js_update_training_status(dataset_name, message)
    # # So, the JS side needs:
    # # eel.expose(js_update_training_status);
    # # function js_update_training_status(dataset_name, message) { /* update UI */ }
    # # This Python function is here if JS needs to call Python for some reason about training status,
    # # which is less common. The typical flow is Python calling JS.
    # # For Python to call JS:
    # # In workthreads.py: eel.js_side_update_training_status(task.name, "message")()
    # # In JS:
    # # eel.expose(js_side_update_training_status);
    # # function js_side_update_training_status(dataset_name, message) {
    # #    document.getElementById('status-for-' + dataset_name).innerText = message;
    # # }
    # print(f"UI Event (Python side): updateTrainingStatus for {dataset_name}: {message}")
    # # This function, if called from JS, doesn't do much unless it's meant to trigger Python logic.
    # # If the goal is Python updating JS, this eel.expose is not what's used by workthreads.
    # pass

# # Similarly for updateCount, updateFileInfo, updateLabelImageSrc, if they are meant
# # to be called *from Python to update JS*, the JS functions need to be exposed in JS,
# # and Python calls them using eel.js_function_name().
# # The @eel.expose here makes them callable *from JS to Python*.