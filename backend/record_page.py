import os
import cv2
import base64
import datetime
import subprocess
# Using gevent for non-blocking subprocess for thumbnails can be complex to set up
# and might conflict with Eel's own async nature or other threading.
# Sticking to standard subprocess or threading for simplicity unless gevent is a hard requirement.
# import gevent
# from gevent.subprocess import Popen, PIPE

import eel
import cbas # Assuming cbas.py is the final name
import gui_state
import threading # For running ffmpeg thumbnail generation in a separate thread

# --- Camera Thumbnail Generation ---

def _fetch_single_frame_ffmpeg(rtsp_url: str, output_frame_location: str) -> tuple[bool, str]:
    """
    Uses ffmpeg to fetch a single frame from an RTSP stream.
    Helper function, not exposed to Eel.
    """
    try:
        if os.path.exists(output_frame_location):
            os.remove(output_frame_location)

        # Command to extract roughly the 35th frame (or one early frame)
        # -vf "select=eq(n\,34)" selects the 35th frame (0-indexed)
        # Using a slightly later frame can sometimes be more reliable if stream startup is slow
        # -ss 1 attempts to seek 1 second in, then take the first frame. May be faster.
        command = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-rtsp_transport", "tcp",     # Use TCP for RTSP
            "-i", rtsp_url,
            # "-ss", "1",                  # Optional: seek to 1 second
            "-vf", "select=eq(n\,34)",   # Select an early frame (e.g., 35th)
            "-vframes", "1",             # Extract only one frame
            output_frame_location
        ]
        print(f"Fetching frame for thumbnail: {' '.join(command)}")
        
        # Timeout for ffmpeg process to prevent indefinite blocking
        process = subprocess.run(command, capture_output=True, text=True, timeout=15) # Increased timeout

        if process.returncode == 0 and os.path.exists(output_frame_location):
            print(f"Thumbnail successfully fetched for {rtsp_url} to {output_frame_location}")
            return True, output_frame_location
        else:
            error_msg = f"FFMPEG error for {rtsp_url}. Return code: {process.returncode}. Stderr: {process.stderr.strip()}"
            print(error_msg)
            return False, error_msg
    except subprocess.TimeoutExpired:
        error_msg = f"FFMPEG timeout fetching frame from {rtsp_url}."
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Exception fetching frame from {rtsp_url}: {e}"
        print(error_msg)
        return False, error_msg

def _process_and_send_thumbnail(camera_name: str, frame_path: str):
    """Helper to read, encode, and send thumbnail via Eel."""
    try:
        frame_image = cv2.imread(frame_path)
        if frame_image is None:
            print(f"Could not read thumbnail image: {frame_path} for camera {camera_name}")
            # Optionally send a default "no connection" image blob
            # eel.updateImageSrc(camera_name, "path/to/default/no_connection_blob_or_image_data")()
            return

        _, encoded_jpeg = cv2.imencode(".jpg", frame_image)
        blob = base64.b64encode(encoded_jpeg.tobytes()).decode("utf-8")
        eel.updateImageSrc(camera_name, blob)() # Call JS function to update UI
        print(f"Thumbnail updated in UI for camera: {camera_name}")
    except Exception as e:
        print(f"Error processing/sending thumbnail for {camera_name} from {frame_path}: {e}")


@eel.expose
def set_camera_thumbnails():
    """
    (Re)loads existing thumbnail images from disk and sends them to the UI.
    This is typically called after download_camera_thumbnails completes or on page load.
    """
    if gui_state.proj is None: return
    print("Setting camera thumbnails from disk...")
    for camera in gui_state.proj.cameras.values():
        frame_location = os.path.join(camera.path, "frame.jpg") # Camera object has .path
        if os.path.exists(frame_location):
            _process_and_send_thumbnail(camera.name, frame_location)
        else:
            print(f"Thumbnail file not found for camera {camera.name} at {frame_location}")
            # Optionally, send a placeholder to the UI
            # eel.updateImageSrc(camera.name, "placeholder_or_error_blob")()


def _download_thumbnails_task():
    """Target function for the thumbnail downloading thread."""
    if gui_state.proj is None: return
    
    camera_info_list = []
    for camera_obj in gui_state.proj.cameras.values(): # Iterate over Camera objects
        camera_info_list.append({
            "name": camera_obj.name, 
            "rtsp_url": camera_obj.rtsp_url,
            "output_path": os.path.join(camera_obj.path, "frame.jpg") # Use camera.path
        })

    for cam_info in camera_info_list:
        print(f"Attempting to download thumbnail for {cam_info['name']}...")
        success, _ = _fetch_single_frame_ffmpeg(cam_info["rtsp_url"], cam_info["output_path"])
        if success:
            # Process and send immediately after successful download for faster UI update
             _process_and_send_thumbnail(cam_info["name"], cam_info["output_path"])
        else:
            # If fetch failed, could try to send a default "error" image to UI for this camera
            # eel.updateImageSrc(cam_info['name'], "path_to_error_image_blob")()
            pass
    
    # After attempting all, one final call to set_camera_thumbnails 
    # can ensure any successfully fetched ones (if not sent individually) are displayed.
    # However, sending individually above is more responsive.
    # eel.set_camera_thumbnails() # This would re-read all from disk.
    print("Thumbnail download process completed.")


@eel.expose
def download_camera_thumbnails():
    """
    Initiates the download of thumbnails for all cameras in a background thread.
    """
    print("Download camera thumbnails requested...")
    # Run the download task in a separate thread to avoid blocking Eel
    # This allows multiple thumbnails to be fetched concurrently if _fetch_single_frame_ffmpeg was async,
    # but with subprocess.run, they are sequential within this thread.
    # For true concurrency, each ffmpeg call would need its own thread or use async subprocess.
    # However, a single thread for all ffmpeg calls is simpler and avoids overwhelming resources.
    download_thread = threading.Thread(target=_download_thumbnails_task, daemon=True)
    download_thread.start()


# --- Camera Configuration ---

@eel.expose
def get_camera_list() -> list[tuple[str, dict]]:
    """Returns a list of (camera_name, camera_settings_dict) for all cameras."""
    if gui_state.proj is None: return []
    camera_list = []
    for camera_name, camera_obj in gui_state.proj.cameras.items():
        camera_list.append((camera_name, camera_obj.settings_to_dict()))
    return camera_list

@eel.expose
def get_camera_settings(camera_name: str) -> dict | None:
    """Returns the settings dictionary for a specific camera."""
    if gui_state.proj is None or camera_name not in gui_state.proj.cameras:
        print(f"Camera '{camera_name}' not found for get_camera_settings.")
        return None
    return gui_state.proj.cameras[camera_name].settings_to_dict()

@eel.expose
def save_camera_settings(camera_name: str, camera_settings: dict):
    """Saves updated settings for a specific camera."""
    if gui_state.proj is None or camera_name not in gui_state.proj.cameras:
        print(f"Camera '{camera_name}' not found for save_camera_settings.")
        return False # Indicate failure
    try:
        gui_state.proj.cameras[camera_name].update_settings(camera_settings)
        print(f"Settings saved for camera: {camera_name}")
        # After saving, re-downloading the thumbnail might be good if URL or crop changed significantly
        # For simplicity, let UI handle re-triggering thumbnail download if needed, or do it here:
        # _download_thumbnails_task() # Or a targeted download for this camera
        return True # Indicate success
    except Exception as e:
        print(f"Error saving settings for camera {camera_name}: {e}")
        return False


@eel.expose
def rename_camera(old_camera_name: str, new_camera_name: str) -> bool:
    """Renames a camera. This involves creating a new one and removing the old."""
    if gui_state.proj is None: return False
    if old_camera_name not in gui_state.proj.cameras:
        print(f"Cannot rename: Old camera '{old_camera_name}' not found.")
        return False
    if new_camera_name in gui_state.proj.cameras:
        print(f"Cannot rename: New camera name '{new_camera_name}' already exists.")
        return False
    if not new_camera_name.strip(): # Basic validation for new name
        print("Cannot rename: New camera name cannot be empty.")
        return False

    print(f"Renaming camera '{old_camera_name}' to '{new_camera_name}'...")
    settings = gui_state.proj.cameras[old_camera_name].settings_to_dict()
    settings['name'] = new_camera_name # Ensure the new name is in the settings dict for the new camera

    # Create new camera (folder and config)
    new_cam = gui_state.proj.create_camera(new_camera_name, settings)
    if new_cam:
        # Optionally, try to move/copy existing thumbnail if applicable
        old_thumb_path = os.path.join(gui_state.proj.cameras[old_camera_name].path, "frame.jpg")
        if os.path.exists(old_thumb_path):
            new_thumb_path = os.path.join(new_cam.path, "frame.jpg")
            try:
                shutil.copy(old_thumb_path, new_thumb_path)
            except Exception as e:
                print(f"Could not copy thumbnail during rename: {e}")
        
        gui_state.proj.remove_camera(old_camera_name) # Removes old camera folder and from dict
        print(f"Camera '{old_camera_name}' successfully renamed to '{new_camera_name}'.")
        return True
    else:
        print(f"Failed to create new camera entry for '{new_camera_name}' during rename.")
        return False


@eel.expose
def create_camera(camera_name: str, rtsp_url: str) -> bool:
    """Creates a new camera with default settings."""
    if gui_state.proj is None: return False
    if not camera_name.strip() or not rtsp_url.strip(): # Basic validation
        print("Camera name and RTSP URL cannot be empty.")
        return False

    settings = {
        "name": camera_name, # Name is part of settings for cbas.Camera
        "rtsp_url": rtsp_url,
        "framerate": 10,    # Default
        "resolution": 256,  # Default
        "crop_left_x": 0.0, # Default
        "crop_top_y": 0.0,  # Default
        "crop_width": 1.0,  # Default
        "crop_height": 1.0, # Default
    }
    if gui_state.proj.create_camera(camera_name, settings):
        print(f"Camera '{camera_name}' created successfully.")
        # Trigger thumbnail download for the new camera
        cam_obj = gui_state.proj.cameras.get(camera_name)
        if cam_obj:
             thumb_thread = threading.Thread(target=_download_thumbnails_task_single, args=(cam_obj,), daemon=True)
             thumb_thread.start()
        return True
    else:
        print(f"Failed to create camera '{camera_name}'. It might already exist.")
        return False

def _download_thumbnails_task_single(camera_obj: cbas.Camera):
    """Helper to download thumbnail for a single camera."""
    output_path = os.path.join(camera_obj.path, "frame.jpg")
    success, _ = _fetch_single_frame_ffmpeg(camera_obj.rtsp_url, output_path)
    if success:
        _process_and_send_thumbnail(camera_obj.name, output_path)

# --- Recording Control ---

@eel.expose
def create_recording_dir(camera_name: str) -> str | bool:
    """Creates a new recording session directory for a camera."""
    if gui_state.proj is None or camera_name not in gui_state.proj.cameras:
        print(f"Camera '{camera_name}' not found for create_recording_dir.")
        return False
    return gui_state.proj.cameras[camera_name].create_recording_dir()

@eel.expose
def get_cbas_status() -> dict:
    """Returns current status of streams and encoding tasks for UI."""
    active_stream_names = []
    if gui_state.proj:
        active_stream_names = list(gui_state.proj.active_recordings.keys())
    
    encode_task_count = 0
    if gui_state.encode_lock: # Check if lock is initialized
        gui_state.encode_lock.acquire()
        encode_task_count = len(gui_state.encode_tasks)
        gui_state.encode_lock.release()

    return {
        "streams": active_stream_names if active_stream_names else False, # JS expects False if empty
        "encode_file_count": encode_task_count
    }

@eel.expose
def start_camera_stream(camera_name: str, destination_dir: str, segment_time_seconds: int) -> bool:
    """Starts recording for a specific camera."""
    if gui_state.proj is None or camera_name not in gui_state.proj.cameras:
        print(f"Camera '{camera_name}' not found for start_camera_stream.")
        return False
    camera = gui_state.proj.cameras[camera_name]
    print(f"Attempting to start recording on camera: {camera_name} to {destination_dir}")
    return camera.start_recording(destination_dir, segment_time_seconds)

@eel.expose
def stop_camera_stream(camera_name: str) -> bool:
    """Stops recording for a specific camera."""
    if gui_state.proj is None or camera_name not in gui_state.proj.cameras:
        print(f"Camera '{camera_name}' not found for stop_camera_stream.")
        return False
    camera = gui_state.proj.cameras[camera_name]
    return camera.stop_recording()

@eel.expose
def get_active_streams() -> list[str] | bool:
    """Returns a list of names of actively recording cameras, or False if none."""
    if gui_state.proj is None: return False
    streams = list(gui_state.proj.active_recordings.keys())
    return streams if streams else False

@eel.expose
def open_camera_live_view(camera_name: str):
    """Opens the camera's RTSP stream in VLC (or default player)."""
    if gui_state.proj is None or camera_name not in gui_state.proj.cameras:
        print(f"Camera '{camera_name}' not found for open_camera_live_view.")
        return
        
    rtsp_url = gui_state.proj.cameras[camera_name].rtsp_url
    # Path to VLC - this should be configurable or found more robustly
    vlc_path_windows = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
    vlc_path_mac = "/Applications/VLC.app/Contents/MacOS/VLC" # Example for macOS
    vlc_path_linux = "vlc" # Assumes VLC is in PATH on Linux

    vlc_exe = None
    if sys.platform == "win32" and os.path.exists(vlc_path_windows):
        vlc_exe = vlc_path_windows
    elif sys.platform == "darwin" and os.path.exists(vlc_path_mac):
        vlc_exe = vlc_path_mac
    elif sys.platform.startswith("linux") and shutil.which("vlc"): # Check if vlc is in PATH
        vlc_exe = "vlc"
    
    if vlc_exe:
        print(f"Opening live view for {camera_name} ({rtsp_url}) with {vlc_exe}...")
        try:
            subprocess.Popen([vlc_exe, rtsp_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Failed to open VLC for {camera_name}: {e}")
            # Fallback or inform user to open manually
            eel.showError(f"Could not open VLC. Please open manually: {rtsp_url}") # Example JS call
    else:
        print(f"VLC not found. Please open RTSP stream manually: {rtsp_url}")
        eel.showError(f"VLC not found. Please open manually: {rtsp_url}") # Example JS call

# This function needs to be exposed if called from JS in record_page.js for camera thumbnails
# However, set_camera_thumbnails is called from Python after download_camera_thumbnails.
# The JS side calls updateImageSrc directly.
# @eel.expose
# def updateImageSrc(camera_name, blob):
#    pass # This would be the Python target if JS called eel.updateImageSrc directly
#         # But it seems Python calls JS: eel.updateImageSrc(camera.name, blob)()