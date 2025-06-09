import threading 
from typing import TYPE_CHECKING, Any, List, Dict, Union # Use List and Dict for more specific collection types

# Import project base classes directly for type hinting if they don't cause circular issues at runtime
# For modules like cbas, it's usually fine as gui_state doesn't import cbas classes that import gui_state.
import cbas 

# For cv2, if it's a core dependency, direct import is okay.
import cv2 # For cv2.VideoCapture type hint

if TYPE_CHECKING:
    # These are for type checking phase only to avoid runtime circular imports
    # if workthreads.py were to import gui_state at the top level.
    # String literals for forward references can also be used directly on attributes.
    from workthreads import TrainingThread, EncodeThread, ClassificationThread
    from watchdog.observers import Observer # Assuming watchdog is type-hint friendly
    from cmap import Colormap # If cmap.Colormap is the actual type

# --- Project State ---
proj: cbas.Project | None = None # Holds the currently loaded cbas.Project instance

# --- Labeling Page State ---
label_capture: cv2.VideoCapture | None = None # OpenCV video capture object for the current labeling video
label_dataset: cbas.Dataset | None = None     # The cbas.Dataset object being labeled
label_videos: List[str] = []                 # List of video file paths for the current labeling dataset
label_vid_index: int = -1                    # Index of the current video in label_videos
label_index: int = -1                        # Current frame index in the label_capture

label_start: int = -1                        # Start frame of a new label being defined
label_type: int = -1                         # Index of the behavior type for a new label being defined
if TYPE_CHECKING:
    label_col_map: Colormap | None = None    # Colormap object (from cmap library)
else:
    label_col_map: Any | None = None         # Fallback to Any at runtime if Colormap isn't imported
label_history: List[Dict[str, Any]] = []     # History of added instances (list of dictionaries)

# --- Background Thread Handles & State ---
# These should be initialized by workthreads.start_threads() called from app.py

# Training Thread
training_thread: 'TrainingThread | None' = None # String literal for forward reference

# Encoding Thread
encode_thread: 'EncodeThread | None' = None 
encode_lock: Union[threading.Lock, None] = None  # <<< CHANGED
encode_tasks: List[str] = []                   

# Classification Thread
classify_thread: 'ClassificationThread | None' = None 
classify_lock: Union[threading.Lock, None] = None # <<< CHANGED
classify_tasks: List[str] = []                      

# File System Watcher for Recordings
recording_observer: 'Observer | None' = None 

# --- Visualization Page State ---
cur_actogram: cbas.Actogram | None = None # Holds the currently generated cbas.Actogram object

# --- UI Feedback / Status ---
# No specific state variables here currently; UI updates are via direct eel calls.