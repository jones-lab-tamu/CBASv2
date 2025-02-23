import cbas
import cv2

proj: cbas.Project = None

label_capture = None
label_dataset: cbas.Dataset
label_videos: list[str] = []
label_vid_index: int = -1
label_index: int = -1

label_start: int = -1
label_type: int = -1
label_col_map = None
label_history = []

training_thread = None

encode_thread = None
encode_lock = None
encode_tasks: list[str] = []

classify_thread = None
classify_lock = None
classify_tasks = []

recording_observer = None

cur_actogram = None