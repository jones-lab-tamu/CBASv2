import threading
import gui_state # Assuming gui_state.py is in the same directory or Python path
import time
import ctypes
import eel # For UI updates
import os
import torch
import cbas # Final name for cbas_rebalanced.py
import yaml

import matplotlib
matplotlib.use("Agg")  # Set the backend to non-GUI mode for servers/threads

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import classifier_head # Final name for classifier_head_corrected.py

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import numpy as np # <<<< ADD THIS LINE

class EncodeThread(threading.Thread):
    def __init__(self, device_str: str): # Use device_str for consistency
        threading.Thread.__init__(self)
        self.device = torch.device(device_str)
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None # Stream only for CUDA
        self.encoder = cbas.DinoEncoder(device_str) # Pass device_str directly

    def run(self):
        while True:
            file_to_encode = None
            gui_state.encode_lock.acquire()
            if gui_state.encode_tasks:
                file_to_encode = gui_state.encode_tasks.pop(0)
            gui_state.encode_lock.release()

            if file_to_encode:
                print(f"[EncodeThread] Starting encoding for: {file_to_encode}")
                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        out_file = cbas.encode_file(self.encoder, file_to_encode)
                else: # CPU or other devices
                    out_file = cbas.encode_file(self.encoder, file_to_encode)
                
                if out_file is None:
                    print(f"[EncodeThread] Failed to encode: {file_to_encode}")
                else:
                    print(f"[EncodeThread] Encoded '{file_to_encode}' as '{out_file}'")
                    gui_state.classify_lock.acquire()
                    gui_state.classify_tasks.append(out_file)
                    gui_state.classify_lock.release()
                    print(f"[EncodeThread] Added {out_file} to classification queue.")
            else:
                time.sleep(1) # Sleep if queue is empty

    def get_id(self):
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
        return None # Should not happen

    def raise_exception(self):
        thread_id = self.get_id()
        if thread_id is not None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
            if res > 1: ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0); print("Exception raise failure in EncodeThread")
        else: print("Could not get thread ID for EncodeThread to raise exception.")


class ClassificationThread(threading.Thread):
    def __init__(self, device_str: str): # Use device_str
        threading.Thread.__init__(self)
        self.device = torch.device(device_str)
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.running = False
        self.torch_model = None
        self.model_meta = None # To store cbas.Model object

    def start_inferring(self, model_obj: cbas.Model, whitelist: list[str]): # Expect cbas.Model object
        gui_state.classify_lock.acquire()
        self.model_meta = model_obj
        self.whitelist = whitelist # Note: whitelist is not currently used in the run loop logic below for picking tasks.

        print(f"[ClassificationThread] Attempting to load model '{self.model_meta.name}' from {self.model_meta.weights_path}")
        try:
            weights = torch.load(self.model_meta.weights_path, map_location=self.device) # Load to target device
            
            self.torch_model = classifier_head.classifier( # Use the imported name
                in_features=768, # Standard DINOv2
                out_features=len(self.model_meta.config["behaviors"]),
                seq_len=self.model_meta.config["seq_len"],
            )
            self.torch_model.load_state_dict(weights) # Assumes weights is a state_dict
            self.torch_model.to(self.device) # Ensure model is on the correct device
            self.torch_model.eval()
            self.running = True
            print(f"[ClassificationThread] Loaded model '{self.model_meta.name}' for classification on {self.device}.")
        except Exception as e:
            print(f"[ClassificationThread] Error loading model '{self.model_meta.name}': {e}")
            self.running = False # Ensure not running if model load fails
        finally:
            gui_state.classify_lock.release()


    def run(self):
        while True:
            file_to_classify = None
            can_run_now = False # Check running status inside lock
            
            gui_state.classify_lock.acquire()
            can_run_now = self.running # Check if allowed to run (model loaded)
            if can_run_now and gui_state.classify_tasks:
                file_to_classify = gui_state.classify_tasks.pop(0)
            gui_state.classify_lock.release()

            if can_run_now and file_to_classify:
                print(f"[ClassificationThread] Starting classification for: {file_to_classify}")
                # Ensure model and meta are available (should be if self.running is True)
                if self.torch_model and self.model_meta:
                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream):
                            out_file = cbas.infer_file(file_to_classify, self.torch_model, self.model_meta.name, 
                                                       self.model_meta.config["behaviors"], self.model_meta.config["seq_len"], 
                                                       device=self.device) # Pass torch.device
                    else: # CPU or other devices
                        out_file = cbas.infer_file(file_to_classify, self.torch_model, self.model_meta.name, 
                                                   self.model_meta.config["behaviors"], self.model_meta.config["seq_len"], 
                                                   device=self.device)
                    
                    if out_file is None:
                        print(f"[ClassificationThread] Failed to classify: {file_to_classify}")
                    else:
                        print(f"[ClassificationThread] Classified '{file_to_classify}' to '{out_file}'")
                    
                    # --- START OF THE FIX ---
                    # After processing a file, check if the queue is now empty.
                    gui_state.classify_lock.acquire()
                    is_queue_empty = not gui_state.classify_tasks
                    gui_state.classify_lock.release()

                    if is_queue_empty:
                        print("[ClassificationThread] Inference queue is empty. Notifying UI.")
                        # Call the same UI update function with a completion message.
                        eel.updateTrainingStatusOnUI(self.model_meta.name, "Inference complete.")()
                    # --- END OF THE FIX ---                        
                        
                        
                else:
                    print(f"[ClassificationThread] Model not ready for {file_to_classify}, re-queuing.")
                    gui_state.classify_lock.acquire()
                    gui_state.classify_tasks.insert(0, file_to_classify) # Add back to front of queue
                    gui_state.classify_lock.release()
                    time.sleep(5) # Wait a bit if model wasn't ready

            else: # Not running or no tasks
                time.sleep(1)

    def get_id(self):
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
        return None

    def raise_exception(self):
        thread_id = self.get_id()
        if thread_id is not None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
            if res > 1: ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0); print("Exception raise failure in ClassificationThread")
        else: print("Could not get thread ID for ClassificationThread to raise exception.")


class TrainingTask():
    def __init__(self, name: str, dataset: cbas.Dataset, behaviors: list[str], 
                 batch_size: int, learning_rate: float, epochs: int, sequence_length: int, training_method = str):
        self.name = name # Dataset name, also used as model name
        self.dataset = dataset # cbas.Dataset object
        self.behaviors = behaviors # List of behavior strings from the dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.training_method = training_method
    
def save_confusion_matrix_plot(confusion_matrix_data: np.ndarray, output_path: str): # Type hint
    if confusion_matrix_data.size == 0:
        print(f"Skipping empty confusion matrix plot for {output_path}")
        return
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data) # Pass the array directly
    fig, ax = plt.subplots(figsize=(8, 7)) # Slightly larger for better label visibility
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation='vertical') # Rotate x-ticks
    ax.set_title("Confusion Matrix") # Add title
    plt.tight_layout() # Adjust layout
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")

def plot_report_list_metric(report_list: list[cbas.PerformanceReport], metric: str,
                            behaviors_plot: list[str], output_dir_path: str):
    if not report_list:
        print(f"No reports to plot for metric {metric}.")
        return

    n_epochs_actual = len(report_list)
    epochs_range = list(range(1, n_epochs_actual + 1))

    plt.figure(figsize=(10, 7))

    for behavior_name in behaviors_plot:
        metric_values = []
        for report_item in report_list:
            value = report_item.sklearn_report.get(behavior_name, {}).get(metric, float('nan'))
            metric_values.append(value)
        
        # USE NP.ISNAN INSTEAD OF MATH.ISNAN
        if not all(np.isnan(x) for x in metric_values):
            plt.plot(epochs_range, metric_values, marker='o', linestyle='-', label=str(behavior_name))

    weighted_avg_values = []
    for report_item in report_list:
        value = report_item.sklearn_report.get('weighted avg', {}).get(metric, float('nan'))
        weighted_avg_values.append(value)
    
    # USE NP.ISNAN HERE AS WELL
    if not all(np.isnan(x) for x in weighted_avg_values):
        plt.plot(epochs_range, weighted_avg_values, marker='x', linestyle='--', label='Weighted Avg')


    plt.xlabel("Epochs")
    plt.ylabel(metric.replace('-', ' ').title())
    plt.title(f"{metric.replace('-', ' ').title()} Over Epochs")
    plt.legend(title="Behaviors", bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    os.makedirs(output_dir_path, exist_ok=True)
    filename = os.path.join(output_dir_path, f"{metric.replace(' ', '_')}-epochs_plot.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved metric plot to {filename}")


class TrainingThread(threading.Thread):
    def __init__(self, device_str: str): # Use device_str
        threading.Thread.__init__(self)
        self.device = torch.device(device_str)
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.training_queue_lock = threading.Lock()
        self.training_queue: list[TrainingTask] = []

    def run(self):
        while True:
            task_to_run = None
            self.training_queue_lock.acquire()
            if self.training_queue:
                task_to_run = self.training_queue.pop(0)
            self.training_queue_lock.release()

            if task_to_run:
                task = task_to_run # Use a local variable for clarity
                print(f"[TrainingThread] --- Starting Training for Dataset: {task.name} ---")
                eel.updateTrainingStatusOnUI(task.name, f"Loading dataset...")() # UI Update

                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        self._execute_training_task(task)
                else:
                    self._execute_training_task(task)
            else:
                time.sleep(1)

    def _update_metrics_in_ui(self, dataset_name, report_dict, behaviors, dataset_obj):
        """Helper to push final metrics to the UI table via Eel."""
        for b_name in behaviors:
            # Update F1, Precision, Recall from the latest training report
            if b_name in report_dict and isinstance(report_dict[b_name], dict):
                f1 = round(report_dict[b_name].get("f1-score", 0.0), 2)
                recall = round(report_dict[b_name].get("recall", 0.0), 2)
                precision = round(report_dict[b_name].get("precision", 0.0), 2)
                eel.updateMetricsOnPage(dataset_name, b_name, "F1 Score", f1)()
                eel.updateMetricsOnPage(dataset_name, b_name, "Recall", recall)()
                eel.updateMetricsOnPage(dataset_name, b_name, "Precision", precision)()
            else:
                eel.updateMetricsOnPage(dataset_name, b_name, "F1 Score", "N/A")()
                eel.updateMetricsOnPage(dataset_name, b_name, "Recall", "N/A")()
                eel.updateMetricsOnPage(dataset_name, b_name, "Precision", "N/A")()

            # Update Train/Test counts from the dataset config (which was updated in label_train_page.py)
            train_count = dataset_obj.config.get("metrics", {}).get(b_name, {}).get("Train #", 0)
            test_count = dataset_obj.config.get("metrics", {}).get(b_name, {}).get("Test #", 0)
            eel.updateMetricsOnPage(dataset_name, b_name, "Train #", train_count)()
            eel.updateMetricsOnPage(dataset_name, b_name, "Test #", test_count)()


    def _execute_training_task(self, task: TrainingTask):
        """
        Helper method to orchestrate the entire training process for a single task,
        including data loading with progress, multi-trial training, and result saving.
        """
        
        # --- 1. Load Dataset with UI Progress Reporting ---
        eel.updateTrainingStatusOnUI(task.name, f"Loading dataset...")()
        
        # Define the callback function that will be called from cbas.py
        def update_ui_progress(percent):
            eel.updateDatasetLoadProgress(task.name, percent)()

        try:
            class_weights = None
            if task.training_method == "weighted_loss":
                print("[TrainingThread] Using 'Weighted Loss' method.")
                train_ds, test_ds, class_weights = gui_state.proj.load_dataset_for_weighted_loss(
                    task.name, seq_len=task.sequence_length, progress_callback=update_ui_progress
                )
                if class_weights is not None:
                    class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            else: # Default to oversampling
                print("[TrainingThread] Using 'Balanced Sampling (Oversampling)' method.")
                train_ds, test_ds = gui_state.proj.load_dataset(
                    task.name, seq_len=task.sequence_length, progress_callback=update_ui_progress
                )

            # Signal to UI that loading is complete
            eel.updateDatasetLoadProgress(task.name, 100)()

            if train_ds is None or len(train_ds) == 0:
                print(f"[TrainingThread] Error: Dataset for '{task.name}' is empty or failed to load.")
                eel.updateTrainingStatusOnUI(task.name, "Error: Dataset empty or failed to load.")()
                return

        except Exception as e:
            print(f"[TrainingThread] Critical error loading dataset {task.name}: {e}")
            eel.updateTrainingStatusOnUI(task.name, f"Fatal Error loading dataset: {e}")()
            # Signal to UI that loading failed
            eel.updateDatasetLoadProgress(task.name, -1)()
            return

        # --- 2. Run Training Trials to Find the Best Model ---
        model_dir = os.path.join(gui_state.proj.models_dir, task.name)
        model_path = os.path.join(model_dir, "model.pth")
        model_config_path = os.path.join(model_dir, "config.yaml")

        overall_best_model_to_save = None
        overall_best_reports_list_to_save = None
        overall_best_epoch_idx_to_save = -1
        highest_f1_across_trials = -1.0
        
        NUM_TRIALS = 10

        for trial_idx in range(NUM_TRIALS):
            status_msg = f"Training Trial {trial_idx + 1}/{NUM_TRIALS}..."
            eel.updateTrainingStatusOnUI(task.name, status_msg)()
            print(f"[TrainingThread] --- Dataset {task.name}: {status_msg} ---")
            
            current_trial_model, current_trial_reports, current_trial_best_epoch_idx = \
                cbas.train_lstm_model(
                    train_ds, test_ds, task.sequence_length, task.behaviors, 
                    lr=task.learning_rate, batch_size=task.batch_size, 
                    epochs=task.epochs, device=self.device,
                    class_weights=class_weights
                )

            if current_trial_model is not None and current_trial_reports and current_trial_best_epoch_idx != -1:
                report_for_f1 = current_trial_reports[current_trial_best_epoch_idx].sklearn_report
                if "weighted avg" in report_for_f1 and "f1-score" in report_for_f1["weighted avg"]:
                    f1_of_this_trials_best_epoch = report_for_f1["weighted avg"]["f1-score"]
                else:
                    f1_of_this_trials_best_epoch = -1.0
                    print(f"Warning: Could not get F1 score for trial {trial_idx+1}, epoch {current_trial_best_epoch_idx+1}")

                print(f"[TrainingThread] --- Trial {trial_idx + 1} Best F1: {f1_of_this_trials_best_epoch:.4f} ---")
                eel.updateTrainingStatusOnUI(task.name, f"Trial {trial_idx + 1} F1: {f1_of_this_trials_best_epoch:.4f}")()

                if f1_of_this_trials_best_epoch > highest_f1_across_trials:
                    highest_f1_across_trials = f1_of_this_trials_best_epoch
                    overall_best_model_to_save = current_trial_model 
                    overall_best_reports_list_to_save = current_trial_reports
                    overall_best_epoch_idx_to_save = current_trial_best_epoch_idx
                    print(f"    >>> New best model found! Trial {trial_idx + 1}, F1: {highest_f1_across_trials:.4f} <<<")
            else:
                print(f"[TrainingThread] --- Trial {trial_idx + 1} did not produce a valid model/report. ---")
                eel.updateTrainingStatusOnUI(task.name, f"Trial {trial_idx + 1} failed or no improvement.")()

        # --- 3. Save Best Model and Artifacts, Update UI with Final Metrics ---
        if overall_best_model_to_save:
            print(f"[TrainingThread] --- Overall Best Model selected. F1: {highest_f1_across_trials:.4f} ---")
            eel.updateTrainingStatusOnUI(task.name, f"Training complete. Best F1: {highest_f1_across_trials:.4f}")()
            
            # Save model and config
            os.makedirs(model_dir, exist_ok=True)
            torch.save(overall_best_model_to_save.state_dict(), model_path)
            model_cfg_to_save = {"seq_len": task.sequence_length, "behaviors": task.behaviors}
            with open(model_config_path, "w") as f: yaml.dump(model_cfg_to_save, f, allow_unicode=True)

            # Save performance reports and plots
            best_report_for_metrics = overall_best_reports_list_to_save[overall_best_epoch_idx_to_save]
            report_dict_for_metrics = best_report_for_metrics.sklearn_report
            
            perf_yaml_path = os.path.join(task.dataset.path, "performance_report.yaml")
            with open(perf_yaml_path, 'w') as f: yaml.dump(report_dict_for_metrics, f, allow_unicode=True)
            
            save_confusion_matrix_plot(best_report_for_metrics.confusion_matrix, os.path.join(task.dataset.path, "confusion_matrix_BEST_MODEL.png"))
            
            cm_dir = os.path.join(task.dataset.path, "epoch_confusion_matrices_of_best_trial")
            os.makedirs(cm_dir, exist_ok=True)
            for i, report in enumerate(overall_best_reports_list_to_save):
                 if report.confusion_matrix.size > 0:
                    save_confusion_matrix_plot(report.confusion_matrix, os.path.join(cm_dir, f"epoch_{i+1}_cm.png"))

            plot_report_list_metric(overall_best_reports_list_to_save, "f1-score", task.behaviors, task.dataset.path)
            plot_report_list_metric(overall_best_reports_list_to_save, "recall", task.behaviors, task.dataset.path)
            plot_report_list_metric(overall_best_reports_list_to_save, "precision", task.behaviors, task.dataset.path)

            # Update project state and UI
            gui_state.proj.models[task.name] = cbas.Model(model_dir)
            self._update_metrics_in_ui(task.name, report_dict_for_metrics, task.behaviors, task.dataset)
            
            print(f"[TrainingThread] --- Model for dataset {task.name} saved and registered. ---")
        else:
            print(f"[TrainingThread] Error: No successful training trial for dataset {task.name} after {NUM_TRIALS} attempts.")
            eel.updateTrainingStatusOnUI(task.name, f"Training failed after {NUM_TRIALS} trials.")()
    
    def queue_task(self, task: TrainingTask): # Added type hint
        self.training_queue_lock.acquire()
        self.training_queue.append(task)
        self.training_queue_lock.release()
        print(f"[TrainingThread] Queued training task for dataset: {task.name}")

    def get_id(self):
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
        return None

    def raise_exception(self):
        thread_id = self.get_id()
        if thread_id is not None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
            if res > 1: ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0); print("Exception raise failure in TrainingThread")
        else: print("Could not get thread ID for TrainingThread to raise exception.")


class VideoFileWatcher(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        print("[VideoFileWatcher] Initialized.")

    def on_any_event(self, event): # Using on_any_event and filtering is often more robust
        if event.event_type == 'created' and not event.is_directory:
            video_path = event.src_path
            # A more robust check for temporary ffmpeg files (often start with . or specific patterns)
            if os.path.basename(video_path).startswith('.') or not video_path.endswith('.mp4'):
                return

            print(f"[VideoFileWatcher] Detected new file: {video_path}")
            # This logic encodes the *previous* segment when a new one is created.
            # This assumes segments are named sequentially (e.g., _00000.mp4, _00001.mp4, ...).
            dirname, basename = os.path.split(video_path)
            try:
                parts = basename.split('.')[0].split('_')
                name_part = parts[0] # e.g. cameraName
                number_str = parts[-1] # e.g. 00001
                current_segment_number = int(number_str)
            except (IndexError, ValueError) as e:
                print(f"[VideoFileWatcher] Could not parse segment number from {basename}: {e}")
                return

            if current_segment_number > 0: # Only process if it's not the first segment
                prev_segment_number_str = str(current_segment_number - 1).zfill(len(number_str)) # Keep same padding
                prev_file_basename = f"{name_part}_{prev_segment_number_str}.mp4"
                prev_file_path = os.path.join(dirname, prev_file_basename)
                
                # Check if previous file actually exists before queueing
                if os.path.exists(prev_file_path):
                    gui_state.encode_lock.acquire()
                    # Avoid duplicate additions if watcher triggers multiple times quickly
                    if prev_file_path not in gui_state.encode_tasks:
                         gui_state.encode_tasks.append(prev_file_path)
                         print(f"[VideoFileWatcher] Queued for encoding: '{prev_file_path}'")
                    else:
                         print(f"[VideoFileWatcher] Already in queue: '{prev_file_path}'")
                    gui_state.encode_lock.release()
                else:
                    print(f"[VideoFileWatcher] Preceding file not found, expected: '{prev_file_path}'")


@eel.expose
def start_threads():
    # Determine device once
    device_string = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_string} for worker threads.")

    gui_state.encode_lock = threading.Lock()
    gui_state.encode_thread = EncodeThread(device_string)
    gui_state.encode_thread.daemon = True # Allow main program to exit even if threads are running
    gui_state.encode_thread.start()
    print("EncodeThread started.")

    gui_state.training_thread = TrainingThread(device_string)
    gui_state.training_thread.daemon = True
    gui_state.training_thread.start()
    print("TrainingThread started.")

    gui_state.classify_lock = threading.Lock()
    gui_state.classify_thread = ClassificationThread(device_string)
    gui_state.classify_thread.daemon = True
    gui_state.classify_thread.start()
    print("ClassificationThread started.")

@eel.expose
def start_recording_watcher():
    if gui_state.proj is None or gui_state.proj.recordings_dir is None:
        print("Cannot start recording watcher: Project or recordings directory not initialized.")
        return
    if not os.path.exists(gui_state.proj.recordings_dir):
        print(f"Cannot start recording watcher: Recordings directory {gui_state.proj.recordings_dir} does not exist.")
        return

    event_handler = VideoFileWatcher()
    gui_state.recording_observer = Observer()
    try:
        gui_state.recording_observer.schedule(event_handler, gui_state.proj.recordings_dir, recursive=True)
        gui_state.recording_observer.start()
        print(f"Recording watcher started on: {gui_state.proj.recordings_dir}")
    except Exception as e:
        print(f"Error starting recording watcher: {e}")
        gui_state.recording_observer = None # Ensure it's None if start fails


def stop_threads():
    print("Attempting to stop worker threads...")
    if gui_state.recording_observer and gui_state.recording_observer.is_alive():
        print("Stopping recording observer...")
        gui_state.recording_observer.stop()
        gui_state.recording_observer.join(timeout=5) # Add timeout
        if gui_state.recording_observer.is_alive(): print("Recording observer did not join.")

    # For other threads, using raise_exception is a bit forceful.
    # A more graceful way would be to use threading.Event or check a stop flag in their run loops.
    # However, given the existing raise_exception mechanism:
    threads_to_stop = [
        ("EncodeThread", gui_state.encode_thread),
        ("ClassificationThread", gui_state.classify_thread),
        ("TrainingThread", gui_state.training_thread)
    ]
    for name, thread_obj in threads_to_stop:
        if thread_obj and thread_obj.is_alive():
            print(f"Stopping {name}...")
            try:
                thread_obj.raise_exception()
                thread_obj.join(timeout=5) # Add timeout
                if thread_obj.is_alive(): print(f"{name} did not join.")
                else: print(f"{name} stopped.")
            except Exception as e:
                print(f"Error stopping {name}: {e}")
    print("Finished attempting to stop threads.")