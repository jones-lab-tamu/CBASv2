import eel
import cbas # Assuming cbas.py is the final name
import gui_state
import workthreads # Assuming workthreads.py is the final name
import os

@eel.expose
def create_project(parent_directory: str, project_name: str) -> tuple[bool, dict | None]:
    """
    Creates a new project.
    Returns: (success_flag, project_info_dict or None)
    """
    print(f"Attempting to create project '{project_name}' in '{parent_directory}'")
    gui_state.proj = cbas.Project.create_project(parent_directory, project_name)

    if gui_state.proj is None:
        print(f"Failed to create project '{project_name}'.")
        return False, None

    print(f"Project '{project_name}' created successfully.")
    project_info = {
        "project_path": gui_state.proj.path, # Changed key for clarity
        "cameras_dir": gui_state.proj.cameras_dir,
        "recordings_dir": gui_state.proj.recordings_dir,
        "models_dir": gui_state.proj.models_dir,
        "data_sets_dir": gui_state.proj.datasets_dir, # Changed key for clarity
    }
    return True, project_info


@eel.expose
def load_project(path: str) -> tuple[bool, dict | None]:
    """
    Loads an existing project.
    Returns: (success_flag, project_info_dict or None)
    """
    print(f"Attempting to load project from: {path}")
    try:
        gui_state.proj = cbas.Project(path)
        print(f"Project loaded successfully: {gui_state.proj.path}")
    except cbas.InvalidProject as e:
        print(f"Error: {e}. Path is not a valid project.")
        return False, None
    except Exception as e: # Catch other potential errors during Project init
        print(f"An unexpected error occurred while loading project {path}: {e}")
        return False, None
    
    # Queue up all unencoded files to be encoded by a workthread
    if gui_state.proj and gui_state.encode_lock is not None: # Ensure lock is initialized
        files_to_queue_for_encoding = []
        for day_recordings in gui_state.proj.recordings.values():
            for recording_instance in day_recordings.values():
                files_to_queue_for_encoding.extend(recording_instance.unencoded_files)
        
        if files_to_queue_for_encoding:
            gui_state.encode_lock.acquire()
            for f_path in files_to_queue_for_encoding:
                if f_path not in gui_state.encode_tasks : # Avoid duplicates
                    gui_state.encode_tasks.append(f_path)
            gui_state.encode_lock.release()
            print(f"Queued {len(files_to_queue_for_encoding)} unencoded files for processing.")
        else:
            print("No unencoded files found to queue.")
    else:
        print("Project or encode_lock not initialized; cannot queue unencoded files.")


    # Start the recording watcher only if project loaded successfully
    if gui_state.proj:
        try:
            if gui_state.recording_observer is None or not gui_state.recording_observer.is_alive():
                 print("Starting recording watcher...")
                 workthreads.start_recording_watcher() # This function needs to handle if observer already started
            else:
                 print("Recording watcher already active.")
        except Exception as e:
            print(f"Error trying to start recording watcher: {e}")


    project_info = {
        "project_path": gui_state.proj.path,
        "cameras_dir": gui_state.proj.cameras_dir,
        "recordings_dir": gui_state.proj.recordings_dir,
        "models_dir": gui_state.proj.models_dir,
        "data_sets_dir": gui_state.proj.datasets_dir,
    }
    return True, project_info