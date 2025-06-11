"""
Manages all backend logic for the 'Visualize' page of the CBAS application.

This includes functions for:
- Building a hierarchical tree of available classified recordings for selection in the UI.
- Generating new actograms based on user-selected parameters.
"""

import os
import eel
import cbas
import gui_state

@eel.expose
def get_recording_tree() -> list:
    """
    Builds and returns a nested list representing the hierarchy of classified recordings.

    This data structure is consumed by the frontend to build the collapsible selection tree.
    It only includes recordings that have at least one classification result from a known model.

    Returns:
        list: A nested list in the format:
              [ (date_str, [ (session_name, [ (model_name, [behavior_name, ...]) ]) ]) ]
              Returns an empty list if no project is loaded or no classified recordings exist.
    """
    if not gui_state.proj:
        return []

    dates_list = []
    for date_str, sessions in gui_state.proj.recordings.items():
        session_list = []
        for session_name, recording in sessions.items():
            model_list = []
            # Iterate through all classifications found in this recording's folder
            for model_name, classifications in recording.classifications.items():
                # Ensure the model is actually known to the project
                if model_name in gui_state.proj.models:
                    model_config = gui_state.proj.models[model_name].config
                    behaviors = model_config.get("behaviors", [])
                    if behaviors:  # Only add if the model has defined behaviors
                        model_list.append((model_name, behaviors))
            
            if model_list:  # Only add the session if it has at least one valid model
                session_list.append((recording.name, model_list))
        
        if session_list:  # Only add the date if it has at least one valid session
            dates_list.append((date_str, session_list))
    
    return dates_list

# =================================================================
# ACTOGRAM GENERATION LOGIC
# =================================================================

def _generate_actograms_task(root: str, sub_dir: str, model: str, behaviors: list, framerate_val: int,
                            binsize_minutes_val: int, start_val: float, threshold_val: float, lightcycle: str,
                            plot_acrophase: bool, task_id: int):
    """
    (WORKER) This function runs in a background thread and does the actual heavy lifting.
    """
    print(f"Starting actogram task {task_id} for: {behaviors}")

    color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    results = []
    
    try:
        recording = gui_state.proj.recordings.get(root, {}).get(sub_dir)
        if not recording or not os.path.isdir(recording.path):
            raise FileNotFoundError(f"Recording path does not exist for {root}/{sub_dir}")

        all_model_behaviors = gui_state.proj.models[model].config.get("behaviors", [])

        for behavior_name in behaviors:
            # Check if this task has been superseded before starting heavy work
            with gui_state.viz_task_lock:
                if task_id != gui_state.latest_viz_task_id:
                    print(f"Cancelling sub-task for '{behavior_name}' in obsolete task {task_id}.")
                    return # Exit the entire function early

            color_for_plot = None
            if len(behaviors) > 1:
                try:
                    behavior_index = all_model_behaviors.index(behavior_name)
                    color_for_plot = color_map[behavior_index % len(color_map)]
                except (ValueError, IndexError):
                    color_for_plot = '#FFFFFF'

            actogram = cbas.Actogram(
                directory=recording.path, model=model, behavior=behavior_name,
                framerate=framerate_val, start=start_val, binsize_minutes=binsize_minutes_val,
                threshold=threshold_val, lightcycle=lightcycle,
                plot_acrophase=plot_acrophase, base_color=color_for_plot
            )

            if actogram.blob:
                results.append({'behavior': behavior_name, 'blob': actogram.blob})
        
        with gui_state.viz_task_lock:
            is_latest = (task_id == gui_state.latest_viz_task_id)

        if is_latest:
            print(f"Task {task_id} is the latest. Sending results to UI.")
            eel.updateActogramDisplay(results, task_id)()
        else:
            print(f"Discarding results for obsolete actogram task {task_id}.")

    except Exception as e:
        print(f"Error in generate_actograms task {task_id}: {e}")
        with gui_state.viz_task_lock:
            if task_id == gui_state.latest_viz_task_id:
                eel.updateActogramDisplay([], task_id)()


@eel.expose
def generate_actograms(root: str, sub_dir: str, model: str, behaviors: list, framerate: str,
                       binsize_from_gui: str, start: str, threshold: str, lightcycle: str,
                       plot_acrophase: bool, task_id: int):
    """
    (LAUNCHER) This function is now a lightweight launcher. It immediately spawns
    the worker task in the background and returns, keeping the UI responsive.
    """
    with gui_state.viz_task_lock:
        gui_state.latest_viz_task_id = task_id
    
    # Parse parameters here so we don't pass strings to the worker
    framerate_val = int(framerate)
    binsize_minutes_val = int(binsize_from_gui)
    start_val = float(start)
    threshold_val = float(threshold) / 100.0

    # Use eel.spawn to run the worker function in a non-blocking background thread
    eel.spawn(
        _generate_actograms_task,
        root, sub_dir, model, behaviors, framerate_val,
        binsize_minutes_val, start_val, threshold_val, lightcycle,
        plot_acrophase, task_id
    )