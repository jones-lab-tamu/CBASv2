"""
Manages all backend logic for the 'Visualize' page of the CBAS application.

This includes functions for:
- Building a hierarchical tree of available classified recordings for selection in the UI.
- Generating new actograms based on user-selected parameters.
- Adjusting existing actograms with new parameters.
"""

import os
import eel

# Local application imports
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


@eel.expose
def make_actogram(root: str, sub_dir: str, model: str, behavior: str, framerate: str,
                  binsize_from_gui: str, start: str, threshold: str, lightcycle: str,
                  plot_acrophase: bool):
    """
    Creates a new Actogram instance based on UI parameters and sends the
    resulting image blob back to the frontend.

    Args:
        root (str): The date directory name (e.g., '20250101').
        sub_dir (str): The session directory name (e.g., 'Camera1-120000-PM').
        model (str): The name of the model whose classifications to use.
        behavior (str): The specific behavior to plot.
        framerate (str): The video framerate.
        binsize_from_gui (str): The bin size in minutes.
        start (str): The start hour for the plot (0-24).
        threshold (str): The probability threshold for an event (1-100).
        lightcycle (str): The light cycle pattern ('LD', 'LL', or 'DD').
        plot_acrophase (bool): Whether to calculate and plot the acrophase.
    """
    print(f"make_actogram called for: {root}/{sub_dir} | {model} - {behavior}")
    try:
        # --- Parameter Parsing and Validation ---
        framerate_val = int(framerate)
        binsize_minutes_val = int(binsize_from_gui)
        start_val = float(start)
        threshold_val = float(threshold) / 100.0
        
        # Ensure the specified recording exists in the project state
        recording = gui_state.proj.recordings.get(root, {}).get(sub_dir)
        if not recording or not os.path.isdir(recording.path):
            print(f"Error: Recording path does not exist for {root}/{sub_dir}")
            eel.updateActogramDisplay(None)()
            return

        # --- Actogram Creation ---
        gui_state.cur_actogram = cbas.Actogram(
            directory=recording.path,
            model=model,
            behavior=behavior,
            framerate=framerate_val,
            start=start_val,
            binsize_minutes=binsize_minutes_val,
            threshold=threshold_val,
            lightcycle=lightcycle,
            plot_acrophase=plot_acrophase
        )

        # --- Send Result to Frontend ---
        if gui_state.cur_actogram and gui_state.cur_actogram.blob:
            eel.updateActogramDisplay(gui_state.cur_actogram.blob)()
        else:
            print("Actogram generation failed or produced no image blob.")
            eel.updateActogramDisplay(None)()

    except Exception as e:
        print(f"Error in make_actogram: {e}")
        eel.updateActogramDisplay(None)()


@eel.expose
def adjust_actogram(framerate: str, binsize_from_gui: str, start: str, threshold: str,
                    lightcycle: str, plot_acrophase: bool):
    """
    Adjusts the currently displayed actogram by re-generating it with new parameters.
    This function reuses the identifiers from the existing actogram object.
    """
    actogram = gui_state.cur_actogram
    if actogram is None:
        print("No current actogram to adjust.")
        return

    print("Adjusting current actogram with new parameters...")
    
    # To get the root and sub_dir, we parse them from the actogram's directory path.
    # This assumes a structure like .../project/recordings/DATE/SESSION
    try:
        path_parts = os.path.normpath(actogram.directory).split(os.sep)
        sub_dir = path_parts[-1]
        root = path_parts[-2]
    except IndexError:
        print(f"Error: Could not parse root and sub_dir from path: {actogram.directory}")
        return

    # Re-call the main creation function with the old identifiers and new parameters.
    make_actogram(
        root=root,
        sub_dir=sub_dir,
        model=actogram.model,
        behavior=actogram.behavior,
        framerate=framerate,
        binsize_from_gui=binsize_from_gui,
        start=start,
        threshold=threshold,
        lightcycle=lightcycle,
        plot_acrophase=plot_acrophase
    )