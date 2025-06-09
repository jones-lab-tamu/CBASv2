import eel
import cbas
import gui_state
import numpy as np
import os

@eel.expose
def get_recording_tree():
    """
    Builds and returns a nested list representing the hierarchy of classified recordings.
    Format: [ (date, [ (session_name, [ (model_name, [behavior_name, ...]) ]) ]) ]
    """
    dates = []
    if gui_state.proj is None:
        return dates

    # Use .values() for more Pythonic iteration
    for date_str, sessions in gui_state.proj.recordings.items():
        session_list = []
        for session_name, recording in sessions.items():
            model_list = []
            for model_name, classifications in recording.classifications.items():
                if model_name in gui_state.proj.models:
                    behaviors = gui_state.proj.models[model_name].config.get("behaviors", [])
                    if behaviors: # Only add if there are behaviors defined
                        model_list.append((model_name, behaviors))
            
            if model_list: # Only add session if it has classified models
                session_list.append((recording.name, model_list))
        
        if session_list: # Only add date if it has sessions with models
            dates.append((date_str, session_list))
    
    return dates


@eel.expose
def make_actogram(
    root,
    sub_dir,
    model,
    behavior,
    framerate,
    binsize_from_gui,
    start,
    threshold,
    lightcycle,
    plot_acrophase
    # Obsolete 'color' and 'norm' parameters are REMOVED
):
    """Creates a new Actogram instance based on UI parameters."""
    print(f"make_actogram called with: root={root}, sub_dir={sub_dir}, model={model}, behavior={behavior}")
    try:
        # --- Parameter Parsing ---
        framerate_val = int(framerate)
        binsize_minutes_val = int(binsize_from_gui)
        start_val = float(start)
        threshold_val = float(threshold) / 100.0
        
        recording_path = gui_state.proj.recordings[root][sub_dir].path
        if not os.path.isdir(recording_path):
            print(f"Error: Recording path does not exist: {recording_path}")
            eel.updateActogramDisplay(None)()
            return

        # --- Actogram Creation ---
        # Note: color and norm are no longer passed to the constructor.
        gui_state.cur_actogram = cbas.Actogram(
            directory=recording_path,
            model=model,
            behavior=behavior,
            framerate=framerate_val,
            start=start_val,
            binsize_minutes=binsize_minutes_val,
            threshold=threshold_val,
            lightcycle=lightcycle,
            plot_acrophase=plot_acrophase
        )

        if gui_state.cur_actogram and gui_state.cur_actogram.blob:
            eel.updateActogramDisplay(gui_state.cur_actogram.blob)()
        else:
            print("Actogram generation failed or produced no blob.")
            eel.updateActogramDisplay(None)()

    except Exception as e:
        print(f"Error in make_actogram: {e}")
        eel.updateActogramDisplay(None)()

@eel.expose
def adjust_actogram(framerate, binsize_from_gui, start, threshold, lightcycle, plot_acrophase):
    """
    Adjusts the current actogram by creating a new one with updated parameters.
    """
    print(f"adjust_actogram called with: framerate={framerate}, binsize_from_gui={binsize_from_gui}, start={start}")
    
    # Get the key identifiers from the existing actogram object
    actogram = gui_state.cur_actogram
    if actogram is None:
        print("No current actogram to adjust.")
        return

    # To get the root and sub_dir, we need to parse them from the actogram's directory path.
    # This assumes a structure like .../recordings/root/sub_dir
    try:
        path_parts = os.path.normpath(actogram.directory).split(os.sep)
        sub_dir = path_parts[-1]
        root = path_parts[-2]
    except IndexError:
        print(f"Error: Could not parse root and sub_dir from path: {actogram.directory}")
        return

    # Now, simply call make_actogram with the existing identifiers and the new UI parameters.
    # This rebuilds the entire actogram from scratch, ensuring all logic is re-run.
    make_actogram(
        root,
        sub_dir,
        actogram.model,
        actogram.behavior,
        framerate,
        binsize_from_gui,
        start,
        threshold,
        lightcycle,
        plot_acrophase
    )