import eel
import cbas

import gui_state

@eel.expose
def create_project(parent_directory, project_name):
    gui_state.proj = cbas.Project.create_project(parent_directory, project_name)

    if gui_state.proj is None:
        return False, None

    return True, {
        "project": gui_state.proj.path,
        "cameras": gui_state.proj.cameras_dir,
        "recordings": gui_state.proj.recordings_dir,
        "models": gui_state.proj.models_dir,
        "data_sets": gui_state.proj.datasets_dir,
    }

@eel.expose
def load_project(path):
    try:
        gui_state.proj = cbas.Project(path)
    except cbas.InvalidProject as invalid_project:
        print(invalid_project.path)

        return False, None

    # TODO: This doesn't need to return all this state.
    return True, {
        "project": gui_state.proj.path,
        "cameras": gui_state.proj.cameras_dir,
        "recordings": gui_state.proj.recordings_dir,
        "models": gui_state.proj.models_dir,
        "data_sets": gui_state.proj.datasets_dir,
    }