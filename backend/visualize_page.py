import eel
import cbas
import gui_state

import numpy as np

@eel.expose
def get_recording_tree():
    dates = []
    for recording_date in gui_state.proj.recordings:
        insts = []
        for recording_inst in gui_state.proj.recordings[recording_date]:
            models = []
            recording = gui_state.proj.recordings[recording_date][recording_inst]
            for model in recording.classifications:
                labels = gui_state.proj.models[model].config["behaviors"].copy()

                models.append((model, labels))

            insts.append((recording.name, models))
                
        dates.append((recording_date, insts))
    
    return dates

@eel.expose
def make_actogram(
    root,
    sub_dir,
    model,
    behavior,
    framerate,
    binsize,
    start,
    color,
    threshold,
    norm,
    lightcycle,
):
    framerate = int(framerate)
    binsize = int(binsize) * framerate * 60
    start = float(start)

    threshold = float(threshold) / 100
    color = str(color.lstrip("#"))
    color = np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)])

    gui_state.cur_actogram = cbas.Actogram(
        gui_state.proj.recordings[root][sub_dir].path,
        model,
        behavior,
        framerate,
        start,
        binsize,
        color,
        threshold,
        int(norm),
        lightcycle,
        width=500,
        height=500,
    )

    eel.updateActogram(gui_state.cur_actogram.blob)

@eel.expose
def adjust_actogram(framerate, binsize, start, color, threshold, norm, lightcycle):

    framerate = int(framerate)
    binsize = int(binsize) * framerate * 60
    start = float(start)

    threshold = float(threshold) / 100
    color = str(color.lstrip("#"))
    color = np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)])

    actogram = gui_state.cur_actogram

    if actogram is not None:
        actogram.framerate = framerate
        actogram.binsize = binsize
        actogram.start = start
        actogram.color = color
        actogram.threshold = threshold
        actogram.norm = int(norm)
        actogram.lightcycle = lightcycle

        actogram.draw()