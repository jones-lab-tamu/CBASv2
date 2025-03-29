import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

def actogram(file_path: str):
#    file_path = r"G:\Shared drives\TAMU Jones Lab\Data\Arthur Mayo\Analysis.software\data\Perry\OVX_DD\exp2\csv\Exp2_DD_2_eating.csv"
    data = pd.read_csv(file_path, header=None)
    start_hour_str = data.iloc[2, 0]
    bin_size_minutes = int(float(data.iloc[3, 0])) // 4
    event_type = data.iloc[0, 0]
    tau = 24 #This value makes the X axis 24h


    try:
        start_datetime = datetime.strptime(
            f"{data.iloc[1, 0]} {data.iloc[2, 0]}",
            "%d-%b-%Y %H:%M"
        )
    except Exception as e:
        raise ValueError(f"Error parsing start datetime: {e}")


    # Process event counts from row 8 onward (column 0)
    raw_event_counts = data.iloc[7:, 0]
    event_counts = pd.to_numeric(raw_event_counts, errors='coerce')
    if event_counts.isnull().any():
        print("Warning: Some event counts could not be converted; setting them to 0.")
        event_counts = event_counts.fillna(0).astype(int)
    else:
        event_counts = event_counts.astype(int)


    # Process light values from row 8 onward (column 1); if missing, use zeros.
    if data.shape[1] > 1:
        raw_light = data.iloc[7:, 1]
        light_values = pd.to_numeric(raw_light, errors='coerce').fillna(0).astype(int)
        light_values = np.where(light_values > 0, 1, 0)
    else:
        light_values = np.zeros_like(event_counts)


    # Determine bins per day.
    bins_per_day = (60 // bin_size_minutes) * 24
    start_hour_int, start_minute_int = map(int, start_hour_str.split(':'))
    start_minutes = start_hour_int * 60 + start_minute_int
    bins_to_pad_start = start_minutes // bin_size_minutes


    # Pad event counts.
    padded_event_counts = np.pad(event_counts, (bins_to_pad_start, 0), 'constant')
    num_days = int(np.ceil(len(padded_event_counts) / bins_per_day))
    padded_event_counts = np.pad(
        padded_event_counts,
        (0, num_days * bins_per_day - len(padded_event_counts)),
        'constant'
    )


    # Pad light values in the same way.
    padded_light = np.pad(light_values, (bins_to_pad_start, 0), 'constant')
    padded_light = np.pad(
        padded_light,
        (0, num_days * bins_per_day - len(padded_light)),
        'constant'
    )




    if tau is None:
        raise ValueError("Tau has not been determined yet.")
    bins_per_period = bins_per_day
    events_per_period = [
        padded_event_counts[i * bins_per_period:(i + 1) * bins_per_period]
        for i in range(len(padded_event_counts) // bins_per_period)
    ]
    num_periods = len(events_per_period)
    fig_height = max(6, num_periods * 0.3)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    cax = ax.imshow(events_per_period, aspect='auto', cmap='Greys',
                    interpolation='none', extent=[0, tau, num_periods, 0])


    light_per_period = [
        padded_light[i * bins_per_period:(i + 1) * bins_per_period]
        for i in range(len(padded_light) // bins_per_period)
    ]
    light_cmap = LinearSegmentedColormap.from_list("light_cmap", ["white", "yellow"])
    ax.imshow(light_per_period, aspect='auto', cmap=light_cmap,
                interpolation='none', extent=[0, tau, num_periods, 0],
                alpha=0.5)
    ax.set_xticks(np.arange(0, tau + 1, 2))
    ax.set_xticklabels([f"{int(tick % 24):02d}" for tick in np.arange(0, tau + 1, 2)])
    ax.set_yticks(np.arange(0.5, num_periods, 1))
    ax.set_yticklabels([f"Period {i+1}" for i in range(num_periods)])
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Period')
    ax.set_title(f'Single-Plotted Actogram for {event_type}')
    plt.colorbar(cax, ax=ax, orientation='vertical', label='Event Count')


    return fig