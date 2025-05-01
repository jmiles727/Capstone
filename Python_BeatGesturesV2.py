import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point2) - np.array(point1))**2))

def detect_gestures(motion, threshold, min_duration):
    gestures = []
    in_gesture = False
    gesture_start_index = -1
    consecutive_above_threshold = 0

    for i, move in enumerate(motion):
        if move > threshold:
            consecutive_above_threshold += 1
            if not in_gesture and consecutive_above_threshold >= min_duration:
                in_gesture = True
                gesture_start_index = i - min_duration + 1 # Start of the sustained period
        else:
            if in_gesture:
                gestures.append((gesture_start_index, i))
                in_gesture = False
            consecutive_above_threshold = 0
    # Handle a gesture that might end at the very end of the motion
    if in_gesture:
        gestures.append((gesture_start_index, len(motion)))
    return gestures

def analyze_beat_gestures(data_file, skip_rows=175, remove_last=150, prominence_threshold=5):
    """
    Analyzes beat gesture frequency based on Left Hand Y movements.

    Args:
        data_file (str): Path to OpenPose data.
        skip_rows (int): Number of initial rows to skip (default 175).
        remove_last (int): Number of last rows to remove (default 150).
        prominence_threshold (int): Minimum vertical prominence (in pixels)
                                    for a peak to be considered a beat gesture.

    Returns:
        pandas.DataFrame: DataFrame containing information about detected peaks (potential beat gestures).
    """
    print(f"Analyzing beat gestures in: {data_file}")
    try:
        df = pd.read_csv(data_file, sep='\t')
        print(f"Total {len(df)} lines loaded.")
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        return None

    # Process data (skip and remove rows)
    df_processed = df.iloc[skip_rows:-remove_last].copy().reset_index(drop=True)
    print(f"Processing {len(df_processed)} lines after skipping {skip_rows} and removing last {remove_last}.")

    if df_processed.empty:
        print("No data to process after skipping and removing rows.")
        return None

    left_hand_y = df_processed['part_7_y'].values

    # Find peaks (potential rises in beat gestures)
    peaks, _ = find_peaks(left_hand_y, prominence=prominence_threshold)
    print(f"Found {len(peaks)} potential beat gesture peaks with prominence >= {prominence_threshold} pixels.")

    # Find valleys (potential dips in beat gestures)
    valleys, _ = find_peaks(-left_hand_y, prominence=prominence_threshold)
    print(f"Found {len(valleys)} potential beat gesture valleys with prominence >= {prominence_threshold} pixels.")

    # Create a DataFrame of the detected peaks and valleys
    peaks_df = pd.DataFrame({'frame': peaks + skip_rows, 'type': 'peak', 'y_value': left_hand_y[peaks]})
    valleys_df = pd.DataFrame({'frame': valleys + skip_rows, 'type': 'valley', 'y_value': left_hand_y[valleys]})
    beat_gestures_df = pd.concat([peaks_df, valleys_df]).sort_values(by='frame').reset_index(drop=True)

    print("\nPotential Beat Gestures (Peaks and Valleys):")
    print(beat_gestures_df.head())

    # Visualize Left Hand Y and the detected peaks
    plt.figure(figsize=(12, 6))
    plt.plot(df_processed.index + skip_rows, left_hand_y, label='Left Hand Y')
    plt.scatter(peaks + skip_rows, left_hand_y[peaks], color='red', marker='^', label='Peaks (Rises)')
    plt.scatter(valleys + skip_rows, left_hand_y[valleys], color='green', marker='v', label='Valleys (Dips)')
    plt.xlabel("Frame Number (Original)")
    plt.ylabel("Left Hand Y Pixel Coordinate")
    plt.title(f"Left Hand Y and Detected Potential Beat Gestures (Prominence >= {prominence_threshold})")
    plt.legend()
    plt.grid(True)
    plt.show()

    return beat_gestures_df

# Example Usage
data_file = r"C:\Austin Full Speech Data.txt"
beat_gestures_df = analyze_beat_gestures(data_file, prominence_threshold=5)

if beat_gestures_df is not None:
    print(f"\nTotal potential beat gesture events (peaks and valleys): {len(beat_gestures_df)}")
