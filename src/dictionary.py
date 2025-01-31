import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from tqdm import tqdm
import os
from constants import PROCESSED_DATA_DIR, FILTERED_DATA_DIR, NUM_SAMPLES, SAMPLE_RATE, SNOMED_DICT, LEADS, NUM_LEADS, PLOT_DIR
from utils import ECGAnalyzer


def load_batch(batch_dir, batch):
    """
    Încarcă fișierele .npy generate anterior: semnalele și metadata.
    """
    batch_path = FILTERED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}.npy"

    if not (batch_path.exists()):
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)

    batch = np.load(batch_path, allow_pickle=True).item()
    return batch



batch=load_batch("01", "010")
batchtest = batch['JS00001']['data']


# # Inițializează obiectul ECGAnalyzer cu semnalul ECG
# ecg_analyzer = ECGAnalyzer(
#     ecg_signal=batchtest,
#     sampling_rate=SAMPLE_RATE,
#     method_peaks="neurokit",
#     method_delineate="dwt"
# )

# # --- Print QRS Count ---
# print("🔹 QRS Count:", ecg_analyzer.get_qrs_count())

# # --- Print Ventricular Rate (bpm) ---
# ventricular_rate = ecg_analyzer.get_ventricular_rate()
# print("🔹 Ventricular Rate (bpm):", ventricular_rate)

# # --- Print Atrial Rate (bpm) ---
# atrial_rate = ecg_analyzer.get_atrial_rate()
# if atrial_rate is None:
#     print("🔹 Atrial Rate: No P-peaks detected or not enough data.")
# else:
#     print("🔹 Atrial Rate (bpm):", atrial_rate)

# # --- Print QRS Durations ---
# qrs_durations = ecg_analyzer.get_qrs_duration()
# if qrs_durations is None:
#     print("🔹 QRS Duration: Could not be determined.")
# else:
#     print("🔹 QRS Duration (seconds per beat):", qrs_durations)

# # --- Print QT Intervals ---
# qt_intervals = ecg_analyzer.get_qt_intervals()
# if qt_intervals is None:
#     print("🔹 QT Interval: Could not be determined.")
# else:
#     print("🔹 QT Interval (seconds per beat):", qt_intervals)

# # --- Print Q Onset / Q Offset ---
# q_onset, q_offset = ecg_analyzer.get_q_onset_offsets()
# if q_onset is None:
#     print("🔹 Q Onset: Not detected.")
# else:
#     print("🔹 Q Onset Indices:", q_onset)

# if q_offset is None:
#     print("🔹 Q Offset: Not detected.")
# else:
#     print("🔹 Q Offset Indices:", q_offset)
plt.figure(figsize=(15, 4))
plt.plot(batchtest)
plt.title('ECG Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
