import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from tqdm import tqdm
import os
from constants import PROCESSED_DATA_DIR, FILTERED_DATA_DIR, NUM_SAMPLES, SAMPLE_RATE, SNOMED_DICT, LEADS, NUM_LEADS, PLOT_DIR, CLASSIFIER_DATA_DIR
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

def calculate_features(cleaned_signal, sex, age):
    signals, info = nk.ecg_process(cleaned_signal, sampling_rate=500)
    # bataia medie
    # print(info)
    #
    # heartbeat_segments = nk.ecg_segment(
    #     signals,
    #     info["ECG_R_Peaks"],
    #     info["sampling_rate"],
    # )
    #
    # mean = np.empty(493)
    #
    # for i, segment in enumerate(heartbeat_segments.values()):
    #     mean += np.array(segment['ECG_Clean'])
    #
    # avg_heartbeat = mean / len(heartbeat_segments.keys()) 

    # qrs count
    _, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=SAMPLE_RATE)
    qrs_count = len(rpeaks["ECG_R_Peaks"])

    # ventricular rate
    ventricular_rate = (qrs_count * 60) / 10

    # atrial rate
    waves_peak, waves_info = nk.ecg_delineate(
        cleaned_signal,
        rpeaks["ECG_R_Peaks"],
        sampling_rate=500,
        method="dwt",  
        show=False
    )

    p_peaks_indices = np.where(waves_peak["ECG_P_Peaks"] == 1)[0]
    atrial_rate_array = nk.signal_rate(p_peaks_indices, sampling_rate=500)
    atrial_rate = np.mean(atrial_rate_array)

    # qt interval
    r_onsets = np.array(info['ECG_R_Onsets'])
    t_offsets = np.array(info['ECG_T_Offsets'])
    qt_interval = np.nanmean(np.abs(r_onsets - t_offsets)) 

    # qrs duration
    r_onsets = np.array(info['ECG_R_Onsets'])
    r_offsets = np.array(info['ECG_R_Offsets'])
    qrs_duration = np.nanmean(np.abs(r_onsets - r_offsets)) 

    # in loc de q offsets si onsets folosim q_peaks, r_onsets si r_offsets
    r_onsets = np.array(info['ECG_R_Onsets'])
    r_offsets = np.array(info['ECG_R_Offsets'])
    q_peaks = np.array(info['ECG_Q_Peaks'])

    valid_indices = [int(idx) for idx in r_onsets if not np.isnan(idx)]
    r_onsets_value_mean = np.mean([cleaned_signal[idx] for idx in valid_indices])

    valid_indices = [int(idx) for idx in r_offsets if not np.isnan(idx)]
    r_offsets_value_mean = np.mean([cleaned_signal[idx] for idx in valid_indices])

    valid_indices = [int(idx) for idx in q_peaks if not np.isnan(idx)]
    q_peaks_value_mean = np.mean([cleaned_signal[idx] for idx in valid_indices])


    sex_binary = 1 if sex.lower() == "male" else 0

    result = np.array([ventricular_rate, atrial_rate, qrs_count, qt_interval, qrs_duration, r_onsets_value_mean, r_offsets_value_mean, q_peaks_value_mean, sex_binary, age])
    return result

batch=load_batch("01", "010")
batchtest = batch['JS00010']['data'].astype(np.float32)
classifier_data = {}
classifier_data['JS00010'] = {}


classifier_data['JS00010']['features'] = calculate_features(batchtest, batch['JS00010']['sex'], batch['JS00010']['age'])
classifier_data['JS00010']['labels'] = batch['JS00010']['dx']
print(classifier_data)
plt.plot(batchtest)
plt.show()
