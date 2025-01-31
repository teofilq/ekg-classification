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

def save_classifier_data(batch, batch_dir, batch_name):
    classifier_batch_dir = CLASSIFIER_DATA_DIR / batch_dir
    classifier_batch_dir.mkdir(parents=True, exist_ok=True)

    save_stem = f"batch_{batch_dir}_{batch_name}"

    batch_path = f'{classifier_batch_dir}/{save_stem}'

    np.save(batch_path, batch)


def save_features_labels():
    CLASSIFIER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    main_folders = sorted([f.name for f in FILTERED_DATA_DIR.iterdir() if f.is_dir()])

    for main_folder in main_folders:
        batches = sorted([
            f.name.split('_')[2].split('.')[0]
            for f in (FILTERED_DATA_DIR / main_folder).iterdir()
            if f.is_file() and f.name.endswith(".npy")
        ])
        with tqdm(total=len(batches), desc=f"Processing {main_folder}") as pbar:
            for batch_name in batches:
                batch_data = load_batch(main_folder, batch_name)

                if batch_data is None:
                    print(f"Batch {batch_name} nu a fost găsit în {main_folder}, îl sărim.")
                    pbar.update(1)
                    continue

                classifier_data = {}

                for record_name, record_data in batch_data.items():
                    classifier_data[record_name] = {}

                    features = classifier_data[record_name]['features'] = calculate_features(batch_data[record_name]['data'], batch_data[record_name]['sex'], batch_data[record_name]['age'])
                    classifier_data[record_name]['features'] = features

                    classifier_data[record_name]['labels'] = np.array(
                        [int(x.strip()) for x in batch_data[record_name]['dx'].split(',')],
                        dtype=np.int32  
                    )

                save_classifier_data(classifier_data, main_folder, batch_name)
                pbar.update(1)

    
if __name__ == "__main__":
    save_features_labels()
