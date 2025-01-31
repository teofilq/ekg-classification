import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from tqdm import tqdm
import os
from constants import PROCESSED_DATA_DIR, FILTERED_DATA_DIR, NUM_SAMPLES, SAMPLE_RATE, SNOMED_DICT, LEADS, NUM_LEADS, PLOT_DIR, CLASSIFIER_DATA_DIR
import shutil


def load_batch(batch_dir, batch):
    """
    Încarcă fișierele .npy din folderul de date filtrate. yey
    """
    batch_path = FILTERED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}.npy"

    if not (batch_path.exists()):
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)

    batch = np.load(batch_path, allow_pickle=True).item()
    return batch


def save_classifier_data(batch, batch_dir, batch_name):
    classifier_batch_dir = CLASSIFIER_DATA_DIR / batch_dir
    classifier_batch_dir.mkdir(parents=True, exist_ok=True)

    save_stem = f"batch_{batch_dir}_{batch_name}"

    batch_path = f'{classifier_batch_dir}/{save_stem}'

    np.save(batch_path, batch)


def calculate_features(ecg_signal, sampling_rate=500, sex="male", age=50):
    vent_rate = 0.0
    atrial_rate = 0.0
    qrs_duration = 0.0
    qt_interval = 0.0
    qrs_count = 0.0
    mean_r_onset_sec = 0.0
    mean_r_offset_sec = 0.0

    try:
        signals_peaks, info_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
        rpeaks = info_peaks.get("ECG_R_Peaks", [])
        qrs_count = float(len(rpeaks))

        if qrs_count > 0:
            vent_rate = qrs_count * (60.0 / (len(ecg_signal) / sampling_rate))
            try:
                signals_del, _ = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, method="dwt")
                ppeaks_idx = np.where(signals_del.get("ECG_P_Peaks", []) == 1)[0]
                if len(ppeaks_idx) >= 2:
                    rate_p = nk.signal_rate(ppeaks_idx, sampling_rate=sampling_rate)
                    atrial_rate = float(np.mean(rate_p))

                r_onsets_idx = np.where(signals_del.get("ECG_R_Onsets", []) == 1)[0]
                r_offsets_idx = np.where(signals_del.get("ECG_R_Offsets", []) == 1)[0]
                if len(r_onsets_idx) > 0 and len(r_offsets_idx) > 0:
                    durations = []
                    for onset in r_onsets_idx:
                        off = r_offsets_idx[r_offsets_idx > onset]
                        if len(off) > 0:
                            durations.append(off[0] - onset)
                    if durations:
                        qrs_duration = np.mean(durations) / sampling_rate
                    mean_r_onset_sec = np.mean(r_onsets_idx) / sampling_rate
                    mean_r_offset_sec = np.mean(r_offsets_idx) / sampling_rate

                t_offsets_idx = np.where(signals_del.get("ECG_T_Offsets", []) == 1)[0]
                if len(r_onsets_idx) > 0 and len(t_offsets_idx) > 0:
                    intervals = []
                    for onset in r_onsets_idx:
                        toff = t_offsets_idx[t_offsets_idx > onset]
                        if len(toff) > 0:
                            intervals.append(toff[0] - onset)
                    if intervals:
                        qt_interval = np.mean(intervals) / sampling_rate
            except:
                pass
    except:
        pass

    sex_binary = 1.0 if sex.lower() == "male" else 0.0
    age_float = float(age)

    return np.array([
        vent_rate,        # 0
        atrial_rate,      # 1
        qrs_duration,     # 2
        qt_interval,      # 3
        qrs_count,        # 4
        mean_r_onset_sec, # 5
        mean_r_offset_sec,# 6
        sex_binary,       # 7
        age_float         # 8
    ], dtype=float)


def cleanup_classifier_directory():
    """Șterge toate datele vechi din CLASSIFIER_DATA_DIR pentru a regenera totul corect"""
    if CLASSIFIER_DATA_DIR.exists():
        shutil.rmtree(CLASSIFIER_DATA_DIR)
        print("Am șters folderul CLASSIFIER_DATA_DIR pentru regenerare completă")
    CLASSIFIER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_features_labels():
    
    cleanup_classifier_directory()

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

                    features = classifier_data[record_name]['features'] = calculate_features(
                        batch_data[record_name]['data'],
                        sampling_rate=SAMPLE_RATE,
                        sex=batch_data[record_name]['sex'],
                        age=batch_data[record_name]['age']
                    )
                    classifier_data[record_name]['features'] = features

                    classifier_data[record_name]['labels'] = np.array(
                        [int(x.strip()) for x in batch_data[record_name]['dx'].split(',')],
                        dtype=np.int64
                    )

                save_classifier_data(classifier_data, main_folder, batch_name)
                pbar.update(1)

    
if __name__ == "__main__":
    save_features_labels()