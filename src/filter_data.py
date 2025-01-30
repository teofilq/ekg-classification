import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from tqdm import tqdm
from constants import PROCESSED_DATA_DIR, FILTERED_DATA_DIR, NUM_SAMPLES, SAMPLE_RATE, SNOMED_DICT, LEADS, NUM_LEADS, PLOT_DIR

def load_ekg_data(batch_dir, batch):
    """
    Încarcă fișierele .npy generate anterior: semnalele și metadata.
    """
    batch_data_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_data.npy"
    batch_metadata_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_metadata.npy"

    if not (batch_data_path.exists() and batch_metadata_path.exists()):
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)

    data = np.load(batch_data_path, allow_pickle=True).item()
    metadata = np.load(batch_metadata_path, allow_pickle=True).item()
    return data, metadata

def filter_patient_data(patient_data):
    filtered_signal = nk.ecg_clean(patient_data, sampling_rate=SAMPLE_RATE, method="neurokit")

    return filtered_signal

def remove_offset_patient_data(patient_data, offset):
    adjusted_data = patient_data - offset

    return adjusted_data

def leave_second_derivation(patient_data):
    return patient_data[:, 1]

def save_filtered_batch(batch, batch_dir, batch_name):
    filtered_batch_dir = FILTERED_DATA_DIR / batch_dir
    filtered_batch_dir.mkdir(parents=True, exist_ok=True)

    save_stem = f"batch_{batch_dir}_{batch_name}"

    batch_path = f'{filtered_batch_dir}/{save_stem}'

    np.save(batch_path, batch)

def process_and_save_filtered_batches():
    """Parcurge toate batch-urile salvate în PROCESSED și le salvează filtrate în FILTERED."""
    FILTERED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    main_folders = sorted([f.name for f in PROCESSED_DATA_DIR.iterdir() if f.is_dir()])
    print(main_folders)

    for main_folder in main_folders:
        batches = sorted([
            f.name.split('_')[2].split('.')[0]
            for f in (PROCESSED_DATA_DIR / main_folder).iterdir()
            if f.is_file() and "_data.npy" in f.name
        ])

        print(f"\n🔹 Procesăm batch-urile din folderul: {main_folder}")

        with tqdm(total=len(batches), desc=f"Processing {main_folder}") as pbar:
            for batch_name in batches:
                batch_data, batch_metadata = load_ekg_data(main_folder, batch_name)

                if batch_data is None or batch_metadata is None:
                    print(f"Batch {batch_name} nu a fost găsit în {main_folder}, îl sărim.")
                    pbar.update(1)
                    continue

                batch = {}

                for record_name, record_data in batch_data.items():
                    batch[record_name] = {}
                    batch[record_name]['offset'] = batch_metadata[record_name]['offsets'][1]
                    batch[record_name]['sex'] = batch_metadata[record_name]['sex']
                    batch[record_name]['age'] = batch_metadata[record_name]['age']
                    batch[record_name]['dx'] = batch_metadata[record_name]['dx']

                    second_derivation = leave_second_derivation(record_data)
                    second_derivation = remove_offset_patient_data(second_derivation, batch[record_name]['offset'])
                    second_derivation = filter_patient_data(second_derivation)

                    batch[record_name]['data'] = second_derivation

                save_filtered_batch(batch, main_folder, batch_name)
                pbar.update(1)

if __name__ == "__main__":
    process_and_save_filtered_batches()


