import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from constants import PROCESSED_DATA_DIR


def load_ekg_data(batch_data_path, batch_metadata_path):
    """
    Încarcă fișierele .npy generate anterior: semnalele și metadata.
    """
    data = np.load(batch_data_path, allow_pickle=True).item()
    metadata = np.load(batch_metadata_path, allow_pickle=True).item()
    return data, metadata

def load_pacient_data(batch_dir, batch, name):
    batch_data_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_data.npy"
    batch_metadata_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_metadata.npy"

    if not (batch_data_path.exists() and batch_metadata_path.exists()):
        print("Fișierele batch_01_010_data.npy și batch_01_010_metadata.npy nu există în path-ul specificat.")
        exit(1)

    batch_data, batch_metadata = load_ekg_data(batch_data_path, batch_metadata_path)

    if name in batch_data:
        return batch_data[name], batch_metadata[name]

    print(f"Nu exista pacientul {name} in batch-ul {batch}")
    exit(1)

if __name__ == "__main__":
    batch_dir = '01'
    batch = '010'
    patient_name = 'JS00001'
    patient_data, patient_metadata = load_pacient_data(batch_dir, batch, patient_name)
    
