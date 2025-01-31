# ***************************************************************************
# # Am folosit ca model:
# https://github.com/zheng120/ECGDenoisingTool/blob/master/NLMDenoising20191120.py
# ***************************************************************************
import numpy as np
import math
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal
from tqdm import tqdm

from constants import (
    PROCESSED_DATA_DIR,
    FILTERED_DATA_DIR,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SNOMED_DICT,
    LEADS,
    NUM_LEADS,
    PLOT_DIR
)


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Folosim device={device} pentru calcule NLM.")


def butter_lowpass_filter(ecg_signal, fs=500):
    fp = 50.0
    fstop = 60.0
    rp = 1.0
    rs = 2.5

    wp = fp / (fs / 2.0)
    ws = fstop / (fs / 2.0)

    n, wn = signal.buttord(wp, ws, rp, rs)
    b, a = signal.butter(n, wn, btype='low')
    filtered = signal.filtfilt(b, a, ecg_signal)  # CPU
    return filtered


def remove_baseline_loess(ecg_signal):
    x = np.arange(len(ecg_signal))
    baseline = sm.nonparametric.lowess(ecg_signal, x, frac=0.1, it=2, return_sorted=False)
    return ecg_signal - baseline


def estimate_noise_std(ecg_signal):
    """
    NoisSTD = 1.4826 * median(|D - median(D)|),
    unde D[k] = (2*ecg[k] - ecg[k-1] - ecg[k+1]) / sqrt(6).
    """
    ecg_signal = np.array(ecg_signal, dtype=float)
    D = ecg_signal.copy()
    for k in range(1, len(ecg_signal) - 1):
        D[k] = (2.0 * ecg_signal[k] - ecg_signal[k - 1] - ecg_signal[k + 1]) / math.sqrt(6)
    medD = np.median(D)
    mad = np.median(np.abs(D - medD))
    return 1.4826 * mad

def create_patches_1d(signal_t, patchHW):
    """
    Construiește, prin clamping la margini, un tablou de patchuri 2D: [N, 2*patchHW+1].
    patch[i, :] = semnal[i - patchHW : i + patchHW + 1] (cu indecșii clampați).
    
    signal_t: tensor [N] pe device
    patchHW : int (ex. 10)
    """
    N = signal_t.shape[0]
    patch_size = 2 * patchHW + 1

    base_idx = torch.arange(N, device=device).unsqueeze(1)
    offsets = torch.arange(-patchHW, patchHW+1, device=device) 
    index_mat = base_idx + offsets 
    index_mat = torch.clamp(index_mat, 0, N-1)

    patches = signal_t[index_mat]
    return patches

def NLM_1d_gpu_vectorized(ecg_signal, Nvar, P, PatchHW):
    """
    Versiune vectorizată pe GPU, cu toată comparația i-j (±P) făcută prin broadcast.

    - ecg_signal: array 1D
    - Nvar: 1.5 * NoisSTD (scala Gauss)
    - P: 5000 (fereastră de căutare)
    - PatchHW: 10 (lățime patch)
    """
    # Convertim semnalul la tensor pe device
    sig_t = torch.tensor(ecg_signal, device=device, dtype=torch.float32)
    N = sig_t.shape[0]
    patch_size = 2*PatchHW + 1

    patches = create_patches_1d(sig_t, PatchHW) 

    patches_i = patches[:, None, :]  
    patches_j = patches[None, :, :]  
    diff_sq = (patches_i - patches_j).pow(2) 

    dist = diff_sq.sum(dim=-1)

    h = 2.0 * patch_size * (Nvar**2)
    w = torch.exp(-dist / h)  

    i_idx = torch.arange(N, device=device).unsqueeze(1)  # [N,1]
    j_idx = torch.arange(N, device=device).unsqueeze(0)  # [1,N]
    mask = (j_idx - i_idx).abs() <= P  
    w = w * mask 


    numerator = w * sig_t.unsqueeze(0)  
    numerator_sum = numerator.sum(dim=1)
    w_sum = w.sum(dim=1)


    eps = 1e-12
    out_t = numerator_sum / (w_sum + eps)  # [N]

    out_t[:PatchHW] = sig_t[:PatchHW]
    out_t[-PatchHW:] = sig_t[-PatchHW:]

    return out_t.cpu().numpy()


def apply_nlm_filter_gpu(ecg_signal):
    """
    Wrapper:
      - Calculează NoisSTD
      - NLM vectorizat pe GPU
      - P=5000, PatchHW=10
    """
    nois_std = estimate_noise_std(ecg_signal)
    Nvar = 1.5 * nois_std
    P = 5000
    PatchHW = 10
    return NLM_1d_gpu_vectorized(ecg_signal, Nvar, P, PatchHW)


def filter_patient_data(patient_data):
    """
    1) Butterworth (CPU)
    2) LOESS (CPU)
    3) NLM vectorizat (GPU)
    """
    # 1) Butterworth
    sig_filt = butter_lowpass_filter(patient_data, fs=SAMPLE_RATE)
    # 2) LOESS
    sig_filt = remove_baseline_loess(sig_filt)
    # 3) NLM (vectorizat, GPU)
    sig_filt = apply_nlm_filter_gpu(sig_filt)
    return sig_filt

def load_ekg_data(batch_dir, batch):
    batch_data_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_data.npy"
    batch_metadata_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_metadata.npy"

    if not (batch_data_path.exists() and batch_metadata_path.exists()):
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        sys.exit(1)

    data = np.load(batch_data_path, allow_pickle=True).item()
    metadata = np.load(batch_metadata_path, allow_pickle=True).item()
    return data, metadata

def leave_second_derivation(patient_data):
    return patient_data[:, 1]

def save_filtered_batch(batch, batch_dir, batch_name):
    filtered_batch_dir = FILTERED_DATA_DIR / batch_dir
    filtered_batch_dir.mkdir(parents=True, exist_ok=True)

    save_stem = f"batch_{batch_dir}_{batch_name}"
    batch_path = filtered_batch_dir / save_stem
    np.save(batch_path, batch)

def process_and_save_filtered_batches():
    FILTERED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    main_folders = sorted([f.name for f in PROCESSED_DATA_DIR.iterdir() if f.is_dir()])
    print("Foldere găsite în PROCESSED:", main_folders)

    for main_folder in main_folders:
        batches = sorted([
            f.name.split('_')[2].split('.')[0]
            for f in (PROCESSED_DATA_DIR / main_folder).iterdir()
            if f.is_file() and "_data.npy" in f.name
        ])

        print(f"\nProcesăm batch-urile din folderul: {main_folder}")
        with tqdm(total=len(batches), desc=f"Processing {main_folder}") as pbar:
            for batch_name in batches:
                batch_data, batch_metadata = load_ekg_data(main_folder, batch_name)
                if batch_data is None or batch_metadata is None:
                    print(f"Batch {batch_name} nu a fost găsit în {main_folder}, îl sărim.")
                    pbar.update(1)
                    continue

                batch = {}
                for record_name, record_data in batch_data.items():
                    sex = batch_metadata[record_name]['sex']
                    age = batch_metadata[record_name]['age']
                    dx = batch_metadata[record_name]['dx']

                    second_derivation = leave_second_derivation(record_data)
                    # Filtrăm pipeline
                    filtered_signal = filter_patient_data(second_derivation)

                    batch[record_name] = {
                        'sex': sex,
                        'age': age,
                        'dx': dx,
                        'data': filtered_signal
                    }

                save_filtered_batch(batch, main_folder, batch_name)
                pbar.update(1)

if __name__ == "__main__":
    process_and_save_filtered_batches()
