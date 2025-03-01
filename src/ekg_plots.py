import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import statsmodels.api as sm
import scipy.signal as signal
from constants import PROCESSED_DATA_DIR, FILTERED_DATA_DIR, NUM_SAMPLES, SAMPLE_RATE, SNOMED_DICT, LEADS, NUM_LEADS, PLOT_DIR


def load_ekg_data(batch_data_path, batch_metadata_path):
    data = np.load(batch_data_path, allow_pickle=True).item()
    metadata = np.load(batch_metadata_path, allow_pickle=True).item()
    return data, metadata

def load_pacient_data(batch_dir, batch, name):
    batch_data_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_data.npy"
    batch_metadata_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_metadata.npy"

    if not (batch_data_path.exists() and batch_metadata_path.exists()):
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)

    batch_data, batch_metadata = load_ekg_data(batch_data_path, batch_metadata_path)

    if name in batch_data:
        return batch_data[name], batch_metadata[name]

    print(f"Nu exista pacientul {name} in batch-ul {batch}")
    exit(1)

def plot_patient_time_domain(patient_name, patient_data, patient_metadata, file_name, plot = False):
    time = np.arange(NUM_SAMPLES) / SAMPLE_RATE
    plt.figure(figsize=(18, 10))  

    spacing = 3  
    for i, lead in enumerate(LEADS):
        plt.plot(time, patient_data[:, i] + (i * spacing), label=lead, linewidth=1)

    age = patient_metadata.get('age', 'N/A')
    sex = patient_metadata.get('sex', 'N/A')
    dx_codes = patient_metadata.get('dx', 'N/A').split(',')

    dx_labels = [SNOMED_DICT.get(dx.strip(), ("Unknown", "Unknown"))[1] for dx in dx_codes]

    metadata_text = f"Patient: {patient_name}\nSex: {sex}\nAge: {age}\nDiagnosis: {', '.join(dx_labels)}"
    plt.gca().text(0.01, 0.99, metadata_text, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Amplitude (mV)', fontsize=14)
    plt.title(f'12-Lead ECG for {patient_name}', fontsize=16, pad=20)
    plt.yticks([i * spacing for i in range(len(LEADS))], LEADS, fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(f'{PLOT_DIR}/{file_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')

    if plot:
        plt.show()
    plt.clf()


def plot_patient_frequency_domain(patient_name, patient_data, patient_metadata, file_name, plot = False):
    freqs = np.fft.rfftfreq(NUM_SAMPLES, d=1/SAMPLE_RATE)
    plt.figure(figsize=(18, 12)) 

    for i, lead in enumerate(LEADS):
        current_lead = patient_data[:, i]
        current_lead -= np.mean(current_lead)
        fft_values = np.abs(np.fft.rfft(current_lead))
        
        plt.subplot(4, 3, i + 1)  
        plt.plot(freqs, fft_values, label=lead, linewidth=1.5)
        plt.title(f"{lead}", fontsize=12)
        plt.xlabel("Frequency (Hz)", fontsize=10)
        plt.ylabel("Amplitude", fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f"Frequency Domain Analysis for Patient {patient_name}", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.savefig(f'{PLOT_DIR}/{file_name}.pdf', format='pdf', dpi=300, bbox_inches='tight')

    if plot:
        plt.show()
    plt.clf()

def filter_patient_data(patient_data, high_cutoff = 50):
    filtered_data = np.zeros((NUM_SAMPLES, NUM_LEADS))  

    for lead_idx in range(NUM_LEADS):
        lead_signal = patient_data[:, lead_idx]  

        filtered_signal = nk.ecg_clean(lead_signal, sampling_rate = SAMPLE_RATE)
        filtered_data[:, lead_idx] = filtered_signal

    return filtered_data

def remove_offset_patient_data(patient_data, offsets):
    offsets = np.array(offsets)

    adjusted_data = patient_data - offsets

    return adjusted_data

def leave_second_derivation(patient_data):
    return patient_data[:, 1]

if __name__ == "__main__":
    batch_dir = '01'
    batch = '010'
    patient_name = 'JS00012'
    patient_data, patient_metadata = load_pacient_data(batch_dir, batch, patient_name)
    
    # plot_patient_frequency_domain(patient_name, patient_data, patient_metadata, f'raw_{patient_name}_time_domain')
    plot_patient_time_domain(patient_name, patient_data, patient_metadata, f'raw_{patient_name}_frequency_domain')

    # patient_data = remove_offset_patient_data(patient_data, patient_metadata['offsets'])
    # patient_data = filter_patient_data(patient_data)

    # plot_patient_frequency_domain(patient_name, patient_data, patient_metadata, f'filtered_{patient_name}_time_domain')
    # plot_patient_time_domain(patient_name, patient_data, patient_metadata, f'filtered_{patient_name}_frequency_domain', plot=True)
    signals, info = nk.ecg_process(patient_data[:, 1], sampling_rate=500)
    nk.ecg_plot(signals, info)
    plt.gcf().set_size_inches(16, 9)

    plt.savefig(f'{PLOT_DIR}/{patient_name}_ekg_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')



