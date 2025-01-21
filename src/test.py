import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Define constants
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SAMPLING_RATE = 500
PROCESSED_DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
BATCH_NAME = 'batch_01_010'

def plot_all_leads_compact(data, leads, sampling_rate):
    """Plot all leads in a compact format on a single figure"""
    time = np.arange(len(data)) / sampling_rate
    plt.figure(figsize=(15, 8))
    
    spacing = 2.5
    for i, lead in enumerate(leads):
        plt.plot(time, data[:, i] + (i * spacing), label=lead)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (mV)')
    plt.title('All ECG Leads')
    plt.yticks([i * spacing for i in range(len(leads))], leads)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

if __name__ == "__main__":
  
    data = np.load(f'{PROCESSED_DATA_DIR}/{BATCH_NAME}_data.npy')
    metadata = np.load(f'{PROCESSED_DATA_DIR}/{BATCH_NAME}_metadata.npy', allow_pickle=True).item()
    
    first_record_data = data[0]
    first_record_id = list(metadata.keys())[0]
    record_meta = metadata[first_record_id]
    
    print(f"Record: {first_record_id}")
    print(f"Age: {record_meta['age']}, Sex: {record_meta['sex']}")
    print(f"Diagnosis: {record_meta['dx']}")
    
    plot_all_leads_compact(first_record_data, LEADS, SAMPLING_RATE)
    plt.show()
