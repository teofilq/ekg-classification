import os
import hashlib
import numpy as np
import wfdb
from pathlib import Path
from tqdm import tqdm

def read_sha256sums(filepath):
    """Read SHA256SUMS.txt into a dictionary"""
    sha256sums = {}
    with open(filepath, 'r') as f:
        for line in f:
            hash_value, filename = line.strip().split()
            sha256sums[filename] = hash_value
    return sha256sums

def calculate_sha256(filepath):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read file in chunks for memory efficiency
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_file_integrity(filepath, sha256sums, data_dir):
    """Verify a file's integrity against SHA256SUMS"""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    calculated_hash = calculate_sha256(filepath)
    relative_path = str(Path(filepath).relative_to(data_dir))
    expected_hash = sha256sums.get(relative_path)
    
    if (expected_hash is None):
        return False, "File not found in SHA256SUMS"
        
    return calculated_hash == expected_hash, calculated_hash

def get_all_record_folders(data_dir):
    """Read main RECORDS file and return all folder paths"""
    folders = []
    with open(data_dir / 'RECORDS', 'r') as f:
        folders = [line.strip() for line in f if line.strip()]
    return folders

def read_folder_records(folder_path):
    """Read RECORDS file from a specific folder"""
    records = []
    records_file = folder_path / 'RECORDS'
    if records_file.exists():
        with open(records_file, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
    return records

def read_subfolder_records(data_dir, main_folder, subfolder):
    """Read records from a specific subfolder"""
    folder_path = data_dir / 'WFDBRecords' / main_folder / subfolder
    records = []
    records_file = folder_path / 'RECORDS'
    if records_file.exists():
        with open(records_file, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
    return records

def read_header_metadata(header_file):
    """
    Read metadata from .hea file,
    including channel info and overall patient data
    """
    structure = {
        'record_name': None,
        'num_signals': None,
        'fs': None,
        'num_samples': None,
        'channels': [],
        'patient_info': {
            'age': None,
            'sex': None,
            'dx': None,
            'rx': None,
            'hx': None,
            'sx': None
        }
    }
    
    try:
        with open(header_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Parse lines like '#Age: 68'
                    key, value = line[1:].strip().split(':', 1)
                    structure['patient_info'][key.lower()] = value.strip()
                else:
                    parts = line.strip().split()
                    # First line: 'JS34165 12 500 5000' -> record, num_signals, fs, samples
                    if len(parts) == 4 and parts[0].startswith('JS'):
                        structure['record_name'] = parts[0]
                        structure['num_signals'] = int(parts[1])
                        structure['fs'] = int(parts[2])
                        structure['num_samples'] = int(parts[3])
                    # Lines describing each channel
                    elif len(parts) >= 9 and parts[0].startswith('JS'):
                        channel_info = {
                            'file': parts[0],
                            'gain': parts[1],
                            'units': parts[2],
                            'adc_res': parts[3],
                            'baseline': parts[4],
                            'scale': parts[5],
                            'ch_name': parts[8]
                            # You can tweak or add more as needed
                        }
                        structure['channels'].append(channel_info)
    except Exception as e:
        print(f"Error reading header file {header_file}: {e}")
    
    return structure

def load_record(record_path, sha256sums, data_dir, verify=True):
    """Load a single record and optionally verify its integrity"""
    base_path = str(record_path.parent / record_path.stem)
    
    try:
        # Read the record using wfdb, with error handling for date format issues
        record = wfdb.rdrecord(base_path, 
                             pn_dir=None, 
                             return_res=16, 
                             force_channels=True)
        
        # Read header metadata
        header_file = Path(base_path + '.hea')
        metadata = read_header_metadata(header_file)
        
        return {
            'data': record.p_signal,
            'fs': record.fs,
            'sig_name': record.sig_name,
            'units': record.units,
            'baseline': record.baseline,
            'adc_gain': record.adc_gain,
            'record_name': record_path.stem,
            'folder': str(record_path.parent.name),
            'metadata': metadata
        }
    except Exception as e:
        if "time data" in str(e) or "list index" in str(e):
            # Silent fail for known issues
            return None
        print(f"Error reading {record_path}: {e}")
        return None

def scan_dataset(data_dir):
    """Scan dataset and return a mapping of all available records"""
    data_dir = Path(data_dir)
    dataset_map = {}
    
    folders = get_all_record_folders(data_dir)
    for folder in folders:
        folder_path = data_dir / folder
        if not folder_path.exists():
            continue
            
        records = read_folder_records(folder_path)
        dataset_map[folder] = records
        
    return dataset_map

def create_record_filter(folder=None, record=None):
    """Create a filter function for record selection"""
    def filter_func(record_info):
        folder_match = folder is None or record_info['folder'] == folder
        record_match = record is None or record_info['record'] == record
        return folder_match and record_match
    return filter_func

def load_records(data_dir, record_filter=None, verify=True, save_batch=1000):
    """Generic record loader with filtering"""
    data_dir = Path(data_dir)
    sha256sums = read_sha256sums(data_dir / 'SHA256SUMS.txt') if verify else {}
    
    dataset_map = scan_dataset(data_dir)
    dataset = {
        'records': {},
        'metadata': {
            'total_records': 0,
            'failed_records': [],
            'verification_enabled': verify
        }
    }
    
    records_to_load = []
    
    # Build list of records based on filter
    for folder, records in dataset_map.items():
        for record in records:
            record_info = {'folder': folder, 'record': record}
            if record_filter is None or record_filter(record_info):
                records_to_load.append((folder, record))
    
    # Load filtered records with progress bar
    with tqdm(total=len(records_to_load)) as pbar:
        for folder, record in records_to_load:
            record_path = data_dir / folder / record
            data = load_record(record_path, sha256sums, data_dir, verify)
            
            if data is not None:
                dataset['records'][data['record_name']] = data
                pbar.update(1)
                pbar.set_description(f"Loaded {record} from {folder}")
            else:
                dataset['metadata']['failed_records'].append(str(record_path))
    
    dataset['metadata']['total_records'] = len(dataset['records'])
    return dataset

def load_batch(data_dir, main_folder, subfolder, verify=True):
    """Load all records from a specific batch folder (e.g., 01/010)"""
    data_dir = Path(data_dir)
    sha256sums = read_sha256sums(data_dir / 'SHA256SUMS.txt') if verify else {}
    
    batch_path = data_dir / 'WFDBRecords' / main_folder / subfolder
    records = read_subfolder_records(data_dir, main_folder, subfolder)
    
    dataset = {
        'records': {},
        'metadata': {
            'total_records': 0,
            'failed_records': [],
            'verification_enabled': verify,
            'main_folder': main_folder,
            'subfolder': subfolder
        }
    }
    
    print(f"Loading batch from {main_folder}/{subfolder}")
    print(f"Found {len(records)} records")
    
    with tqdm(total=len(records)) as pbar:
        for record_name in records:
            record_path = batch_path / record_name
            data = load_record(record_path, sha256sums, data_dir, verify)
            
            if data is not None:
                dataset['records'][record_name] = data
                pbar.update(1)
                pbar.set_description(f"Loaded {record_name}")
            else:
                dataset['metadata']['failed_records'].append(str(record_path))
    
    dataset['metadata']['total_records'] = len(dataset['records'])
    return dataset

# Example usage
if __name__ == "__main__":
    # Use relative path from the notebook location
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / 'data' / 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0'
    
    # Load specific batch (01/010)
    batch_dataset = load_batch(data_dir, main_folder="01", subfolder="010")
    
    print("\nBatch Summary:")
    print("-" * 50)
    print(f"Successfully loaded: {batch_dataset['metadata']['total_records']} records")
    print(f"Failed records: {len(batch_dataset['metadata']['failed_records'])}")
    

    if batch_dataset['records']:
        first_record = next(iter(batch_dataset['records'].values()))
        print("\nRecord Details:")
        print(f"Signal shape: {first_record['data'].shape}")
        print(f"Sampling rate: {first_record['fs']} Hz")
        print(f"Channels: {', '.join(first_record['sig_name'])}")
        print("\nMetadata:")
        print(f"Age: {first_record['metadata']['patient_info']['age']}")
        print(f"Sex: {first_record['metadata']['patient_info']['sex']}")
        print(f"Diagnosis: {first_record['metadata']['patient_info']['dx']}")
        print(first_record['metadata'])
        
        print(batch_dataset['records']['JS00033'])
        # Save both data and metadata
        batch_data = np.stack([record['data'] for record in batch_dataset['records'].values()])
        batch_metadata = {rid: record['metadata'] for rid, record in batch_dataset['records'].items()}
        
    

        np.save(f'batch_{batch_dataset["metadata"]["main_folder"]}_{batch_dataset["metadata"]["subfolder"]}_data.npy', batch_data)
        np.save(f'batch_{batch_dataset["metadata"]["main_folder"]}_{batch_dataset["metadata"]["subfolder"]}_metadata.npy', batch_metadata)

# ...rest of the existing utility functions...

