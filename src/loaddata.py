#step 1: import the necessary libraries

import os
import numpy as np
import wfdb
from pathlib import Path
from tqdm import tqdm

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SAMPLE_RATE = 500  
NUM_SAMPLES = 5000  
NUM_LEADS = 12   
PROCESSED_DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'

#step 2: load data from the dataset



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
    folder_path = data_dir / main_folder / subfolder
    records = []
    records_file = folder_path / 'RECORDS'
    if records_file.exists():
        with open(records_file, 'r') as f:
            records = [line.strip() for line in f if line.strip()]
    return records

def read_header_metadata(header_file):
    """Read only essential metadata from .hea file"""
    metadata = {
        'name': None,
        'offsets': [],    # Separate list for offsets
        'checksums': [],  # Separate list for checksums
        'age': None,
        'sex': None,
        'dx': None
    }
    
    try:
        with open(header_file, 'r') as f:
            lines = f.readlines()
            metadata['name'] = lines[0].split()[0]
            
            # Next 12 lines: get offset and checksum correctly
            # parts[4] is offset, parts[5] is checksum
            for line in lines[1:13]:
                parts = line.strip().split()
                metadata['offsets'].append(int(parts[5]))     
                metadata['checksums'].append(int(parts[6]))   
                
            # Get patient info
            for line in lines[13:]:
                if line.startswith('#'):
                    key, value = line[1:].strip().split(':', 1)
                    key = key.lower()
                    if key in ['age', 'sex', 'dx']:
                        metadata[key] = value.strip()
                        
    except Exception as e:
        print(f"Error reading header file {header_file}: {e}")
    
    return metadata

def load_record(record_path, data_dir, verify=True):
    """Load a single record with minimal metadata"""
    base_path = str(record_path.parent / record_path.stem)
    
    try:
        # Read the record using wfdb
        record = wfdb.rdrecord(base_path, 
                             pn_dir=None, 
                             return_res=16)
        
        # Read minimal header metadata
        header_file = Path(base_path + '.hea')
        metadata = read_header_metadata(header_file)
        
        return {
            'name': metadata['name'],
            'data': record.p_signal,
            'metadata': metadata
        }
    except Exception as e:
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
            data = load_record(record_path, data_dir, verify)
            
            if data is not None:
                dataset['records'][data['name']] = data
                pbar.update(1)
                pbar.set_description(f"Loaded {record} from {folder}")
            else:
                dataset['metadata']['failed_records'].append(str(record_path))
    
    dataset['metadata']['total_records'] = len(dataset['records'])
    return dataset

def load_batch(data_dir, main_folder, subfolder, verify=True):
    """Load all records from a specific batch folder (e.g., 01/010)"""
    data_dir = Path(data_dir)
    batch_path = data_dir / main_folder / subfolder
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
            data = load_record(record_path, data_dir, verify)
            
            if data is not None:
                dataset['records'][record_name] = data
                pbar.update(1)
                pbar.set_description(f"Loaded {record_name}")
            else:
                dataset['metadata']['failed_records'].append(str(record_path))
    
    dataset['metadata']['total_records'] = len(dataset['records'])
    return dataset


#step 3: load the data


if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent
    print("Current directory:", current_dir)
    data_dir = current_dir / 'data' /'raw'/ 'WFDBRecords'
    
    # Create processed directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load batch (01/010)
    batch_dataset = load_batch(data_dir, main_folder="01", subfolder="010")

    print("\nBatch Summary:")
    print("-" * 50)
    print(f"Successfully loaded: {batch_dataset['metadata']['total_records']} records")
    print(f"Failed records: {len(batch_dataset['metadata']['failed_records'])}")


    if batch_dataset['records']:
        first_record = next(iter(batch_dataset['records'].values()))
        print("\nRecord Details:")
        print(f"Signal shape: {first_record['data'].shape}")
        print(f"Sampling rate: {SAMPLE_RATE} Hz")  # Using constant
        print("\nMetadata:")
        print(f"Age: {first_record['metadata']['age']}")
        print(f"Sex: {first_record['metadata']['sex']}")
        print(f"Diagnosis: {first_record['metadata']['dx']}")

        # Save both data and metadata
        batch_data = np.stack([record['data'] for record in batch_dataset['records'].values()])
        batch_metadata = {rid: {
            'name': record['metadata']['name'],
            'offsets': record['metadata']['offsets'],
            'checksums': record['metadata']['checksums'],
            'age': record['metadata']['age'],
            'sex': record['metadata']['sex'],
            'dx': record['metadata']['dx']
        } for rid, record in batch_dataset['records'].items()}
        
        # Modified save commands to use PROCESSED_DATA_DIR
        save_path = PROCESSED_DATA_DIR / f'batch_{batch_dataset["metadata"]["main_folder"]}_{batch_dataset["metadata"]["subfolder"]}'
        np.save(f'{save_path}_data.npy', batch_data)
        np.save(f'{save_path}_metadata.npy', batch_metadata)
        print(f"\nSaved processed data to: {save_path}_data.npy")
        print(f"Saved metadata to: {save_path}_metadata.npy")

    # Test print for JS00004
    if 'JS00004' in batch_dataset['records']:
        record = batch_dataset['records']['JS00004']
        print("\nJS00004 Details:")
        print("-" * 50)
        print("Offsets and Checksums by lead:")
        for i, (lead, offset, checksum) in enumerate(zip(LEADS, 
                                                        record['metadata']['offsets'],
                                                        record['metadata']['checksums'])):
            print(f"{lead}: offset={offset}, checksum={checksum}")
        print("\nPatient Info:")
        print(f"Age: {record['metadata']['age']}")
        print(f"Sex: {record['metadata']['sex']}")
        print(f"Diagnosis: {record['metadata']['dx']}")
        
    # Save with updated metadata structure
    batch_metadata = {rid: {
        'name': record['metadata']['name'],
        'offsets': record['metadata']['offsets'],
        'checksums': record['metadata']['checksums'],
        'age': record['metadata']['age'],
        'sex': record['metadata']['sex'],
        'dx': record['metadata']['dx']
    } for rid, record in batch_dataset['records'].items()}


