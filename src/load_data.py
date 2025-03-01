import numpy as np
import wfdb
from pathlib import Path
from tqdm import tqdm
from constants import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEBUG

def debug_print(*args, **kwargs):
    """Print only if DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)

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
        'offsets': [],    
        'checksums': [], 
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
        debug_print(f"Error reading header file {header_file}: {e}")
    
    return metadata

def load_record(record_path, data_dir, verify=True):
    """Load a single record with minimal metadata"""
    base_path = str(record_path.parent / record_path.stem)
    
    try:
        record = wfdb.rdrecord(base_path, 
                             pn_dir=None, 
                             return_res=16)
        
        header_file = Path(base_path + '.hea')
        metadata = read_header_metadata(header_file)
        
        return {
            'name': metadata['name'],
            'data': record.p_signal,
            'metadata': metadata
        }
    except Exception as e:
        debug_print(f"Error reading {record_path}: {e}")
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

def load_records(data_dir, record_filter=None, verify=True):
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
    
    debug_print(f"Loading batch from {main_folder}/{subfolder}")
    debug_print(f"Found {len(records)} records")
    
    for record_name in records:
        record_path = batch_path / record_name
        data = load_record(record_path, data_dir, verify)
        
        if data is not None:
            dataset['records'][record_name] = data
        else:
            dataset['metadata']['failed_records'].append(str(record_path))
    
    dataset['metadata']['total_records'] = len(dataset['records'])
    return dataset

# functie de save la fiecare batch, unde un batch e de ex 01/010
def save_batch(dataset, output_dir):
    """
    Save the given batch dataset into two .npy files (data & metadata).
    The naming will be based on dataset['metadata']['main_folder'] and
    dataset['metadata']['subfolder'].
    """
    if not dataset or not dataset['records']:
        debug_print("No records to save in this dataset.")
        return

    # batch_data = np.stack([record['data'] for record in dataset['records'].values()])
    batch_data = {rec['metadata']['name']: rec['data'] for rec in dataset['records'].values()}
    batch_metadata = {
        rid: {
            'name': rec['metadata']['name'],
            'offsets': rec['metadata']['offsets'],
            'checksums': rec['metadata']['checksums'],
            'age': rec['metadata']['age'],
            'sex': rec['metadata']['sex'],
            'dx': rec['metadata']['dx']
        }
        for rid, rec in dataset['records'].items()
    }

    main_folder = dataset['metadata']['main_folder']
    subfolder = dataset['metadata']['subfolder']
    save_stem = f"batch_{main_folder}_{subfolder}"

    data_path = output_dir / f"{save_stem}_data.npy"
    meta_path = output_dir / f"{save_stem}_metadata.npy"

    np.save(data_path, batch_data)
    np.save(meta_path, batch_metadata)

    debug_print(f"\nSaved processed data to: {data_path}")
    debug_print(f"Saved metadata to: {meta_path}")

# incarca toate batch-urile dintr-un folder, de ex, toate subfolderele din 01
def load_batches_from_folder(main_folder):
    subfolders = [f.name for f in (RAW_DATA_DIR / main_folder).iterdir() if f.is_dir()]
    processed_folder = PROCESSED_DATA_DIR / main_folder
    processed_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading all batches from main folder: {main_folder}")
    
    with tqdm(total=len(subfolders), desc=f"Processing {main_folder}") as pbar:
        for subfolder in subfolders:
            batch_dataset = load_batch(RAW_DATA_DIR, main_folder, subfolder)
            save_batch(batch_dataset, processed_folder)
            pbar.update(1)  
            pbar.set_description(f"Loaded batch {main_folder}/{subfolder}")

if __name__ == "__main__":
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    subfolders = sorted([f.name for f in (RAW_DATA_DIR).iterdir() if f.is_dir()])
    for subfolder in subfolders:
        load_batches_from_folder(subfolder)
