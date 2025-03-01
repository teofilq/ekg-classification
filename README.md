# ekg-classification-pipeline
Pipeline for processing and classifying EKG signals

## Overview
This project focuses on detecting and classifying cardiac arrhythmias in 12-lead EKG signals using sparse representation techniques and dictionary learning. It combines signal preprocessing, sparse coding, and machine learning classification to create an efficient and robust system for analyzing EKG data.

### Key Features:
- **Signal Preprocessing:** Filtering and segmentation of EKG signals to extract relevant cardiac beats.
- **Sparse Representation:** Utilizing algorithms like Orthogonal Matching Pursuit (OMP) and K-SVD for feature extraction.
- **Machine Learning Classification:** Employing models such as SVM or logistic regression to classify arrhythmic beats.
- **Extensibility:** The pipeline can be adapted for other signal processing tasks such as noise reduction, inpainting, or multi-class arrhythmia classification.

---

## Setup and Installation

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-username/ekg-classification-pipeline.git
   cd ekg-classification-pipeline
   ```

2. **Create and Activate a Virtual Environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Required Packages:**
   ```sh
   pip3 install -r requirements.txt
   ```

4. **Download the Dataset:**
   - Obtain a suitable 12-lead EKG dataset (PhysioNet’s MIT-BIH Arrhythmia Database) and place it in the `data/` directory.
  

## References

1. [PhysioNet EKG Arrhythmia Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
2. [Heart Arrhythmias Overview](https://www.physio-pedia.com/Heart_Arrhythmias)

## Data Processing Structure

### Processed Data Organization
The processed data is stored in batches in the `data/processed/` directory:

```
data/processed/
└── batch_01_010_data.npy    # Signal data array
└── batch_01_010_metadata.npy # Metadata dictionary
```

### Data Structure
Each batch contains two files:
1. **Data Array** (.npy):
   - Shape: (N, 5000, 12) where:
     - N = number of records in batch
     - 5000 = samples per record
     - 12 = number of leads
   - Type: numpy float array
   
2. **Metadata** (.npy):
   - Dictionary structure:
   ```python
   {
       'JS00001': {                    # Record ID
           'name': 'JS00001',          # Record name
           'offsets': [off1,...,off12],# Offset values for each lead
           'checksums': [ch1,...,ch12],# Checksum values for each lead
           'age': '85',               # Patient age
           'sex': 'Male',             # Patient sex
           'dx': '164889003,...'      # Diagnosis codes
       },
       # ... more records
   }
   ```


# w00ps files
- WFDBRecords/01/019/JS01052.hea
- WFDBRecords/23/236/JS23074.hea

# Useful links
- [Sparse representation-based ECG signal enhancement and QRS detection](https://pubmed.ncbi.nlm.nih.gov/27811395/)
- [Sparse representation of ECG signals for automated recognition of cardiac arrhythmias](https://www.sciencedirect.com/science/article/abs/pii/S0957417418301842)
- [Block sparse multi-lead ECG compression exploiting between-lead collaboration](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-spr.2018.5076)
- [PERSONALIZED ECG MONITORING by sparse representation based feature extraction](https://trepo.tuni.fi/bitstream/handle/10024/119250/JoronenManu.pdf?sequence=2&isAllowed=y)


# Altele
Ce am folosit
- https://www.nature.com/articles/s41598-020-59821-7


- Implementare Nlm
https://github.com/zheng120/ECGDenoisingTool


- Exemplu date fara zgomot
  https://figshare.com/collections/ChapmanECG/4560497/2

