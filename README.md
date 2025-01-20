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
   - Obtain a suitable 12-lead EKG dataset (e.g., PhysioNet’s MIT-BIH Arrhythmia Database) and place it in the `data/` directory.


## References
1. [PhysioNet EKG Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
2. [Heart Arrhythmias Overview](https://www.physio-pedia.com/Heart_Arrhythmias)


