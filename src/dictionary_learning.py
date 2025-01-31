import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from constants import PROCESSED_DATA_DIR, CLASSIFIER_DATA_DIR

# load data
def load_raw_signal(batch_dir, batch):
    batch_data_path = PROCESSED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}_data.npy"
    if not batch_data_path.exists():
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)
    
    data = np.load(batch_data_path, allow_pickle=True).item()
    return data

def load_signal_features(batch_dir, batch):
    batch_path = CLASSIFIER_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}.npy"
    if not batch_path.exists():
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)

    batch = np.load(batch_path, allow_pickle=True).item()
    return batch

def load_dictionary_data(batch_dir, batch):
    raw_signal_batch = load_raw_signal(batch_dir, batch)
    signal_features_batch = load_signal_features(batch_dir, batch)

    d = {}
    for record_name in signal_features_batch.keys():
        d[record_name] = {}
        d[record_name]['raw_signal'] = raw_signal_batch[record_name][:, 1]
        d[record_name]['features'] = signal_features_batch[record_name]['features']

    return d

# dictionary learning
def train_ksvd(dictionary_data, dict_size=100, sparsity=10, iterations=10):
    ekg_signals = []
    values = []

    for record in dictionary_data.keys():
        ekg_signals.append(dictionary_data[record]['raw_signal'])
        values.append(dictionary_data[record]['features'])

    ekg_signals = np.array(ekg_signals, dtype=np.float32).T  
    values = np.array(values, dtype=np.float32).T  

    dict_learner = DictionaryLearning(n_components=dict_size, transform_algorithm='omp', transform_n_nonzero_coefs=sparsity)
    dictionary = dict_learner.fit(ekg_signals).components_

    transform_matrix = np.random.randn(dict_size, values.shape[0])

    for _ in range(iterations):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
        sparse_codes = omp.fit(dictionary.T, ekg_signals.T).coef_.T  # (num_samples, dict_size)

        # minimizarea erorii
        transform_matrix = np.linalg.pinv(sparse_codes) @ values.T  # pseudo-inversă

        predicted_values = sparse_codes @ transform_matrix  # (num_samples, 6)
        error = values.T - predicted_values  

        dictionary = dict_learner.fit(ekg_signals - (error @ transform_matrix.T)).components_

    return dictionary, transform_matrix

def predict_features(dictionary, transform_matrix, new_ekg):
    new_ekg = np.array(new_ekg, dtype=np.float32).reshape(1, -1)  
    
    # reprezentarea sparse cu OMP
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
    sparse_code = omp.fit(dictionary.T, new_ekg).coef_.T

    # estimarea caracteristicilor
    predicted_values = sparse_code @ transform_matrix
    return predicted_values.flatten()

def test_model(dictionary, transform_matrix, test_data):
    results = {}
    for record in test_data.keys():
        test_ekg = test_data[record]['raw_signal']
        predicted_values = predict_features(dictionary, transform_matrix, test_ekg)
        results[record] = predicted_values
    return results

def save_dictionary(dictionary, transform_matrix, filename="dictionary.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((dictionary, transform_matrix), f)

def load_dictionary(filename="dictionary.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def test_model(dictionary, transform_matrix, test_data):
    results = {}
    actual_values = []
    predicted_values_list = []
    
    for record in test_data.keys():
        test_ekg = test_data[record]['raw_signal']
        predicted_values = predict_features(dictionary, transform_matrix, test_ekg)
        results[record] = predicted_values
        actual_values.append(test_data[record]['features'])
        predicted_values_list.append(predicted_values)
    
    actual_values = np.array(actual_values)
    predicted_values_list = np.array(predicted_values_list)
    
    mae = mean_absolute_error(actual_values, predicted_values_list)
    mse = mean_squared_error(actual_values, predicted_values_list)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    
    return results

training_batch_data = load_dictionary_data('01', '010')
test_batch_data = load_dictionary_data('01', '011')
print(training_batch_data['JS00001']['features'])

# dictionary, transform_matrix = train_ksvd(training_batch_data)
