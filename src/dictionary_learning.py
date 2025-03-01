import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import sparse_encode
from sklearn.metrics import mean_absolute_error, mean_squared_error
from constants import FILTERED_DATA_DIR, PROCESSED_DATA_DIR, CLASSIFIER_DATA_DIR

# load data
def load_raw_signal(batch_dir, batch):
    batch_path = FILTERED_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}.npy"

    if not (batch_path.exists()):
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)

    return np.load(batch_path, allow_pickle=True).item()


def load_signal_features(batch_dir, batch):
    batch_path = CLASSIFIER_DATA_DIR / f"{batch_dir}/batch_{batch_dir}_{batch}.npy"
    if not batch_path.exists():
        print(f"Batch-ul {batch} nu exista in folderul {batch_dir}!")
        exit(1)
    return np.load(batch_path, allow_pickle=True).item()


def load_dictionary_data(batch_dir, batch):
    raw_signals = load_raw_signal(batch_dir, batch)
    features = load_signal_features(batch_dir, batch)

    return {
        name: {
            'raw_signal': raw_signals[name]['data'],  
            'features': features[name]['features'][:-2]
        }
        for name in features.keys()
    }


def load_and_concat_data(batches):
    all_data = {}
    for (bdir, bnum) in batches:
        data = load_dictionary_data(bdir, bnum)
        all_data.update(data)
    return all_data

# dictionary learning
def train_ksvd(dictionary_data, dict_size=512, sparsity=15, iterations=25):
    signals = [rec['raw_signal'] for rec in dictionary_data.values()]
    signals = np.array([(s - np.mean(s))/np.std(s) for s in signals], dtype=np.float32)  
    
    kmeans = KMeans(n_clusters=min(dict_size, len(signals)), n_init=10)
    kmeans.fit(signals)
    dictionary = kmeans.cluster_centers_  
    dictionary = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)  

    codes = sparse_encode(signals, dictionary, algorithm='omp', n_nonzero_coefs=sparsity)
    target_features = np.array([r['features'] for r in dictionary_data.values()], dtype=np.float32)
    transform_matrix = np.linalg.lstsq(codes, target_features, rcond=None)[0]  

    for _ in range(iterations):
        codes = sparse_encode(signals, dictionary, algorithm='omp', n_nonzero_coefs=sparsity)
        
        for i in range(dictionary.shape[0]):
            idx = np.where(codes[:, i] != 0)[0]
            if len(idx) == 0:
                continue

            E = signals[idx] - codes[idx] @ dictionary + np.outer(codes[idx, i], dictionary[i])
            
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            dictionary[i] = Vt[0]
            codes[idx, i] = S[0] * U[:, 0]

        transform_matrix = np.linalg.lstsq(codes, target_features, rcond=None)[0]

    return dictionary, transform_matrix

# predict
def predict_features(dictionary, transform_matrix, new_signal, sparsity=15):
    s = (new_signal - np.mean(new_signal)) / np.std(new_signal)
    s = s.reshape(1, -1)  
    code = sparse_encode(s, dictionary, algorithm='omp', n_nonzero_coefs=sparsity)  
    return code @ transform_matrix  

def extract_features_from_new_signal(dictionary, transform_matrix, raw_signal, sparsity=15):
    predicted = predict_features(dictionary, transform_matrix, raw_signal, sparsity=sparsity)
    return predicted.flatten()  

# test
def test_model(dictionary, transform_matrix, test_data, sparsity=15):
    actual, predicted = [], []
    for name, data in test_data.items():
        try:
            pred = predict_features(dictionary, transform_matrix, data['raw_signal'], sparsity=sparsity)
            actual.append(data['features'])
            predicted.append(pred.flatten())
        except Exception as e:
            print(f"Eroare la {name}: {str(e)}")
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    return dict(zip(test_data.keys(), predicted))

# save / load
def save_model(model, path="../models/dictionary"):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    joblib.dump({'dictionary': model[0], 'transform_matrix': model[1]}, path)

def load_model(path="../models/dictionary"):
    model = joblib.load(path)
    return model['dictionary'], model['transform_matrix']

if __name__ == "__main__":
    train_batches = [
        ('01', '010'),
        ('01', '011'),
        ('01', '012'),
        ('01', '013'),
        ('01', '014'),
        ('01', '015'),
        ('01', '016'),
        ('01', '017'),
        ('01', '018'),
    ]

    test_batches = [
        ('01', '019'),
    ]

    train_data = load_and_concat_data(train_batches)

    DICT_SIZE = 512   
    SPARSITY = 15    
    ITERATIONS = 25 

    dictionary, tm = train_ksvd(
        dictionary_data=train_data,
        dict_size=DICT_SIZE,
        sparsity=SPARSITY,
        iterations=ITERATIONS
    )

    save_model((dictionary, tm))

    loaded_dict, loaded_tm = load_model()

    test_data = load_and_concat_data(test_batches)
    results = test_model(loaded_dict, loaded_tm, test_data, sparsity=SPARSITY)

    for k, v in list(results.items())[:3]:
        print(f"\n{k}:")
        print(f"Real: {test_data[k]['features']}")
        print(f"Pred: {v}")
