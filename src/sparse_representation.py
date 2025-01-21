import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from pathlib import Path

##############################################################################
#                           1. Încărcarea Datelor                            #
##############################################################################
def load_ekg_data(batch_data_path, batch_metadata_path):
    """
    Încarcă fișierele .npy generate anterior: semnalele și metadata.
    """
    # Încarcă semnalele
    data = np.load(batch_data_path)
    # Încarcă metadata
    metadata = np.load(batch_metadata_path, allow_pickle=True).item()
    return data, metadata

##############################################################################
#                   2. Preprocesare Minimă (exemplu simplu)                  #
##############################################################################
def preprocess_signal(signal):
    """
    Exemplu minimal de preprocesare:
    - selectăm doar un singur canal (lead) pentru demo
    - normalizăm semnalul
    """
    # `signal` are forma (5000, 12) dacă este un singur EKG cu 12 derivații
    # În exemplu alegem lead-ul 0 => shape final (5000,)
    single_lead = signal[:, 0]
    
    # Normalizare (scade media, împarte la abaterea standard)
    single_lead = single_lead - np.mean(single_lead)
    std_val = np.std(single_lead)
    if std_val > 1e-12:
        single_lead = single_lead / std_val
    
    return single_lead

##############################################################################
#            3. Segmentare în Ferestre & Construirea Matricii de Date         #
##############################################################################
def segment_signal_into_patches(signal, patch_size=100, step=50):
    """
    Segmentăm un semnal 1D în ferestre (patch-uri) suprapuse.
    - patch_size: dimensiunea unei ferestre
    - step: cât ne deplasăm între ferestre
    Returnează un array de forma (num_patches, patch_size).
    """
    patches = []
    idx = 0
    while (idx + patch_size) <= len(signal):
        window = signal[idx:idx+patch_size]
        patches.append(window)
        idx += step
    return np.array(patches)

##############################################################################
#                    4. Implementare Simplificată K-SVD                      #
##############################################################################
def ksvd(data, dict_size, sparsity, max_iter=10):
    """
    Implementare educațională (și incomplet optimizată) a K-SVD.
    
    Parametri:
    ----------
    data : array de forma (dimensionalitate, num_exemple)
           Fiecare coloană e un exemplu de antrenament.
    dict_size : int, numărul de atomi din dicționar (coloane în D).
    sparsity : int, numărul maxim de coeficienți nenuli în reprezentarea OMP.
    max_iter : int, numărul de iterații K-SVD.
    
    Returnează:
    -----------
    D : array (dimensionalitate, dict_size)
        Dicționarul antrenat.
    X : array (dict_size, num_exemple)
        Coeficienții (sparse codes) finali pentru data.
    """
    (dim, num_samples) = data.shape
    
    # Inițializare dicționar D (random, normalizat pe coloane)
    D = np.random.randn(dim, dict_size)
    for j in range(dict_size):
        D[:, j] /= np.linalg.norm(D[:, j]) + 1e-12
    
    # Matricea de coeficienți X inițial
    X = np.zeros((dict_size, num_samples))
    
    for it in range(max_iter):
        # ============================================================
        # 1) Sparse Coding: folosim OMP pe fiecare exemplu
        # ============================================================
        for i in range(num_samples):
            y = data[:, i]  # exemplu i
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
            omp.fit(D, y)
            coef = omp.coef_
            X[:, i] = coef
        
        # ============================================================
        # 2) Actualizare Dicționar (K-SVD)
        # ============================================================
        for k in range(dict_size):
            # Indicele probelor care folosesc atomul k
            idxs = np.where(np.abs(X[k, :]) > 1e-12)[0]
            if len(idxs) == 0:
                # Dacă atomul nu e folosit, reinițializează-l random
                D[:, k] = np.random.randn(dim)
                D[:, k] /= (np.linalg.norm(D[:, k]) + 1e-12)
                continue
            
            # Erorile reziduale pentru acele probe
            E = data[:, idxs] - np.dot(D, X[:, idxs])
            
            # Contribuția altor atomi, exceptând k
            # Dar K-SVD clasic o face altfel; aici simplificăm
            # totuși calculăm doar reziduul adăugând înapoi atomul k.
            Xk = X[k, idxs]
            D[:, k] = 0
            E += np.outer(D[:, k], Xk)  # reconstituim parțial (aici e zero oricum)
            
            # Pe submatrice
            U, S, Vt = np.linalg.svd(E, full_matrices=False)
            
            # Actualizează atomul k
            D[:, k] = U[:, 0]
            # Actualizează coeficienții corespunzători
            X[k, idxs] = S[0] * Vt[0, :]
        
    return D, X

##############################################################################
#                    5. Codare Sparse cu OMP folosind D final                #
##############################################################################
def omp_encode(data, D, sparsity):
    """
    Realizează codare sparse pentru 'data' (coloane) folosind
    dicționarul D și un număr maxim de coeficienți nenuli = sparsity.
    Returnează matricea coeficienților.
    """
    (dim, num_samples) = data.shape
    dict_size = D.shape[1]
    X = np.zeros((dict_size, num_samples))
    
    for i in range(num_samples):
        y = data[:, i]
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
        omp.fit(D, y)
        X[:, i] = omp.coef_
    return X

##############################################################################
#                     6. Exemplu de clasificare cu SVM                       #
##############################################################################
def classify_with_svm(features, labels):
    """
    Clasifică cu un SVM simplu (RBF kernel).
    features: array (num_samples, num_features)
    labels: array (num_samples,) cu etichete
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                        test_size=0.3, random_state=42)
    svm_model = SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    print("Raport clasificare:\n", classification_report(y_test, y_pred))

##############################################################################
#                          7. MAIN de Demonstrație                            #
##############################################################################
if __name__ == "__main__":
    # Folosim Path pentru a obține calea corectă la directorul părinte
    current_dir = Path(__file__).parent.parent
    processed_dir = current_dir / 'data' / 'processed'
    print(f"Directorul cu date procesate: {processed_dir}")
    
    # Construim căile către fișiere folosind Path
    batch_data_path = processed_dir / "batch_01_010_data.npy"
    batch_metadata_path = processed_dir / "batch_01_010_metadata.npy"
    
    if not (batch_data_path.exists() and batch_metadata_path.exists()):
        print("Fișierele batch_01_010_data.npy și batch_01_010_metadata.npy nu există în path-ul specificat.")
        exit(1)
    
    # 1) Încărcare date
    all_data, all_metadata = load_ekg_data(batch_data_path, batch_metadata_path)
    # 'all_data' shape: (N, 5000, 12) - N = numărul de înregistrări
    # 'all_metadata' este un dict
    
    # 2) Construim un set de patch-uri de la primele, de ex. 10 înregistrări
    max_records = 10
    patch_size = 100
    step = 50
    
    patch_list = []
    labels_list = []
    
    np.random.seed(123)  # doar pt reproducere label random
    
    for idx in range(min(max_records, len(all_data))):
        signal_12_lead = all_data[idx]  # shape (5000,12)
        
        # Preprocesează: extrage un singur lead, normalizează
        proc_sig = preprocess_signal(signal_12_lead)  # shape (5000,)

        # Segmentare în patch-uri
        patches = segment_signal_into_patches(proc_sig, patch_size, step)  # (num_patches, 100)
        
        # Adaugă la listă
        patch_list.append(patches)
        
        # Exemplu: generăm random etichete 0/1 
        # (În realitate, extragi din 'all_metadata' -> 'dx' sau alt câmp)
        random_label = np.random.randint(0, 2, size=(len(patches),))
        labels_list.append(random_label)
    
    # Concatenăm toate patch-urile într-o matrice mare
    # patch_list[i] are shape (num_patches_i, patch_size)
    # Le concatenăm pe axa 0 => shape total: (X, patch_size)
    all_patches = np.concatenate(patch_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    # Avem all_patches shape = (num_total_patches, patch_size)
    # Transpunem să avem coloane = exemple => shape devine (patch_size, num_total_patches)
    data_for_dict = all_patches.T  # (100, num_total_patches)
    
    print(f"Forma finală a datelor pentru K-SVD: {data_for_dict.shape}")
    print(f"Număr total de patch-uri: {all_patches.shape[0]}")
    
    # 3) Antrenăm un dicționar cu K-SVD
    dict_size = 30    # ex. 30 atomi în dicționar
    sparsity = 5      # max 5 coeficienți nenuli
    max_iter_ksvd = 5 # nr. de iterații K-SVD
    
    print("Antrenare dicționar cu K-SVD...")
    D, X_train_ksvd = ksvd(data_for_dict, dict_size, sparsity, max_iter=max_iter_ksvd)
    
    # 4) Reprezentăm sparse patch-urile (features) folosind D
    #    (Practic X_train_ksvd e deja codarea patch-urilor de antrenament,
    #     dar să fim consecvenți și să apelăm o funcție separată, mai ales
    #     dacă ai avea set de test diferit.)
    print("Codare OMP a patch-urilor (features) cu dicționarul D...")
    X_sparse = omp_encode(data_for_dict, D, sparsity)  # shape: (dict_size, num_total_patches)

    # 5) Pregătim *features* pentru SVM -> transpunem la (num_patches, dict_size)
    features_for_svm = X_sparse.T
    
    # 6) Clasificare
    print("Clasificare cu SVM...\n")
    classify_with_svm(features_for_svm, all_labels)
    
    # 7) Vizualizări: comparăm un patch original cu reconstrucția din dicționar
    #    pentru a vedea cum arată semnalul înainte/după
    sample_idx = 0  # un patch oarecare
    original_patch = data_for_dict[:, sample_idx]  # shape (100,)
    
    # Reconstrucția = D * X_sparse_col
    sparse_code = X_sparse[:, sample_idx]
    reconstructed_patch = D.dot(sparse_code)
    
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(original_patch, label='Patch Original')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(reconstructed_patch, label='Patch Reconstruit (K-SVD + OMP)', color='orange')
    plt.legend()
    plt.suptitle("Comparație patch original vs. reconstruit")
    plt.show()
