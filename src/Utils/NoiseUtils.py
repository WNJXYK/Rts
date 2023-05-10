import numpy as np
import os

__all__ = ['generate_noise']

def generate_noise_by_trans_matrix(labels, trans_matrix, rng=None):
    res = []
    m = trans_matrix.shape[0]
    for label in labels:
        noisy_label = rng.choice(m, 1, p=trans_matrix[label, :])
        res.append(noisy_label)
    return np.array(res).ravel()

def generate_symmetric_noise(labels, classes, rng=None):
    m = len(classes)
    return rng.choice(m, labels.shape[0]).ravel()

def generate_asymmetric_noise(labels, classes, rng=None):
    m = len(classes)
    trans_matrix = np.eye(m, k=1) + np.eye(m, k=1-m)
    return generate_noise_by_trans_matrix(labels, trans_matrix, rng)

def generate_noise(samples, labels, classes, dataset, ratio=0.4, noise_type="sym", seed=998244353):
    # Save clean labels
    clean_labels = labels.copy()
    
    # Generate indices per classes
    indices = [[] for i in range(len(classes))]
    print(samples.shape, labels.shape)
    for i in range(len(labels)):
        indices[labels[i]].append(i)
    
    # Return noise labels
    noise_file = f"./Noise/{dataset}-{noise_type}-{int(ratio * 100.0)}.npy"
    if os.path.exists(noise_file):
        print("Load %s" % noise_file)
        labels = np.load(noise_file)
        return samples, labels, clean_labels
    
    # Generate label noise
    rng = np.random.default_rng(seed)
    for c in classes:
        siz = len(indices[c])
        noise_siz = int(siz * ratio)
        
        rng.shuffle(indices[c])
        noise_idx = indices[c][: noise_siz]
        
        if noise_type == "sym":
            labels[noise_idx] = generate_symmetric_noise(labels[noise_idx], classes, rng)
        if noise_type == "asym":
            labels[noise_idx] = generate_asymmetric_noise(labels[noise_idx], classes, rng)

    print("Generate %s" % noise_file)
    np.save(noise_file, labels)
    return samples, labels, clean_labels
        