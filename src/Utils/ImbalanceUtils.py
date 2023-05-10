import numpy as np

'''
Estimate the imbalance ratio of dataset
'''
def estimate_IR(train_y):
    labels = np.unique(train_y)
    train_y = list(train_y)
    mi, mx = len(train_y), 0
    for l in labels:
        cnt = train_y.count(l)
        mi = min(mi, cnt)
        mx = max(mx, cnt)
    return mx / mi