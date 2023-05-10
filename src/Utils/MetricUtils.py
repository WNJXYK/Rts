from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import os, json
import numpy as np
import collections.abc
import pandas as pd

__all__ = ["evaluate_results", "Results"]

def evaluate_results(y_true, y_hat_proba, y_hat_labels):
    if y_hat_proba.shape[1] == 2: y_hat_proba = y_hat_proba[:, 1]
    ma_f1 = f1_score(y_true, y_hat_labels, average='macro')
    gmean = geometric_mean_score(y_true, y_hat_labels, average='macro')
    bacc  = balanced_accuracy_score(y_true, y_hat_labels)
    
    df = pd.DataFrame(columns=['Macro F1', 'GMean', "Balanced Accuracy"])
    df.loc['Metrics'] = [ma_f1, gmean, bacc]
    pd.set_option('display.max_columns', None)
    print(df)
    
    return ma_f1, gmean, bacc

def dictUpdate(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dictUpdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class Results:
    def __init__(self, args):
        self.args = args
        self.file = f"./Results/{self.args.method}.json"
        self.metrics = ["MaF1", "GMEAN", "BACC"]
        self.data = {}
        try:
            with open(self.file, 'r') as f: self.data = json.load(f)
        except: pass
        
    def update(self, ma_f1, gmean, bacc):
        noise = f"{self.args.noise_type}-{int(self.args.noise_rate * 100.0)}"
        config = {
            self.args.dataset: {
                noise: {
                    str(self.args.seed): {
                        "MaF1": ma_f1,
                        "GMEAN": gmean,
                        "BACC": bacc,
                    }
                }
            }
        }
        self.data = dictUpdate(self.data, config)
        
        cur = self.data[self.args.dataset][noise]
        arr = [[] for i in range(len(self.metrics))]
        for k in cur:
            if k == "Mean" or k == "Stdev": continue
            v = cur[k]
            for i in range(len(self.metrics)):
                arr[i].append(v[self.metrics[i]])
                
        for i in range(len(self.metrics)):
            config = {
                self.args.dataset: { noise: {
                    "Mean": { self.metrics[i]: np.mean(arr[i]) },
                    "Stdev": { self.metrics[i]: np.std(arr[i]) }
                } }
            }
            self.data = dictUpdate(self.data, config)
        
        self.save_results()

    def save_results(self):
        with open(self.file, 'w') as f: json.dump(self.data, f)