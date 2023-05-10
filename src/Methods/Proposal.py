import time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
import src.Methods.Models as Models
import src.Utils as Utils
from tqdm import trange
from imblearn.over_sampling import ADASYN
import scipy.linalg as la
from scipy.stats import multivariate_normal
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score

def ReguCovar(P, N):
    dims = P.shape[1]
    
    mean = np.mean(P, axis=0)
    cov  = np.cov(P.T)
    d, V = la.eig(cov)
    d = np.real(d)
    V = np.real(V)
    V[d < 0, :] = 0
    d[d < 0] = 0
    
    dmean = np.mean(d)
    M = 0
    for i in range(dims):
        M = i
        if d[i] <= 5e-3: break
    M = min(M, dims // 2)

    tcov = np.cov(np.vstack([P, N]).T)
    dT = np.dot(np.dot(V.T, tcov), V)
    dT = np.diag(dT)
    
    dMod = np.zeros_like(dT)
    Alpha = d[0] * d[M] * M / (d[0] - d[M])
    Beta  = ((M + 1) * d[M] - d[0]) / (d[0] - d[M])
    for i in range(dims):
        if i < M:
            dMod[i] = d[i]
        else:
            dMod[i] = Alpha / (i + 1 + Beta)
            if dMod[i] > dT[i]: dMod[i] = dT[i]

    return mean, V, dMod, M

def dist(x, S):
    n = S.shape[0]
    return np.linalg.norm(S - x, axis=1)

def OverSamp(Me, V, D, P, N, R, M, Num2Gen, n_jobs=8):
    Rn = M + 1
    Un = len(Me) - Rn
    
    MuR = np.zeros(Rn)
    SigmaR = np.eye(Rn)
    MuU = np.zeros(Un)
    SigmaU = np.eye(Un)
    
    SampGen = np.zeros((int(Num2Gen * R), len(Me)))
    SampSel = np.zeros((int(Num2Gen), len(Me)))
    Prob    = np.zeros((int(Num2Gen * R)))
    
    cnt = 0
    V[D < 0, :] = 0
    D[D < 0] = 0
    DD = np.sqrt(D)
    
    # with Pool(4) as p:
        # ret = p.starmap(generate_OS, [(MuR, SigmaR, MuU, SigmaU, DD, V, Me, P, N, R > 1, i) for i in range(int(R * Num2Gen))])
    for i in range(int(R * Num2Gen)):
        ret = generate_OS(MuR, SigmaR, MuU, SigmaU, DD, V, Me, P, N, R > 1, i)
        SampGen[i, :] = ret[0]
        Prob[i] = ret[1]
            # SampGen[i, :] = ret[i][0]
            # Prob[i] = ret[i][1]
    if len(SampGen) == len(SampSel): return SampGen
    
    for i in range(int(Num2Gen)):
        tmp, ind = np.min(Prob), np.argmin(Prob)
        Prob[ind] = 1e18
        SampSel[i, :] = SampGen[ind, :]
    return SampSel

def generate_OS(MuR, SigmaR, MuU, SigmaU, DD, V, Me, P, N, tp_flag, process_id):
    cnts = 0
    while True:
        cnts += 1
        # print("ID", process_id, "Cnts", cnts)
        aR = np.random.multivariate_normal(MuR, SigmaR, 1)
        tp = 1
        if tp_flag: tp = multivariate_normal.pdf(aR, MuR, SigmaR)
        # print("ID", process_id, "PDF")
        if len(MuU) > 0:
            # print("ID", process_id, "SwitchA")
            # print(MuU, SigmaU)
            aU = np.random.multivariate_normal(MuU, SigmaU, 1)
            # print("ID", process_id, "SwitchA-Middle")
            a = np.hstack([aR, aU]) * DD
            # print("ID", process_id, "SwitchA-Finished")
        else:
            # print("ID", process_id, "SwitchB")
            a = aR * DD
        x = np.dot(a, V.T) + Me
        # print("ID", process_id, "Gen")
        PDist = dist(x, P)
        NDist = dist(x, N)
        tmp, ind = np.min(NDist), np.argmin(NDist)
        if np.min(PDist) < tmp:
            PPDist = dist(N[[ind], :], P)
            if tmp >= np.min(PPDist) and tmp <= np.max(PPDist): 
                return x, tp
    return None

def Selective_Oversampling(X, y, R=1.0, Per=0.8):
    m = len(np.unique(y))
    x_shape = X.shape
    X = X.reshape(x_shape[0], -1)
    y_list = list(y)
    label_count = [y_list.count(i) for i in range(m)]
    GEN_count = [np.max(int(np.max(label_count) * R) - i, 0) for i in label_count]
    INOS_count   = [np.ceil(GEN_count[i] * Per) for i in range(m)]
    ADASYN_count = {}
    for i in range(m): ADASYN_count[i] = max(0, GEN_count[i] - INOS_count[i]) + label_count[i]

    X_arr, y_arr = [], []

    try:
        ada = ADASYN(sampling_strategy=ADASYN_count)
        X_res, y_res = ada.fit_resample(X, y)
        X_arr.append(X_res)
        y_arr.append(y_res)
    except:
        X_arr.append(X)
        y_arr.append(y)
        Per = 1.0
        INOS_count   = [np.ceil(GEN_count[i] * Per) for i in range(m)]
    
    for i in trange(m):
        if INOS_count[i] == 0: continue
        indices = y == i
        P, N = X[indices, :], X[~indices, :]
        Me, V, D, M = ReguCovar(P, N)
        X_gen = OverSamp(Me, V, D, P, N, R, M, INOS_count[i])
        X_arr.append(X_gen)
        y_arr.append(np.ones(X_gen.shape[0]) * i)
    
    X_res = np.vstack(X_arr)
    y_res = np.concatenate(y_arr)
    X_res = X_res.reshape(X_res.shape[0], x_shape[1], x_shape[2])
    
    return X_res, y_res


__all__ = ["main_wrapper", "evaluate_model"]

def evaluate_model(args, model, x_test, Y_test):
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())    
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    prediction, gts = [], []
    with torch.no_grad():
        model.eval()
        for data in loader:
            inputs, target = data
            inputs = inputs.float().to(args.device)
            target = target.long().to(args.device)
            _, logits, _ = model(inputs)
            prediction.append(logits.cpu().numpy())
            gts.append(target.cpu().numpy())
            
    pred = np.concatenate(prediction, axis=0)
    y_true = np.concatenate(gts, axis=0)
    y_hat_proba = softmax(pred, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    return y_true, y_hat_proba, y_hat_labels

def SoftCrossEntropy(inputs, target):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    return torch.sum(torch.mul(log_likelihood, target), dim=1)

class ExpEnsemble:
    def __init__(self, shape, alpha=0.8):
        self.value = torch.zeros(shape)
        self.alpha = alpha
        self.counts = 0
    
    def update(self, v):
        assert v.shape == self.value.shape
        self.value = self.value * (1.0 - self.alpha) + v * self.alpha
        self.counts += 1
    
    def get(self):
        assert self.counts > 0
        return self.value / (1.0 - (1.0 - self.alpha) ** self.counts)

def main_wrapper(args, x_train, x_valid, x_test, Y_train, Y_valid, Y_test, Y_train_clean, Y_valid_clean, smooth=0.8, beta=1.0, ensambling=0.8, nt_train=True, oversample_train=True):
    print("Main wrapper start")
    # Prepare
    n_classes = len(np.unique(Y_train_clean))
    classifier = Models.Classifier(d_in=args.classifier_dim, n_class=n_classes)
    encoder = Models.AutoEncoder(input_size=x_train.shape[2], 
                            num_filters=args.filters, 
                            embedding_dim=args.embedding_size,
                            seq_len=x_train.shape[1], 
                            kernel_size=args.kernel_size, 
                            stride=args.stride,
                            padding=args.padding, 
                            dropout=args.dropout, 
                            normalization=args.normalization)
    model = Models.AEModel(encoder=encoder, classifier=classifier).to(args.device)
    
    print("Model prepraed")
    x_train = np.concatenate((x_train, x_valid))
    Y_train = np.concatenate((Y_train, Y_valid))
    Y_train_clean = np.concatenate((Y_train_clean, Y_valid_clean))
    n_samples = x_train.shape[0]
    x_train = torch.from_numpy(x_train).float()
    Y_train = torch.from_numpy(Y_train).long()
    Y_train_clean = torch.from_numpy(Y_train_clean).long()
    print("Data prepraed")
    
    # Train Model
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    message = trange(args.epochs)
    loss_ensemble = ExpEnsemble(n_samples, alpha=ensambling)
    clean_flag = torch.zeros(n_samples)
    noisy_flag = torch.zeros(n_samples)
    loss_filter = torch.ones(n_samples)
    threshold = 0
    auc_score = []
    for epoch in message:
        model.train()
        ce_arr = []
        mse_arr = []
        logi_arr = []
        loss_arr = torch.zeros(n_samples)
        n_samples = x_train.shape[0]
        train_dataset = TensorDataset(torch.arange(n_samples), x_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        
        mse_loss_fn = nn.MSELoss()
        model.train()
        for ids, x, y_hat in train_loader:
            soft_y = torch.zeros((y_hat.shape[0], n_classes)).scatter_(1, y_hat.view(-1, 1), 1).to(args.device)
            x, y_hat = x.float().to(args.device), y_hat.long().to(args.device)
            if nt_train:
                soft_y = soft_y * smooth + (1.0 - soft_y) / (soft_y.shape[1] - 1) * (1 - smooth)
            recons, logits, features = model(x)
            features = features.squeeze()

            loss = 0
            if epoch <= 80:
                weight = loss_filter[ids].to(args.device)
                ce_loss = SoftCrossEntropy(logits, soft_y)
                loss = torch.mean(ce_loss * weight)
            else:
                ce_loss = SoftCrossEntropy(logits, soft_y)
                loss = torch.mean(ce_loss)
            mse_loss = mse_loss_fn(recons, x)
            
            if nt_train:
                loss = loss + beta * mse_loss

            logi_arr.append(logits.detach().cpu())
            if epoch <= 80: loss_arr[ids] += ce_loss.detach().cpu()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ce_arr += [loss.item()] * x.shape[0]
            mse_arr += [mse_loss.item()] * x.shape[0]
        ce_loss = np.mean(ce_arr)
        mse_arr = np.mean(mse_arr)
        message.set_description("CE=%.4f MSE=%.4f" % (ce_loss, mse_arr))
        
        if epoch <= 80: loss_ensemble.update(loss_arr)
        warmup = 10
        if epoch <= 80:
            if epoch >= warmup: loss_filter[:] = 0
            for c in range(n_classes):
                ids = Y_train == c
                model_loss = loss_ensemble.get()[ids]
                threshold = torch.mean(model_loss)
                class_ids  = torch.arange(Y_train.shape[0])[ids]
                good_data_idx = model_loss < threshold
                if epoch >= warmup: loss_filter[class_ids[good_data_idx]] = 1.0
                lb_score = 1.0 - (model_loss / torch.max(model_loss)).cpu().numpy()
                lb_true  = (Y_train[class_ids] == Y_train_clean[class_ids])
            
        if epoch == 80 and oversample_train:
            sel_X, sel_y = x_train[loss_filter > 0].numpy(), Y_train[loss_filter > 0].numpy()
            sel_X, sel_y = Selective_Oversampling(sel_X, sel_y)
            x_train = torch.from_numpy(sel_X).float()
            Y_train = torch.from_numpy(sel_y).long()
        for param_group in optimizer.param_groups: param_group["lr"] = (1e-2 if epoch <= warmup else 1e-3)
    torch.cuda.empty_cache()
    
    
    # Evaluate Model
    model.eval()
    y_true, y_hat_proba, y_hat_labels = evaluate_model(args, model, x_test, Y_test)
    torch.cuda.empty_cache()
    
    return model, (y_true, y_hat_proba, y_hat_labels)
