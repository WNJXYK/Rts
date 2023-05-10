import argparse
import numpy as np
import src.Utils as Utils
from sklearn.model_selection import train_test_split

import src.Methods.Proposal as Proposal
import torch

parser = argparse.ArgumentParser(description='Trace')
parser.add_argument('--dataset', type=str, default="ECG5000")
parser.add_argument('--method', type=str, default="Proposal", choices=["Proposal"])
parser.add_argument('--noise-type', type=str, default="sym", choices=['sym', 'asym'])
parser.add_argument('--noise-rate', type=float, default=0.0)
parser.add_argument('--gpu' , default='0', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--lr'  , default=1e-3, type=float, dest='lr')
parser.add_argument('--wd'  , default=1e-4, type=float, dest='wd')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=-1, type=int)
parser.add_argument('--num-workers', default=0, type=int)
parser.add_argument('--normalization', type=str, default='batch')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--classifier-dim', type=int, default=128, help='Dimension of classifier')
parser.add_argument('--embedding-size', type=int, default=128, help='Dimension of embedding')
parser.add_argument('--kernel-size', type=int, default=4)
parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--padding', type=int, default=2)
parser.add_argument('--patience', type=int, default=10)

def main():
    # Init
    args = parser.parse_args()
    Utils.reset_environment(args)
    results = Utils.Results(args)
    X, y = Utils.load_dataset(args.dataset)
    X, y, clean_y = Utils.generate_noise(X, y, classes=np.unique(y), dataset=args.dataset, ratio=args.noise_rate, noise_type=args.noise_type)
    train_X, train_y, clean_train_y, valid_X, valid_y, clean_valid_y, test_X, test_y = Utils.process_data(X, y, clean_y)
    if args.batch_size == -1: args.batch_size = min(128, int(train_X.shape[0] / 10.0))
    
    # Train & Evaluation
    results = Utils.Results(args)
    main_wrapper = Proposal.main_wrapper
    model, (y_true, y_hat_proba, y_hat_labels) = main_wrapper(args, train_X, valid_X, test_X, train_y, valid_y, test_y, clean_train_y, clean_valid_y)
    ma_f1, gmean, bacc = Utils.evaluate_results(y_true, y_hat_proba, y_hat_labels)
    results.update(ma_f1, gmean, bacc)

if __name__ == "__main__": main()