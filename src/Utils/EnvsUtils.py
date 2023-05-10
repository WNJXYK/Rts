import os, torch, numpy
import random
import torch

__all__ = ['reset_environment']

def reset_environment(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    return args.device