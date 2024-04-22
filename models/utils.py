import importlib
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_write(print_str, log_file):
    print(*print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha>0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1-lam)*x[index, :]
    y_b = y[index]
    
    return mixed_x, y_b, lam

def save_checkpoint(states, filename):
    torch.save(states, filename)

def score_func(D, first_sing_vecs):
    measure = first_sing_vecs
    corr = correlation(measure, D)
    score = np.arccos(corr)
    if len(corr.shape) == 3:
        score = np.min(score, axis=1)
        score = np.min(score, axis=0)

    elif len(corr.shape) == 2:
        score = np.min(score, axis=0)

    return score


def calculate_sing_vec(A):
    try:
        import irlb
        # print('irlb package is installed for fast svd, using irlb')
        USV = irlb.irlb(A, 2)
    except ImportError:
        # print('No irlb package installed for fast svd, using numpy')
        USV = np.linalg.svd(A)
    first_sing_vec = USV[0][:, 0]
    return first_sing_vec

def preprocess(D, labels=None):
    if labels is None:
        data = np.array(D)
        D_out = data.transpose()
    else:
        D_out = []
        for l in set(labels):
            data = np.array(D[labels == l])
            if len(data) != 0:
                D_out.append(data.transpose())
    return D_out


def correlation(A, B):
    corr = np.matmul(A, B)
    if len(B.shape) == 2:
        corr /= np.linalg.norm(B, axis=0) + 1e-4
    elif len(B.shape) == 3:
        corr /= np.linalg.norm(B, axis=1)[:, None, :] + 1e-8
    corr = np.abs(corr)
    return corr


def OOD_test(D, score_func, first_sing_vecs):
    score = score_func(D, first_sing_vecs)
    return score

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))