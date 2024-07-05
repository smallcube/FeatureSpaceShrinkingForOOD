import importlib
import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from numpy.linalg import norm, pinv
from scipy.special import logsumexp, softmax
from tqdm import tqdm
from sklearn.decomposition import PCA


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

def calculate_pca(X):
    pca = PCA(n_components=50)  
    X_pca = pca.fit_transform(X.T)
    X_pca = X_pca.T
    p = pca.components_
    #print(X.shape[0])
    #print(X.shape[1])  
    #print(X_pca.shape[0])
    #print(X_pca.shape[1])
    #print(p.shape[0])
    #print(p.shape[1])
    return X_pca,p
   
    X_pca = pca.fit_transform(X_scaled)   
def score_func4(mean, prec, feature_id_val):
    score = -np.array(
        [1-(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
         for f in tqdm(torch.from_numpy(feature_id_val).cuda().float())])
    return score
    
def score_func3(D, alpha, lo, u, NS):
    vlogit_id_val = norm(np.matmul(D - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(lo, axis=-1)
    score = 1-(-vlogit_id_val + energy_id_val)
    return score

def score_func2(D, alpha, lo, R):##D[dim*batch_size]R[dim-num_classes*dim]lo[num_classes*batch_size]
    device=R.device
    score=[]
    with torch.no_grad():
        logits = lo
        D_tensor = torch.tensor(D, dtype=torch.float32, device=device) 
        _,batch_size = D.shape
        print(batch_size) 
        logits_complement = D_tensor.t() @ R.t() @ R @ D_tensor
        logits=np.exp(logits)
        for i in range(0, batch_size):
            raw_score=alpha * torch.sqrt(logits_complement[i][i])
            logits_score=raw_score/(raw_score+np.sum(logits[:,i]))
            score.append(logits_score.item())
    scores_np = np.array(score) 
    return scores_np 

def score_func5(D, first_sing_vecs,singpca):
    num_classes = first_sing_vecs.shape[0]
    corr = []
    for i in range(0,num_classes):
        measure = np.array(singpca[i])
        #print(D.shape[0])
        #print(D.shape[1])
        #print(measure.shape[0])
        #print(measure.shape[1])
        D_pca = measure @ D
        #print(D_pca.shape[0])
        #print(D_pca.shape[1])
        #print(first_sing_vecs[i].shape[0])
        #print(first_sing_vecs[i].shape[1])
        measure = first_sing_vecs[i].reshape(1,-1)
        #print(D_pca.shape[0])
        #print(D_pca.shape[1])
        #print(measure.shape[0])
        #print(measure.shape[1])
        co = correlation(measure, D_pca)
        #print(co.shape[0])
        #print(co.shape[1])
        co = np.squeeze(co)
        corr.append(co)
    #measure = first_sing_vecs
    #corr = correlation(measure, D)
    corr = np.array(corr)
    #print(corr.shape[0])
    #print(corr.shape[1])
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

def score_func(D, train):
    num_cases = D.shape[1]
    corr = []
    #print(D.shape[0])
    #print(D.shape[1])
    for i in range(0,num_cases):
        corr.append([])
        now = D[:,i]
        now = np.squeeze(now)
        now = now.reshape(1,-1)
        for l,data in enumerate(train):
            data_new = np.concatenate((data.T,now),axis=0)
            try:
               import irlb
               # print('irlb package is installed for fast svd, using irlb')
               U,S,V = irlb.irlb(data_new, 2)
            except ImportError:
               # print('No irlb package installed for fast svd, using numpy')
               U,S,V = np.linalg.svd(data_new)
            total_sum = np.sum(S)
            svd_k = 0.6
            lim = total_sum*svd_k
            k = 0
            now_total=0
            for j,singular_value in enumerate(S):
                now_total+=singular_value
                if now_total>=lim:
                   k = j+1
                   break
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        V_k = V[:k, :]
        re_now = U_k@S_k@V_k
        co = correlation(re_now[-1].reshape(1,-1), now.T)
        corr[i].append(co)
    corr = np.array(corr)
    score = np.arccos(corr)
    if len(corr.shape) == 3:
        score = np.min(score, axis=2)
        score = np.min(score, axis=1)
    elif len(corr.shape) == 2:
        score = np.min(score, axis=1)

def correlation(A, B):
    corr = np.matmul(A, B)
    if len(B.shape) == 2:
        corr /= np.linalg.norm(B, axis=0) + 1e-4
    elif len(B.shape) == 3:
        corr /= np.linalg.norm(B, axis=1)[:, None, :] + 1e-8
    corr = np.abs(corr)
    return corr


def OOD_test(D, score_func, train):
    score = score_func(D, train)
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