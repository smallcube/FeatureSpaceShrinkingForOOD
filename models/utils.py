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
    
    #modified by Bigcircle
    score = np.arccos(corr) 
    #score = corr
    if len(corr.shape) == 3:
        score = np.min(score, axis=1)
        score = np.min(score, axis=0)

    elif len(corr.shape) == 2:
        score = np.min(score, axis=0)

    return score

def get_svd_per_class(train, labels, topN=0.6):
    svd_per_class, svd_values_class = [], []
    for l in set(labels):
        data = np.array(train[labels == l])
        data = data.transpose()

        #print("data.shape=", data.shape)

        USV = np.linalg.svd(data)
        K = round( min(USV[0].shape[0], USV[0].shape[1])*topN)
        #print(USV[0][:, 0:K].shape)
        svd_per_class.append(USV[0][:, 0:K])
        svd_values_class.append(USV[1][0:K])
    
    return svd_per_class, svd_values_class

def score_func_svd(D, first_sing_vecs, split_N=10, top_N=0.6):
    scores = []
    for current_pos in range(0, D.shape[0], split_N):
        test_feature = D[current_pos:current_pos+split_N, :] if current_pos+split_N<=D.shape[0] else D[current_pos:, :]
        #original_features = test_feature / np.linalg.norm(test_feature, axis=1, ord=2, keepdims=True)
        original_features = test_feature
        similar = []
        
        for i, first_sing_vecs_i in enumerate(first_sing_vecs):
            data = np.concatenate([first_sing_vecs_i, test_feature.transpose()], axis=1)
            K = round(min(data.shape[0], data.shape[1])*top_N)
            U, S, V = np.linalg.svd(data)
            U_k = U[:, :K]
            S_k = np.diag(S[:K])
            V_k = V[:K, :]
            #print("U_k.shape=", U_k.shape, "   S_k.shape=", S_k.shape, "  V_k.shape=", V_k.shape, "   k=", K, "   V.shape=", V.shape)
            reconstructed = U_k@S_k@V_k
            reconstructed_features = reconstructed[:, -test_feature.shape[0]:]

            
            #reconstructed_features = reconstructed_features / np.linalg.norm(reconstructed_features, axis=0, ord=2, keepdims=True)
            cosine_similirty = original_features @ reconstructed_features

            cosine_similirty = np.diag(cosine_similirty).reshape(-1, 1)
            #print("i=", i, "   cosine_similirty=", cosine_similirty, "    shape=", cosine_similirty.shape)
            similar += [np.abs(cosine_similirty)]

        similar = np.concatenate(similar, axis=1)
        scores += [similar]
    scores = np.concatenate(scores, axis=0)
    #print(scores, "   shape=", scores.shape)
    return scores

def score_func_svd_weighted(D, first_sing_vecs, svd_values, split_N=10, top_N=0.6):
    scores = []
    for current_pos in range(0, D.shape[0], split_N):
        test_feature = D[current_pos:current_pos+split_N, :] if current_pos+split_N<=D.shape[0] else D[current_pos:, :]
        #original_features = test_feature / np.linalg.norm(test_feature, axis=1, ord=2, keepdims=True)
        original_features = test_feature
        similar = []
        
        for i, first_sing_vecs_i in enumerate(first_sing_vecs):
            corr = np.matmul(test_feature, first_sing_vecs_i)
            #corr /= np.linalg.norm(first_sing_vecs_i, axis=0) + 1e-4
            corr = np.abs(corr)
            corr = np.matmul(corr, svd_values[i].reshape(-1, 1)) / np.sum(svd_values[i])
            cosine_weighted = corr.reshape(-1, 1)
            similar += [cosine_weighted]

        similar = np.concatenate(similar, axis=1)
        scores += [similar]
    scores = np.concatenate(scores, axis=0)
    #print(scores, "   shape=", scores.shape)
    return scores

def score_func_svd_distance(D, first_sing_vecs, split_N=10, top_N=0.6):
    scores = []
    for current_pos in range(0, D.shape[0], split_N):
        test_feature = D[current_pos:current_pos+split_N, :] if current_pos+split_N<=D.shape[0] else D[current_pos:, :]
        #original_features = test_feature / np.linalg.norm(test_feature, axis=1, ord=2, keepdims=True)
        original_features = test_feature
        similar = []
        
        for i, first_sing_vecs_i in enumerate(first_sing_vecs):
            data = np.concatenate([first_sing_vecs_i, test_feature.transpose()], axis=1)
            K = round(min(data.shape[0], data.shape[1])*top_N)
            U, S, V = np.linalg.svd(data)
            U_k = U[:, :K]
            S_k = np.diag(S[:K])
            V_k = V[:K, :]
            #print("U_k.shape=", U_k.shape, "   S_k.shape=", S_k.shape, "  V_k.shape=", V_k.shape, "   k=", K, "   V.shape=", V.shape)
            reconstructed = U_k@S_k@V_k
            reconstructed_features = reconstructed[:, -test_feature.shape[0]:]

            
            #reconstructed_features = reconstructed_features / np.linalg.norm(reconstructed_features, axis=0, ord=2, keepdims=True)
            distance = np.linalg.norm(original_features - reconstructed_features.transpose(), ord=2)

            #distance = np.diag(distance).reshape(-1, 1)
            #print("i=", i, "   cosine_similirty=", cosine_similirty, "    shape=", cosine_similirty.shape)
            similar += [np.array(np.abs(distance)).reshape(-1, 1)]

        similar = np.concatenate(similar, axis=1)
        scores += [similar]
    scores = np.concatenate(scores, axis=0)
    #print(scores, "   shape=", scores.shape)
    return scores


def calculate_sing_vec(A, topN=1):
    try:
        import irlb
        # print('irlb package is installed for fast svd, using irlb')
        USV = irlb.irlb(A, 2)
    except ImportError:
        # print('No irlb package installed for fast svd, using numpy')
        USV = np.linalg.svd(A)
    
    first_sing_vec = USV[0][:, 0]
    #print("USV.shape=", USV[0].shape, "   A.shape=", A.shape, "   sing_vec.shape=", first_sing_vec.shape)
    return first_sing_vec

def preprocess(D, labels=None):
    if labels is None:
        data = np.array(D)
        D_out = data.transpose()
    else:
        '''
        data = np.array(D)
        D_out = data.transpose()
        '''
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
