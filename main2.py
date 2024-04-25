import argparse
import numpy as np
import yaml
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from sklearn import metrics



from tqdm import tqdm
from datasets.split import splits_2020 as splits
from datasets.utils import get_dataloader

from models.utils import *

cudnn.enabled = True
benchmark = True


def get_loss(logits, logits_mixed, y_a, y_b=None, features=None, mixed_features=None, lam=1, base_weight=1, gamma=1, mixed_loss=True, weight=None):
    w = weight
    batch_size = logits.shape[0]
    if weight is None:
        features = features / features.norm(dim=1, keepdim=True)
        mixed_features = mixed_features / mixed_features.norm(dim=1, keepdim=True)
        features_logits = features @ mixed_features.t()
        modulating_factor = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()
        if mixed_loss:
            logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits_mixed, dim=1)
            loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)

        else:
            logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logpt, y_a)
        w = modulating_factor
    else:
        if mixed_loss:
            logpt = (base_weight+weight)**gamma * F.log_softmax(logits_mixed, dim=1)
            loss = lam*F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)

        else:
            logpt = (base_weight+weight)**gamma * F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logpt, y_a)
        
    return loss, w

def get_loss2(logits, y_a, y_b=None, features=None, mixed_features=None, lam=1, base_weight=1, gamma=1):
    batch_size = logits.shape[0]
    if features is not None:
        features = features / features.norm(dim=1, keepdim=True)
        mixed_features = mixed_features / mixed_features.norm(dim=1, keepdim=True)
        features_logits = features @ mixed_features.t()
        modulating_factor = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(logits.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()
        logpt = (base_weight+modulating_factor)**gamma * F.log_softmax(logits, dim=1)
            
        if y_b is None:
            loss = F.nll_loss(logpt, y_a)
        else:
            loss = lam * F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
    else:
        logpt = F.log_softmax(logits, dim=1)
        if y_b is None:
            loss = F.nll_loss(logpt, y_a)
        else:
            loss = lam * F.nll_loss(logpt, y_a) + (1-lam)*F.nll_loss(logpt, y_b)
        
    return loss


def evaluate(classifier, test_loader, ood_loader, train_features, train_labels, device="cuda"):
    classifier.eval()
    test_preds = []
    test_labels = []
    test_features = []

    out_preds = []
    out_labels = []
    out_features = []

    with torch.no_grad():
        #test_loader
        for _, data in enumerate(test_loader, 0):
            imgs, labels = data[0].to(device), data[1].to(device)
            logits, features = classifier(imgs, return_feat=True)
            #pt = torch.softmax(logits, 1)
            _, preds = torch.max(logits, 1)
            
            test_preds.append(torch2numpy(preds))
            test_labels.append(torch2numpy(labels))
            test_features.append(torch2numpy(features))
        # OOD_loader
        for _, data in enumerate(ood_loader, 0):
            imgs, labels = data[0].to(device), data[1].to(device)
            logits, features = classifier(imgs, return_feat=True)
            _, preds = torch.max(logits, 1)

            out_preds.append(torch2numpy(preds))
            out_labels.append(torch2numpy(labels))
            out_features.append(torch2numpy(features))

    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_features = np.concatenate(test_features, axis=0)

    out_preds = np.concatenate(out_preds, axis=0)
    out_labels = np.concatenate(out_labels, axis=0)
    out_features = np.concatenate(out_features, axis=0)
    #print("scores=", scores)

    train = preprocess(train_features, train_labels)
    test_in = preprocess(test_features)
    test = preprocess(out_features)
    

    # calculating the first singular vectors using training features
    first_sing_vecs = []
    for l, data in enumerate(train):
        if len(data) != 0:
            u = calculate_sing_vec(data)
            first_sing_vecs.append(u)

    
    first_sing_vecs = np.array(first_sing_vecs)   #shape=[num_classes, n_dimension]

    # OOD test
    score_in = OOD_test(test_in, score_func, first_sing_vecs)
    target_in = np.zeros_like(score_in)

    score_out = OOD_test(test, score_func, first_sing_vecs)
    target_out = np.ones_like(score_out)

    targets = np.concatenate((target_in, target_out))
    scores = np.concatenate((score_in, score_out))

    fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
    fpr95 = fpr[tpr >= 0.95][0]
    auc = metrics.auc(fpr, tpr)
    detection_error = np.min(0.5 * (1 - tpr) + 0.5 * fpr)

    test_acc = float(np.sum(test_preds==test_labels))/test_preds.shape[0]
    print('FPR @95TPR:', fpr95)

    print('Detection Error:', detection_error)

    print('AUC: ', auc)

    results = {"auc":auc, "fpr":fpr95, "detect_error":detection_error, "closed_accuracy": test_acc}
    
    return results
    


def train(config):
    training_opt = config['training_opt']
    dataset_cofig = config['dataset']
    
    log_dir = os.path.join(training_opt['log_dir'], str(dataset_cofig['fold']))
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')
    
    #step 1: prepare data_loader
    data_loader, dataset_cofig = get_dataloader(dataset_cofig)
    num_classes = dataset_cofig['num_classes']
    device = torch.device("cuda") 
    
    #step 3: prepare CNN or ViT
    classifier_args = config['networks']
    classifier_args['params']['num_classes'] = num_classes
    classifier_def_file = classifier_args['def_file']
    classifier = source_import(classifier_def_file).create_model(**classifier_args['params'])
    classifier = nn.DataParallel(classifier)
    classifier = classifier.to(device)

    optimizer_args = config['optim_params']
    optimizer = optim.SGD(classifier.parameters(), lr=optimizer_args['lr'], 
                          momentum=optimizer_args['momentum'], 
                          weight_decay=optimizer_args['weight_decay'])
    
    #step 4: prepare metrics
    #evaluator = OSREvaluation(data_loader['test'], data_loader['ood'])

    #scheduler
    if config['coslr']:
        print("===> Using coslr eta_min={}".format(config['endlr']))
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=training_opt['warmup_epoch'], t_total=training_opt['num_epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['steps'])

    acc_record = 0
    total_steps = 0
    for epoch in range(1, training_opt['num_epochs'] + 1):
        torch.cuda.empty_cache()
        total_preds = []
        total_labels = []
        total_features = []

        classifier.train()
        for data in tqdm(data_loader['train']):
            total_steps = total_steps + 1
            x = data[0].to(device)
            y = data[1].to(device).long()
            x_mixed, y_mixed, lam = mixup_data(x, y, alpha=training_opt['alpha'])
            
            if training_opt['mixed_loss']:
                logits_mixed1, feat_mixed1 = classifier(x_mixed, return_feat=True)
                
                loss1 = get_loss2(logits=logits_mixed1, y_a=y, y_b=y_mixed, lam=lam)
            
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

                logits_mixed2, feat_mixed2 = classifier(x_mixed, return_feat=True)
                
                loss = get_loss2(logits=logits_mixed2, 
                                y_a=y, y_b=y_mixed,
                                features=feat_mixed1, 
                                mixed_features=feat_mixed2,
                                base_weight=training_opt['base_weight'], 
                                gamma=training_opt['gamma'])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    logits_mixed2, feat_mixed2 = classifier(x, return_feat=True)
            
            else:
                logits_mixed1, feat_mixed1 = classifier(x, return_feat=True)
                
                loss1 = get_loss2(logits=logits_mixed1, y_a=y)
            
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

                logits_mixed2, feat_mixed2 = classifier(x, return_feat=True)
                
                loss = get_loss2(logits=logits_mixed2, 
                                y_a=y,
                                features=feat_mixed1, 
                                mixed_features=feat_mixed2,
                                base_weight=training_opt['base_weight'], 
                                gamma=training_opt['gamma'])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                


            '''
            logits_mixed, feat_mixed = classifier(x_mixed, return_feat=True)
            
            loss.backward()
            if total_steps % training_opt['num_accmutations']==0 or total_steps % len(data_loader['train'])==0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            '''

            #for OOD
            _, pred = torch.max(logits_mixed2, dim=1)
            total_preds.append(torch2numpy(pred))
            total_labels.append(torch2numpy(y))
            total_features.append(torch2numpy(feat_mixed2))
            

        total_preds = np.concatenate(total_preds, axis=0)
        total_labels= np.concatenate(total_labels, axis=0)
        total_features = np.concatenate(total_features, axis=0)
        train_acc = float(np.sum(total_preds==total_labels))/total_preds.shape[0]


        #evaluation
        
        if epoch>=training_opt['num_epochs']-10:
            test_result = evaluate(classifier, data_loader['test'], data_loader['ood'], total_features, total_labels)

            lr_current = max([param_group['lr'] for param_group in optimizer.param_groups])

            log_str = ['[%d/%d]  learing_rate: %.5f  train_acc: %.4f  test_acc: %.4f  AUROC: %0.4f  detection_error: %.4f   loss: %.4f' 
                    % (epoch, training_opt['num_epochs'], lr_current, \
                        train_acc, test_result['closed_accuracy'], \
                        test_result['auc'], test_result['detect_error'],  \
                        loss.item())]
        else:
            lr_current = max([param_group['lr'] for param_group in optimizer.param_groups])

            log_str = ['[%d/%d]  learing_rate: %.5f  train_acc: %.4f  loss: %.4f' 
                    % (epoch, training_opt['num_epochs'], lr_current, \
                        train_acc,  loss.item())]

        
        #print(log_str)
        print_write(log_str, log_file)
    
        states = {
            'epoch':epoch,
            'acc': train_acc,
            'classifier_dict':classifier.state_dict(),
            'optimizer':optimizer.state_dict()
        }

        if train_acc>acc_record:
            acc_record = train_acc
            filename = "best_checkpoint.pth"
            filename = os.path.join(log_dir, filename)
            save_checkpoint(states, filename)
            feature_dir = os.path.join(log_dir, "best_feature.npz")
            np.savez(feature_dir, features=total_features)
        
        filename = "last_checkpoint.pth"
        filename = os.path.join(log_dir, filename)
        save_checkpoint(states, filename)

        feature_dir = os.path.join(log_dir, "last_feature.npz")
        np.savez(feature_dir, features=total_features)

        #scheduler
        scheduler.step()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/CIFAR10/experiment.yaml', type=str)
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    
    # ============================================================================
    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)

    config['local_rank'] = args.local_rank
    options = config['dataset']
    
    for i in range(len(splits[options['dataset']])):
        config['dataset']['fold'] = i
        train(config)
    print('ALL COMPLETED.')

