import otdd
from otdd.pytorch.datasets import load_imagenet, load_torchvision_data, load_torchvision_data_shuffle, load_torchvision_data_perturb, load_torchvision_data_keepclean
from otdd.pytorch.distance import DatasetDistance, FeatureCost

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

import matplotlib.pyplot as plt
from torch import tensor
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from copy import deepcopy as dpcp
import pickle 
import time

# import torchshow as ts

from torchvision.utils import make_grid
from torch.utils.data import random_split, Dataset, TensorDataset, DataLoader

import argparse

from flalg.experiments import *
from otdd.pytorch.distance import  FeatureCost
from otdd.pytorch.moments import *
from otdd.pytorch.utils import *


def get_args():
    
    parser = argparse.ArgumentParser()

    # 
    parser.add_argument('--cnum', type=int, required=True,
                    help='number of cuda in the server')
    parser.add_argument('--n', type=int, required=True,
                        help='number of data')

    # fl-algorithms 
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')


    
    args = parser.parse_args()
    return args



class DatasetSplit(Dataset):
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.LongTensor(self.dataset.targets)[idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

class InterpMeas:
    def __init__(self, metric: str = "sqeuclidean", t_val: float = 0.5):
    self.metric = metric
    self.t_val = t_val

def fit(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    # a: np.ndarray | None = None,
    # b: np.ndarray | None = None,
):
    """

    Args:
        X `numpy.ndarray`
        Y `numpy.ndarray`
        a `numpy.ndarray` | `NONE` The weights of the empirical distribution X . Defaults to None with equal weights.
        b `numpy.ndarray` | `NONE` The weights of the empirical distribution X . Defaults to None with equal weights.

    Returns:
    """

    t_val = np.random.rand() if self.t_val == None else self.t_val

    nx, ny = X.shape[0], Y.shape[0]
    p = 2 if self.metric == "sqeuclidean" else 1

    a = np.ones((nx,), dtype=np.float64) / nx
    b = np.ones((ny,), dtype=np.float64) / ny

    M = ot.dist(X, Y, metric=self.metric)

    norm = np.max(M) if np.max(M) > 1 else 1
    G0 = ot.emd(a, b, M / norm)

    Z = (1 - t_val) * X + t_val * (G0 * nx) @ Y

    return Z
   
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 10)
#         self.linear1 = nn.Linear(128, 10)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
#         return out # only for embedder
#         out = self.linear1(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])



def get_fl_model_log_error(train_loaders, test_loader,args):

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
    global_model = global_models[0]

    global_para = global_model.state_dict()

    if args.is_same_initial:
        for net_id, net in nets.items():
            net.load_state_dict(global_para)
 

    global_test_accuracy = []
    global_test_loss = []

    weight_trainerr  = []
    uni_trainerr  = [] 
    if args.alg == 'fedavg':

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            selected = arr[:int(args.n)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                nets[idx].load_state_dict(global_para)

            nets, local_train_loss = local_train_net(nets, args, train_loaders, test_dl = test_loader, device=device)
     
            # # update global model
            total_data_points = sum([len(dl.dataset) for dl in train_loaders])
            fed_avg_freqs = [len(dl.dataset) / total_data_points for dl in train_loaders]


            for idx in range(args.n):
                net_para = nets[idx].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            global_model.to(device)
         
            test_acc, test_loss = compute_test(global_model, test_loader, device=device)

            global_test_accuracy.append(test_acc)
            global_test_loss.append(test_loss)

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(args.n)])
            uni_loss =  np.sum(local_train_loss) / len(local_train_loss)

            weight_trainerr.append(weight_loss)
            uni_trainerr.append(uni_loss)


    elif args.alg == 'fedprox':

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

          
            selected = arr[:int(args.n)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            nets, local_train_loss = local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)



            global_model.to(device)

            test_acc, test_loss = compute_test(global_model, test_loader, device=device)

            global_test_accuracy.append(test_acc)
            global_test_loss.append(test_loss)

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(args.n)])
            uni_loss =  np.sum(local_train_loss) / len(local_train_loss)

            weight_trainerr.append(weight_loss)
            uni_trainerr.append(uni_loss)


    return global_test_accuracy, global_test_loss, weight_trainerr, uni_trainerr



def process_data(data_loader):
    
    net_test = PreActResNet18()
    net_test = net_test.to(device)
    net_test.load_state_dict(torch.load('checkpoint/preact_resnet18.pth', map_location=str('cuda:'+str(cuda_num))))
    net_test.eval()

    embedder = net_test.to(device)
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False
    
    features = data_loader.dataset.tensors[0]  
    labels = data_loader.dataset.tensors[1] 

    with torch.no_grad(): 
        embedded_features = embedder(features)

    dim = embedded_features.size(1)
    
    dataset = TensorDataset(embedded_features, labels)
    new_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    vals, cts = torch.unique(labels, return_counts=True)
    # min_labelcount = 2
    # classes = torch.sort(vals[cts >= min_labelcount])[0]
    idxs = np.arange(len(labels))  
    
    M, C = compute_label_stats(new_loader, labels, idxs, classes, diagonal_cov=True)
    
    DA = (embedded_features.view(-1, dim), labels.to(device))
    XA = augmented_dataset(DA, means=M, covs=C, maxn=10000)
    
    return XA


def get_ot_dist_private(local_train_loaders,val_loader):
    """"
    k : supporting size of the global gamma 
    t_val : could be any value between (0,1)
    """

    k = 200  
    t_val = 0.5 

    aug_train_data = []
    for local_dl in local_train_loaders:
        aug_train_data.append(process_data(local_dl))
    
    aug_val_data = process_data(val_loader)
    
    dim = aug_train_data[0].dim[1]
    global_gamma = np.random.randn(k, dim)
    interp_mea = InterpMeas(metric='sqeuclidean', t_val=t_val)
    
    train_int = [] 
    for local_data in aug_train_data:
        train_int.append( interp_mea.fit(local_data, global_gamma) ) 
    
    val_int  =  interp_mea.fit(aug_val_data, global_gamma)
    

    dist = 0 
    return dist 

def get_ot_dist(train_loader, test_loader, n=5000):
    #Todo:change the centralized calculation to decentrlized calculation
    
    net_test = PreActResNet18()
    net_test = net_test.to(device)
    net_test.load_state_dict(torch.load('checkpoint/preact_resnet18.pth', map_location=str('cuda:'+str(cuda_num))))
    net_test.eval()

    embedder = net_test.to(device)
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets
    feature_cost = FeatureCost(src_embedding = embedder,
                               src_dim = (3,32,32),
                               tgt_embedding = embedder,
                               tgt_dim = (3,32,32),
                               p = 2,
                               device='cuda')

    dist = DatasetDistance(train_loader, test_loader,
                           inner_ot_method = 'exact',
                           debiased_loss = True,
                           feature_cost = feature_cost,
                           λ_x=1.0, λ_y=1.0,
                           sqrt_method = 'spectral',
                           sqrt_niters=10,
                           precision='single',
                           p = 2, entreg = 1e-2,
                           device='cuda')
    k = dist.distance(maxsamples = n, return_coupling = True)

    return k[0].item()

def dataset_q(q1_amt, q2_amt, num, train_feats, train_labels,label_idx):
    # two datasets, q=0 -> dataset2, q=1 -> dataset1
    # validation set: unbiased sample from MNIST validation set
    # dataset1: class 0-4: 99% (19.8% each class), class 5-9: 1% (0.2% each class)
    # dataset2: class 0-4: 2% (0.4% each class), class 5-9: 98% (19.6% each class)
    # near balance at q=0.5

    ds1_idx = []
    ds2_idx = []
    ds3_idx = []
    ds1_labels = []
    ds2_labels = []
    ds3_labels = []
    # ds1_features = []
    # ds2_features = []

    d1c1 = 0.2425
    d1c2 = 0.005
    d1c3 = 0.005

    d2c1 = 0.0057
    d2c2 = 0.32
    d2c3 = 0.0057

    d3c1 = 0.0014
    d3c2 = 0.0014
    d3c3 = 0.33
    
    
    
    # sample size
    n = num # size of dataset for training (use for construct)
    # ratio
    q1 = q1_amt # q * dataset 1
    q2 = q2_amt # q * dataset 1
    q3 = 1-q1-q2 # q * dataset 1

    for i in range(4):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c1)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c1)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c1)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c1)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c1)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c1)))*i)
    for i in range(4, 7):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c2)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c2)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c2)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c2)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c2)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c2)))*i)
    for i in range(7, 10):
        ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c3)))])
        ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c3)))])
        ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c3)))])
        ds1_labels.append(np.ones(int(np.rint(n*q1*d1c3)))*i)
        ds2_labels.append(np.ones(int(np.rint(n*q2*d2c3)))*i)
        ds3_labels.append(np.ones(int(np.rint(n*q3*d3c3)))*i)

    ds1_features_fl = train_feats[np.concatenate(ds1_idx)]
    ds2_features_fl = train_feats[np.concatenate(ds2_idx)]
    ds3_features_fl = train_feats[np.concatenate(ds3_idx)]
    ds1_features = train_feats[np.concatenate(ds1_idx)]
    ds2_features = train_feats[np.concatenate(ds2_idx)]
    ds3_features = train_feats[np.concatenate(ds3_idx)]
    train_x_2d = np.concatenate([ds1_features, ds2_features, ds3_features])

    ds1_labels = np.concatenate(ds1_labels)
    ds2_labels = np.concatenate(ds2_labels)
    ds3_labels = np.concatenate(ds3_labels)

    # train_x = np.concatenate([ds1_features_fl, ds2_features_fl, ds3_features_fl])
    # train_y = np.concatenate([ds1_labels, ds2_labels, ds3_labels])

    
    
    return ds1_features_fl, ds2_features_fl, ds3_features_fl,ds1_labels, ds2_labels, ds3_labels


def main(args, data_dict):

    breaks = 10
    reps = 2

        
    train_features = data_dict['train_features']
    train_labels = data_dict['train_labels']
    test_features = data_dict['test_features']
    test_labels = data_dict['test_labels']
    train_label_idx = data_dict['train_label_index']
    test_label_idx = data_dict['test_label_index']

    n = args.n
    batch_size = args.batch_size

    # make test dataloader
    test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_features).permute(0,3,1,2), torch.LongTensor(test_labels)), 
                                                    batch_size=batch_size, 
                                                    shuffle=False)

    qstrainerrlog = []
    qstesterrlog = []
    qsotlog = []
    qsaccs = []
    for l in range(breaks+1):
        
        trainerrlog = []
        testerrlog = []
        otlog = []
        accs = []
        
        for j in range(breaks+1): # going through q, from 0 to 1 - 20 points
            start_t = time.time()
            q1 = l/10
            q2 = j/10
            q3 = 1-q1-q2
            if q3<0:
                break

            # create dataset
            # train_x, train_y = dataset_q(q1, q2, n, train_features, train_labels)
            train_x_1,train_x_2, train_x_3, train_y_1,train_y_2, train_y_3 = dataset_q(q1, q2, n, train_features, train_labels,train_label_idx)
            

            train_combine_x = np.concatenate([train_x_1, train_x_2, train_x_3])
            train_combine_y = np.concatenate([train_y_1, train_y_2, train_y_3])
            

            train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_combine_x).permute(0,3,1,2), 
                                                    torch.LongTensor(train_combine_y)), 
                                                    batch_size=batch_size, 
                                                    shuffle=True)
            
            train_x_ls = [train_x_1,train_x_2,train_x_3]
            train_y_ls = [train_y_1,train_y_2,train_y_3]

            local_train_loaders = []
            # make train dataloader
            for i in range(args.n):
                local_train_loaders.append(torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_x_ls[i]).permute(0,3,1,2), 
                                                    torch.LongTensor(train_y_ls[i])), 
                                                    batch_size=batch_size, 
                                                    shuffle=True))
            for rep in range(reps):
                # get OT dist
                ot_dist_combine = get_ot_dist(train_loader, test_loader, n=n)
                

                test_loss, test_acc, train_loss = get_fl_model_log_error(local_train_loaders, test_loader, args)

                trainerrlog.append(train_loss)
                testerrlog.append(test_loss) 
                accs.append(test_acc)
                otlog.append(ot_dist_combine)
                
        
        qstrainerrlog.append(trainerrlog)
        qstesterrlog.append(testerrlog)
        qsotlog.append(otlog)
        qsaccs.append(accs)


    # pickle.dump([qstrainerrlog,qstesterrlog,qsotlog,qsaccs], open(f'cif10_3sources/{args.alg}_unbalanced_{n}.res', 'wb' ))


    args_dict = vars(args)
    
    results = {
        'args': args_dict,
        'train_err_log': qstrainerrlog,
        'test_err_log': qstesterrlog,
        'ot_log': qsotlog,
        'accuracies': qsaccs
    }
    pickle.dump(results, open(f'cif10_3sources/{args.alg}_unbalanced_{args.n}.res', 'wb'))
       
  

if __name__ == "__main__":

    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'

    
    


    print(f"procs cnum {args.cnum}")

    print(f"data cnum {args.n}")
        
    print("end")


    cuda_num = args.cnum
    import torch
    print(torch.__version__)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_num)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    torch.cuda.set_device(cuda_num)
    print("Cuda device: ", torch.cuda.current_device())
    print("cude devices: ", torch.cuda.device_count())
    device = torch.device('cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu')


    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    data_all = pickle.load( open('data/cifar10.data', 'rb') )
    train_features, train_labels, test_features, test_labels  = data_all

    label_idx = []
    for i in range(10):
        label_idx.append((train_labels==i).nonzero()[0])
        
    test_label_idx = []
    for i in range(10):
        test_label_idx.append((test_labels==i).nonzero()[0])


        # 创建数据字典
    data_dict = {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        'train_label_index':label_idx,
        'test_label_index': test_label_idx
    }

    main(args,data_dict)