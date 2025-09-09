


# import otdd
# from otdd.pytorch.datasets import load_imagenet, load_torchvision_data, load_torchvision_data_shuffle, load_torchvision_data_perturb, load_torchvision_data_keepclean
# from otdd.pytorch.distance import DatasetDistance, FeatureCost

# import torch
# import torchvision


# import torch.optim as optim
# import torchvision.models as models
# from torch.autograd import Variable

# import matplotlib.pyplot as plt
# from torch import tensor
# from torchvision import datasets, transforms
# import pandas as pd
# import numpy as np
# from copy import deepcopy as dpcp
# import pickle 
# import time

# import copy

# # import torchshow as ts

# from torchvision.utils import make_grid
# from torch.utils.data import random_split, Dataset, TensorDataset, DataLoader

# import argparse

# # from flalg.experiments import *
# from flalg.model import SimpleCNN
# # from flalg.utils import compute_acc_loss

# from wd_utils import *
# from data_generator import * 
# from encoder_model import * 
# # import mkdirs

# from main import get_ot_dist


# def dataset_q(q1_amt, q2_amt, num, train_feats, train_labels,label_idx):
#     # two datasets, q=0 -> dataset2, q=1 -> dataset1
#     # validation set: unbiased sample from MNIST validation set
#     # dataset1: class 0-4: 99% (19.8% each class), class 5-9: 1% (0.2% each class)
#     # dataset2: class 0-4: 2% (0.4% each class), class 5-9: 98% (19.6% each class)
#     # near balance at q=0.5

#     ds1_idx = []
#     ds2_idx = []
#     ds3_idx = []
#     ds1_labels = []
#     ds2_labels = []
#     ds3_labels = []
#     # ds1_features = []
#     # ds2_features = []

#     d1c1 = 0.2425
#     d1c2 = 0.005
#     d1c3 = 0.005

#     d2c1 = 0.0057
#     d2c2 = 0.32
#     d2c3 = 0.0057

#     d3c1 = 0.0014
#     d3c2 = 0.0014
#     d3c3 = 0.33
    
    
    
#     # sample size
#     n = num # size of dataset for training (use for construct)
#     # ratio
#     q1 = q1_amt # q * dataset 1
#     q2 = q2_amt # q * dataset 1
#     q3 = 1-q1-q2 # q * dataset 1

#     for i in range(4):
#         ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c1)))])
#         ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c1)))])
#         ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c1)))])
#         ds1_labels.append(np.ones(int(np.rint(n*q1*d1c1)))*i)
#         ds2_labels.append(np.ones(int(np.rint(n*q2*d2c1)))*i)
#         ds3_labels.append(np.ones(int(np.rint(n*q3*d3c1)))*i)
#     for i in range(4, 7):
#         ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c2)))])
#         ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c2)))])
#         ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c2)))])
#         ds1_labels.append(np.ones(int(np.rint(n*q1*d1c2)))*i)
#         ds2_labels.append(np.ones(int(np.rint(n*q2*d2c2)))*i)
#         ds3_labels.append(np.ones(int(np.rint(n*q3*d3c2)))*i)
#     for i in range(7, 10):
#         ds1_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q1*d1c3)))])
#         ds2_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q2*d2c3)))])
#         ds3_idx.append(label_idx[i][np.random.randint(len(label_idx[i]), size=int(np.rint(n*q3*d3c3)))])
#         ds1_labels.append(np.ones(int(np.rint(n*q1*d1c3)))*i)
#         ds2_labels.append(np.ones(int(np.rint(n*q2*d2c3)))*i)
#         ds3_labels.append(np.ones(int(np.rint(n*q3*d3c3)))*i)

#     ds1_features_fl = train_feats[np.concatenate(ds1_idx)]
#     ds2_features_fl = train_feats[np.concatenate(ds2_idx)]
#     ds3_features_fl = train_feats[np.concatenate(ds3_idx)]
#     ds1_features = train_feats[np.concatenate(ds1_idx)]
#     ds2_features = train_feats[np.concatenate(ds2_idx)]
#     ds3_features = train_feats[np.concatenate(ds3_idx)]
#     train_x_2d = np.concatenate([ds1_features, ds2_features, ds3_features])

#     ds1_labels = np.concatenate(ds1_labels)
#     ds2_labels = np.concatenate(ds2_labels)
#     ds3_labels = np.concatenate(ds3_labels)

#     # train_x = np.concatenate([ds1_features_fl, ds2_features_fl, ds3_features_fl])
#     # train_y = np.concatenate([ds1_labels, ds2_labels, ds3_labels])

    
    
#     return ds1_features_fl, ds2_features_fl, ds3_features_fl,ds1_labels, ds2_labels, ds3_labels


# with open('./data/datafile.data', 'rb') as f:
#     train_features, train_labels,test_features,test_labels = pickle.load(f)

# with open('./data/balance_test_100.data', 'rb') as f:
#     test_features,test_labels = pickle.load(f)



# train_features = train_features
# test_features = test_features
# test_features = test_features.reshape(-1,32,32,3)
# train_features = train_features.reshape(-1,32,32,3)



# label_idx = []
# for i in range(10):
#     label_idx.append((train_labels==i).nonzero()[0])
    
# test_label_idx = []
# for i in range(10):
#     test_label_idx.append((test_labels==i).nonzero()[0])



# q1=1
# q2=0

# n = 4000
# train_x_1,train_x_2, train_x_3, train_y_1,train_y_2, train_y_3 = dataset_q(q1, q2, n, train_features, train_labels,label_idx)
# train_combine_x = np.concatenate([train_x_1, train_x_2, train_x_3])
# train_combine_y = np.concatenate([train_y_1, train_y_2, train_y_3])
# train_x_ls = [train_x_1,train_x_2,train_x_3]
# train_y_ls = [train_y_1,train_y_2,train_y_3]
# train_x_ls = [item for item in train_x_ls if len(item)>0 ]
# train_y_ls = [item for item in train_y_ls if len(item)>0 ]

# train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_combine_x).permute(0,3,1,2), 
#                                                 torch.LongTensor(train_combine_y)), 
#                                                 batch_size=64, 
#                                                 shuffle=True)
# local_train_loaders = []
# for i in range(len(train_x_ls)):
#     local_train_loaders.append(torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_x_ls[i]).permute(0,3,1,2), 
#                                         torch.LongTensor(train_y_ls[i])), 
#                                         batch_size=64, 
#                                         shuffle=True))

# test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_features).permute(0,3,1,2), torch.LongTensor(test_labels)), 
#                                                 batch_size=64, 
#                                                 shuffle=False)
# ot_dist_combine = get_ot_dist(train_loader, test_loader, n=n,device='cuda:0')


# print(ot_dist_combine)




# q1=0
# q2=1

# n = 4000
# train_x_1,train_x_2, train_x_3, train_y_1,train_y_2, train_y_3 = dataset_q(q1, q2, n, train_features, train_labels,label_idx)
# train_combine_x = np.concatenate([train_x_1, train_x_2, train_x_3])
# train_combine_y = np.concatenate([train_y_1, train_y_2, train_y_3])
# train_x_ls = [train_x_1,train_x_2,train_x_3]
# train_y_ls = [train_y_1,train_y_2,train_y_3]
# train_x_ls = [item for item in train_x_ls if len(item)>0 ]
# train_y_ls = [item for item in train_y_ls if len(item)>0 ]

# train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_combine_x).permute(0,3,1,2), 
#                                                 torch.LongTensor(train_combine_y)), 
#                                                 batch_size=64, 
#                                                 shuffle=True)
# local_train_loaders = []
# for i in range(len(train_x_ls)):
#     local_train_loaders.append(torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_x_ls[i]).permute(0,3,1,2), 
#                                         torch.LongTensor(train_y_ls[i])), 
#                                         batch_size=64, 
#                                         shuffle=True))

# test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_features).permute(0,3,1,2), torch.LongTensor(test_labels)), 
#                                                 batch_size=64, 
#                                                 shuffle=False)
# ot_dist_combine = get_ot_dist(train_loader, test_loader, n=n,device='cuda:0')


# print(ot_dist_combine)



# q1=0
# q2=0

# n = 4000
# train_x_1,train_x_2, train_x_3, train_y_1,train_y_2, train_y_3 = dataset_q(q1, q2, n, train_features, train_labels,label_idx)
# train_combine_x = np.concatenate([train_x_1, train_x_2, train_x_3])
# train_combine_y = np.concatenate([train_y_1, train_y_2, train_y_3])
# train_x_ls = [train_x_1,train_x_2,train_x_3]
# train_y_ls = [train_y_1,train_y_2,train_y_3]
# train_x_ls = [item for item in train_x_ls if len(item)>0 ]
# train_y_ls = [item for item in train_y_ls if len(item)>0 ]

# train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_combine_x).permute(0,3,1,2), 
#                                                 torch.LongTensor(train_combine_y)), 
#                                                 batch_size=64, 
#                                                 shuffle=True)
# local_train_loaders = []
# for i in range(len(train_x_ls)):
#     local_train_loaders.append(torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_x_ls[i]).permute(0,3,1,2), 
#                                         torch.LongTensor(train_y_ls[i])), 
#                                         batch_size=64, 
#                                         shuffle=True))

# test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_features).permute(0,3,1,2), torch.LongTensor(test_labels)), 
#                                                 batch_size=64, 
#                                                 shuffle=False)
# ot_dist_combine = get_ot_dist(train_loader, test_loader, n=n,device='cuda:0')


# print(ot_dist_combine)




import otdd
from otdd.pytorch.datasets import load_imagenet, load_torchvision_data, load_torchvision_data_shuffle, load_torchvision_data_perturb, load_torchvision_data_keepclean
from otdd.pytorch.distance import DatasetDistance, FeatureCost

import torch
import torchvision


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

import copy

# import torchshow as ts

from torchvision.utils import make_grid
from torch.utils.data import random_split, Dataset, TensorDataset, DataLoader

import argparse

# from flalg.experiments import *
from flalg.model import SimpleCNN
# from flalg.utils import compute_acc_loss

from wd_utils import *
from data_generator import * 
from encoder_model import * 
# import mkdirs

def get_args():
    
    parser = argparse.ArgumentParser()

    # general 
    parser.add_argument('--cnum', type=int, required=True,
                    help='number of cuda in the server')


    # interpolating measures 
    parser.add_argument('--gamma_size', type=int, default=100, help='supporting size of gamma')
    parser.add_argument('--t_val', type=float, default=0.5,help='value')
    parser.add_argument('--metric', type=str, default='sqeuclidean', help='metric for wasserstein distance')
    parser.add_argument('--budget', type=int, default=1000, help='number of training budget')
    
    
    # fl-algorithms 
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    # parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=3,  help='number of workers in a distributed cluster')
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
    # parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    # parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
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






def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32,16,8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16,8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(args,train_dataloader, net, device="cpu"):


    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # if type(train_dataloader) == type([1]):
    #     pass
    # else:
    #     train_dataloader = [train_dataloader]

    for epoch in range(args.epochs):

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            # epoch_loss_collector.append(loss.item())

        # epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector) 
    
    net.to('cpu')
    logger.info(' ** Training complete **')
    

def local_train_net_scaffold(args, train_loaders, nets, global_model, c_nets, c_global, device="cpu"):

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():

        net.to(device)

        c_nets[net_id].to(device)

        c_delta_para = train_net_scaffold(args, train_loaders[net_id], net, global_model, c_nets[net_id], c_global, device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        
    for key in total_delta:
        total_delta[key] /= len(train_loaders)
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)


    nets_list = list(nets.values())
    return nets_list

def local_train_net_fednova(args, train_loaders, nets, global_model, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)

    for net_id, net in nets.items():

        # move the model to cuda device:
        net.to(device)
        a_i, d_i = train_net_fednova(args, train_loaders[net_id], net, global_model, device=device)

        a_list.append(a_i)
        d_list.append(d_i)

    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list

def train_net_scaffold(args, train_dataloader, net, global_net, c_local, c_global,device="cpu"):
    
    if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.rho, weight_decay=args.reg)
    
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_net.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(args.epochs):
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1


    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_net.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)


    


    net.to('cpu')
    logger.info(' ** Training complete **')

    return  c_delta_para

def train_net_fednova(args, train_dataloader, net, global_net, device="cpu"):
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]


    tau = 0

    for epoch in range(args.epochs):

        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

               
    global_net.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_net.to(device)
    global_model_para = global_net.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_net.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)
    
    net.to('cpu')
    logger.info(' ** Training complete **')

    return a_i, norm_grad


def train_net_fedprox(args, train_dataloader, net, global_net, device="cpu"):

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.rho, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(args.epochs):

        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((args.mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
  
    # train_acc,train_loss = compute_acc_loss(net, train_dataloader, device=device)

    # print('train_acc',train_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')


def local_train_net(args, train_loaders, nets,device="cpu"):
    
    for net_id, net in nets.items():
        # move the model to cuda device:
        net.to(device)

        train_net(args, train_loaders[net_id], net, device=device)

   
    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(args, train_loaders, nets, global_model,  device="cpu"):
    # local_train_loss = []

    for net_id, net in nets.items():
        net.to(device)

        train_net_fedprox(args, train_loaders[net_id], net, global_model, device=device)
        # local_train_loss.append(trainloss)
   
    nets_list = list(nets.values())
    return nets_list



def compute_acc_loss(model, dataloader, moon_model=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    correct, total = 0, 0
    loss = 0

    criterion = nn.CrossEntropyLoss().to(device)
    

    with torch.no_grad():

        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device,dtype=torch.int64)
            if moon_model:
                _, _, out = model(x)
            else:
                out = model(x)

            loss += criterion(out, target).item()
           
            _, pred_label = torch.max(out.data, 1)


            
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()


    if was_training:
        model.train()
    return correct/float(total), loss/float(total)


def get_fl_model_log_error(train_loaders, test_loader,args):
 

    global_test_accuracy = []
    global_test_loss = []

    weight_trainerr  = []
    uni_trainerr  = [] 
    arr = np.arange(len(train_loaders))

    if args.alg == 'fedavg':   
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p,len(train_loaders), args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        for round in tqdm(range(args.comm_round)):
            logger.info("in comm round:" + str(round))

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in arr:
                        nets[idx].load_state_dict(global_para)
            else:
                nets[idx].load_state_dict(global_para)

            local_train_net(args, train_loaders, nets,  device=device)
     
            # # update global model
            total_data_points = sum([len(dl.dataset) for dl in train_loaders])
            fed_avg_freqs = [len(dl.dataset) / total_data_points for dl in train_loaders]


            for idx in range(len(train_loaders)):
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

            local_train_loss =[]
            for i in range(len(train_loaders)):
                _, l_loss= compute_test(global_model, train_loaders[i], device=device)
                local_train_loss.append(l_loss)

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(len(local_train_loss))])
            uni_loss =  np.sum(local_train_loss) / len(local_train_loss)

            weight_trainerr.append(weight_loss)
            uni_trainerr.append(uni_loss)


            print('test_acc',test_acc)
    elif args.alg == 'fedprox':
        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p,len(train_loaders), args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in tqdm(range(args.comm_round)):
            logger.info("in comm round:" + str(round))

          
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in arr:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in arr:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(args, train_loaders, nets, global_model, device=device)
            global_model.to('cpu')

          

            # # update global model
            total_data_points = sum([len(dl.dataset) for dl in train_loaders])
            fed_avg_freqs = [len(dl.dataset) / total_data_points for dl in train_loaders]


            for idx in range(len(train_loaders)):
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

            local_train_loss =[]
            for i in range(len(train_loaders)):
                _, l_loss= compute_test(global_model, train_loaders[i], device=device)
                local_train_loss.append(l_loss)

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(len(local_train_loss))])
            uni_loss =  np.sum(local_train_loss) / len(local_train_loss)

            weight_trainerr.append(weight_loss)
            uni_trainerr.append(uni_loss)
    
            print('test_acc',test_acc)

    elif args.alg =='fednova':

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p,len(train_loaders), args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]


        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(len(train_loaders))]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(len(train_loaders)):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0
            

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in tqdm(range(args.comm_round)):
            logger.info("in comm round:" + str(round))

          
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in arr:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in arr:
                    nets[idx].load_state_dict(global_para)
# args, train_loaders, nets, global_model,
            
            _, a_list, d_list = local_train_net_fednova(args, train_loaders, nets, global_model, device=device)
            # global_model.to('cpu')

            total_data_points = sum([len(dl.dataset) for dl in train_loaders])
            fed_avg_freqs = [len(dl.dataset) / total_data_points for dl in train_loaders]

            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for idx in range(len(train_loaders)):
                d_para = d_list[idx]
                for key in d_para:
                    #if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    #else:
                    d_total_round[key] += d_para[key] * fed_avg_freqs[idx]

                
             # update global model
            coeff = 0.0
            for idx in range(len(train_loaders)):
                coeff = coeff + a_list[idx] * fed_avg_freqs[idx]

            updated_model = global_model.state_dict()
            for key in updated_model:
                #print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)


            global_model.to(device)
         
            test_acc, test_loss = compute_test(global_model, test_loader, device=device)

            global_test_accuracy.append(test_acc)
            global_test_loss.append(test_loss)

            local_train_loss =[]
            for i in range(len(train_loaders)):
                _, l_loss= compute_test(global_model, train_loaders[i], device=device)
                local_train_loss.append(l_loss)

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(len(local_train_loss))])
            uni_loss =  np.sum(local_train_loss) / len(local_train_loss)

            weight_trainerr.append(weight_loss)
            uni_trainerr.append(uni_loss)
    
            print('test_acc',test_acc)
    elif args.alg =='scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p,len(train_loaders), args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, len(train_loaders), args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()

        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in tqdm(range(args.comm_round)):
            logger.info("in comm round:" + str(round))

          
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in arr:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in arr:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(args, train_loaders, nets, global_model, c_nets, c_global, device=device)
    

            # # update global model
            total_data_points = sum([len(dl.dataset) for dl in train_loaders])
            fed_avg_freqs = [len(dl.dataset) / total_data_points for dl in train_loaders]


            for idx in range(len(train_loaders)):
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

            local_train_loss =[]
            for i in range(len(train_loaders)):
                _, l_loss= compute_test(global_model, train_loaders[i], device=device)
                local_train_loss.append(l_loss)

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(len(local_train_loss))])
            uni_loss =  np.sum(local_train_loss) / len(local_train_loss)

            weight_trainerr.append(weight_loss)
            uni_trainerr.append(uni_loss)
    
            print('test_acc',test_acc)

    return global_test_loss, global_test_accuracy, weight_trainerr, uni_trainerr



def process_data(data_loader):
    
    net_test = PreActResNet18()
    net_test = net_test.to(device)
    net_test.load_state_dict(torch.load('checkpoint/preact_resnet18.pth', map_location=str('cuda:'+str(args.cnum))))
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


def get_ot_dist_triangle(args,local_train_loaders,val_loader):
    """"
    k : supporting size of the global gamma 
    t_val : could be any value between (0,1)
    """

    k = args.gamma_size
    t_val = args.t_val 
    metric = args.metric

    aug_train_data = []
    for local_dl in local_train_loaders:
        aug_train_data.append(process_data(local_dl))
    
    aug_val_data = process_data(val_loader)
    
    dim = aug_train_data[0].dim[1]
    global_gamma = np.random.randn(k, dim)
    interp_mea = InterpMeas(metric= metric, t_val=t_val)
    
    train_IntMea = [] 
    for local_data in aug_train_data:
        train_IntMea.append( interp_mea.fit(local_data, global_gamma) ) 
    
    train_IntMea  = np.vstack(train_IntMea)
    val_IntMea  =  interp_mea.fit(aug_val_data, global_gamma)
    
    cost = cal_distance(train_IntMea,val_IntMea,metric)
    
    return cost 

def get_ot_dist(train_loader, test_loader, n=5000,device='cpu'):
    #Todo:change the centralized calculation to decentrlized calculation
    
    net_test = PreActResNet18()
    net_test = net_test.to(device)
    # net_test.load_state_dict(torch.load('checkpoint/preact_resnet18.pth', map_location=str('cuda:'+str(args.cnum))))

    net_test.load_state_dict(torch.load('checkpoint/preact_resnet18.pth', map_location=str('cuda:0')))
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



# import pickle
# with open('./data/datafile.data', 'rb') as f:
#     train_features, train_labels,test_features,test_labels = pickle.load(f)

# with open('./data/balance_test_1000.data', 'rb') as f:
#     test_features,test_labels = pickle.load(f)

# train_features = train_features/255
# test_features = test_features/255
# # test_features = test_features.reshape(-1,32,32,3)
# # train_features = train_features.reshape(-1,32,32,3)
# # 分割数据
# client_data, class_distribution = split_data_to_clients(train_features, train_labels)


# # 生成比例
# proportions = generate_proportions()

# # 打印前10组查看
# print("Generated proportions (first 10 samples):")
# for i, p in enumerate(proportions[:10]):
#     print(f"Group {i+1}: {[round(x, 4) for x in p]}, Sum: {round(sum(p), 4)}")

# # 验证总数
# print(f"\nTotal groups generated: {len(proportions)}")
# print("Verification:")
# print(f"All sums equal to 1: {all(abs(sum(p)-1) < 1e-10 for p in proportions)}")

# # 统计不同客户端数量的分布
# client_counts = []
# for p in proportions:
#     count = sum(1 for x in p if x > 0)
#     client_counts.append(count)

# from collections import Counter
# print("\nClient count distribution:")
# print(Counter(client_counts))

def main(args, data_dict,device):
    print(f'algorithm {args.alg} budget {args.budget}')
    breaks = 10
    reps = 1

        
    train_features = data_dict['train_features']
    train_labels = data_dict['train_labels']
    test_features = data_dict['test_features']
    test_labels = data_dict['test_labels']
    train_label_idx = data_dict['train_label_index']
    test_label_idx = data_dict['test_label_index']

    n = args.budget
    batch_size = args.batch_size

    # make test dataloader
    test_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(test_features).permute(0,3,1,2), torch.LongTensor(test_labels)), 
                                                    batch_size=batch_size, 
                                                    shuffle=False)

   
    if args.n_parties==5:
        client_data, class_distribution = split_data_to_clients(train_features, train_labels)
        proportions = generate_proportions()
    

    total_q_ls = [[0,0,1],
                [0,1,0],
                [0.0,0.5,0.5],
                [0.8,0.1,0.1],
                [0.3,0.3,0.3],
                    ]
    qstrainerrlog_weight = []
    qstrainerrlog_uni =[]
    qstesterrlog = []
    qsotlog = []
    qsaccs = []
    pro_idx = 0

    for n in [4000,6000,8000,10000,12000]:
        # trainerrlog = []
        trainerrlog_weight = []
        trainerrlog_uni = []
        testerrlog = []
        otlog = []
        accs = []
        for q_ls in total_q_ls:
            
            
        
            start_t = time.time()
            q1 = q_ls[0]
            q2 = q_ls[1]
            q3 = 1-q1-q2
            if q3<0:
                break
            
            print(f'######### p1 {q1},p2 {q2},p3 {q3} #########')

            if args.n_parties==3:
                train_x_1,train_x_2, train_x_3, train_y_1,train_y_2, train_y_3 = dataset_q(q1, q2, n, train_features, train_labels,train_label_idx)
                train_combine_x = np.concatenate([train_x_1, train_x_2, train_x_3])
                train_combine_y = np.concatenate([train_y_1, train_y_2, train_y_3])
                train_x_ls = [train_x_1,train_x_2,train_x_3]
                train_y_ls = [train_y_1,train_y_2,train_y_3]

                train_x_ls = [item for item in train_x_ls if len(item)>0 ]
                train_y_ls = [item for item in train_y_ls if len(item)>0 ]
            

            if args.n_parties==5:
                print('proportions',proportions[pro_idx])

                data_dict = sample_data_from_clients(proportions[pro_idx], client_data, n)
                pro_idx+=0 
                train_x_ls = []
                train_y_ls = []
            
                for i in range(len(data_dict)):
                    train_x_ls.append(data_dict[i]['features'])
                    train_y_ls.append(data_dict[i]['labels'])
                train_x_ls = [item for item in train_x_ls if len(item)>0 ]
                train_y_ls = [item for item in train_y_ls if len(item)>0 ]

                train_combine_x = np.vstack(train_x_ls)
                train_combine_y = np.hstack(train_y_ls)

            local_train_loaders = []
            # make train dataloader
            for i in range(len(train_x_ls)):
                local_train_loaders.append(torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_x_ls[i]).permute(0,3,1,2), 
                                                    torch.LongTensor(train_y_ls[i])), 
                                                    batch_size=batch_size, 
                                                    shuffle=True))
            
            train_loader = torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_combine_x).permute(0,3,1,2), 
                                                torch.LongTensor(train_combine_y)), 
                                                batch_size=batch_size, 
                                                shuffle=True)
            for rep in range(reps):
                # get OT dist
                
                # ot_dist_combine = get_ot_dist_triangle(args,local_train_loaders,test_loader)
                ot_dist_combine = get_ot_dist(train_loader, test_loader, n=n,device=device)
                test_loss, test_acc, trainloss_weight, trainloss_uni = get_fl_model_log_error(local_train_loaders, test_loader, args)

                print('ot_dist_combine',ot_dist_combine)
                trainerrlog_weight.append(trainloss_weight)
                trainerrlog_uni.append(trainloss_uni)
                testerrlog.append(test_loss) 
                accs.append(test_acc)
                otlog.append(ot_dist_combine)
                
        
        qstrainerrlog_weight.append(trainerrlog_weight)
        qstrainerrlog_uni.append(trainerrlog_uni)
        qstesterrlog.append(testerrlog)
        qsotlog.append(otlog)
        qsaccs.append(accs)

    args_dict = vars(args)
    
    results = {
        'args': args_dict,
        'train_err_log_weight': qstrainerrlog_weight,
        'train_err_log_uni': qstrainerrlog_uni,
        'test_err_log': qstesterrlog,
        'ot_log': qsotlog,
        'accuracies': qsaccs
    }

    # if args.alg=='fedavg':
    #     pickle.dump(results, open(f'results/cif10_{args.n_parties}sources/{args.alg}_{args.budget}_{args.comm_round}_{args.epochs}.res', 'wb'))
    # elif args.alg=='fedprox':
    #     pickle.dump(results, open(f'results/cif10_{args.n_parties}sources/{args.alg}_{args.budget}_{args.comm_round}_{args.epochs}_{args.mu}.res', 'wb'))
    # else:
    #     pickle.dump(results, open(f'results/cif10_{args.n_parties}sources/{args.alg}_{args.budget}_{args.comm_round}_{args.epochs}.res', 'wb'))

    pickle.dump(results, open(f'results/cif10_{args.n_parties}sources/{args.alg}_{args.comm_round}_{args.epochs}_trial_runs.res', 'wb'))
    

if __name__ == "__main__":


    # todo: federated fintuning :  CLUES ( NeurIPS 2024) / https://arxiv.org/pdf/2401.06432
    args = get_args()
    # mkdirs(args.logdir)
    # mkdirs(args.modeldir)
    # if args.log_file_name is None:
    #     argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    # else:
    #     argument_path=args.log_file_name+'.json'

    

    print(f"procs cnum {args.cnum}")
        
    print("end")


    cuda_num = args.cnum
    import torch
    print(torch.__version__)
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda_num)
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    # torch.cuda.set_device(cuda_num)
    # print("Cuda device: ", torch.cuda.current_device())
    # print("cude devices: ", torch.cuda.device_count())
    device = 'cuda:' + str(cuda_num) if torch.cuda.is_available() else 'cpu'


    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    # data_all = pickle.load( open('data/cifar10.data', 'rb') )
    # train_features, train_labels, test_features, test_labels  = data_all


    import pickle



    with open('./data/datafile.data', 'rb') as f:
        train_features, train_labels,test_features,test_labels = pickle.load(f)

    with open('./data/balance_test_100.data', 'rb') as f:
        test_features,test_labels = pickle.load(f)
    
    test_features = test_features.reshape(-1,32,32,3)
    train_features = train_features.reshape(-1,32,32,3)

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

    main(args,data_dict,device)