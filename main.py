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

# import torchshow as ts

from torchvision.utils import make_grid
from torch.utils.data import random_split, Dataset, TensorDataset, DataLoader

import argparse

from flalg.experiments import *

from wd_utils import *
from data_generator import * 
from encoder_model import * 

def get_args():
    
    parser = argparse.ArgumentParser()

    # general 
    parser.add_argument('--cnum', type=int, required=True,
                    help='number of cuda in the server')


    # interpolating measures 
    parser.add_argument('--gamma_size', type=int, default=100, help='supporting size of gamma')
    parser.add_argument('--t_val', type=float, default=0.5,help='value')
    parser.add_argument('--metric', type=str, default='sqeuclidean', help='metric for wasserstein distance')
    
    
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
    arr = np.arange(args.n_parties)

    if args.alg == 'fedavg':
        
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in arr:
                        nets[idx].load_state_dict(global_para)
            else:
                nets[idx].load_state_dict(global_para)

            nets, local_train_loss = local_train_net(nets, args, train_loaders, test_dl = test_loader, device=device)
     
            # # update global model
            total_data_points = sum([len(dl.dataset) for dl in train_loaders])
            fed_avg_freqs = [len(dl.dataset) / total_data_points for dl in train_loaders]


            for idx in range(args.n_parties):
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

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(args.n_parties)])
            uni_loss =  np.sum(local_train_loss) / len(local_train_loss)

            weight_trainerr.append(weight_loss)
            uni_trainerr.append(uni_loss)


    elif args.alg == 'fedprox':

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

          
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in arr:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in arr:
                    nets[idx].load_state_dict(global_para)

            nets, local_train_loss = local_train_net_fedprox(nets, global_model, args, device=device)
            global_model.to('cpu')

          

            # # update global model
            total_data_points = sum([len(dl.dataset) for dl in train_loaders])
            fed_avg_freqs = [len(dl.dataset) / total_data_points for dl in train_loaders]


            for idx in range(args.n_parties):
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

            weight_loss = np.sum([ local_train_loss[i]*fed_avg_freqs[i] for i in range(args.n_parties)])
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



def main(args, data_dict):

    breaks = 10
    reps = 2

        
    train_features = data_dict['train_features']
    train_labels = data_dict['train_labels']
    test_features = data_dict['test_features']
    test_labels = data_dict['test_labels']
    train_label_idx = data_dict['train_label_index']
    test_label_idx = data_dict['test_label_index']

    n = args.n_parties
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
            for i in range(args.n_parties):
                local_train_loaders.append(torch.utils.data.DataLoader(dataset=TensorDataset(torch.Tensor(train_x_ls[i]).permute(0,3,1,2), 
                                                    torch.LongTensor(train_y_ls[i])), 
                                                    batch_size=batch_size, 
                                                    shuffle=True))
            for rep in range(reps):
                # get OT dist
               
                ot_dist_combine = get_ot_dist_triangle(args,local_train_loaders,test_loader)
                # ot_dist_combine = get_ot_dist(train_loader, test_loader, n=n)
                test_loss, test_acc, train_loss = get_fl_model_log_error(local_train_loaders, test_loader, args)

                trainerrlog.append(train_loss)
                testerrlog.append(test_loss) 
                accs.append(test_acc)
                otlog.append(ot_dist_combine)
                
        
        qstrainerrlog.append(trainerrlog)
        qstesterrlog.append(testerrlog)
        qsotlog.append(otlog)
        qsaccs.append(accs)

    args_dict = vars(args)
    
    results = {
        'args': args_dict,
        'train_err_log': qstrainerrlog,
        'test_err_log': qstesterrlog,
        'ot_log': qsotlog,
        'accuracies': qsaccs
    }
    pickle.dump(results, open(f'cif10_3sources/{args.alg}_unbalanced_{args.n_parties}.res', 'wb'))
       
    # pickle.dump([qstrainerrlog,qstesterrlog,qsotlog,qsaccs], open(f'cif10_3sources/{args.alg}_unbalanced_{n}.res', 'wb' ))


  

if __name__ == "__main__":

    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'

    

    print(f"procs cnum {args.cnum}")

    print(f"data cnum {args.n_parties}")
        
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