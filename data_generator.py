import numpy as np 
import torch

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
