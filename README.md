# Federated Data Selection with Wasserstein Distance


## 1. FL training 


```bash
python main.py --cnum 0 --n 3 \
               --dataset cifar10 \
               --batch_size 64 \
               --lr 0.1 \
               --comm_round 80 \
               --epochs 10 \
               --alg fedavg
```

Description: 
- `--cnum`: cuda num 
- `--n`: number of clients 
- `--lr`: learning rate 
- `--epochs`: local epochs 
- `--comm_round`: global iterations
- `--alg`: fedavg/fedprox

## 2. Projektor 

to be updated 