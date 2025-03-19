# Federated Data Selection with Wasserstein Distance


## Stage 1: FL training 

### Fedprox

```bash
python main.py --cnum 0 --n 3 \
               --dataset cifar10 \
               --model simple-cnn \
               --alg fedprox \
               --mu 0.1 \
               --batch_size 64 \
               --lr 0.01 \
               --comm_round 80 \
               --epochs 10 \
               
```

Description: 
- `--cnum`: cuda num 
- `--n`: number of clients 
- `--lr`: learning rate 
- `--epochs`: local epochs 
- `--comm_round`: global iterations
- `--alg`: fedavg/fedprox
- `--mu`: penalty term in fedprox

## Stage 2: Projektor

to be updated 