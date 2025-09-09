
python main.py --cnum=1 --budget=6000 --dataset=cifar10 --model=simple-cnn --alg=fedavg --comm_round=80 --epochs=10
python main.py --cnum=1 --budget=6000 --dataset=cifar10 --model=simple-cnn --alg=fednova --comm_round=80 --epochs=10
python main.py --cnum=1 --budget=6000 --dataset=cifar10 --model=simple-cnn --alg=scaffold --comm_round=80 --epochs=10
python main.py --cnum=1 --budget=6000 --dataset=cifar10 --model=simple-cnn --alg=fedprox --comm_round=80 --mu=0.3 --epochs=10