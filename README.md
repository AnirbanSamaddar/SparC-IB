# SparC-IB

Please follow the below code blocks to run the models used in the experiments section. Note that, the below codes are to run the models on MNIST dataset. Please pass ```--dataset CIFAR10``` and the correct hyperparameters from the Appendix to run the models on CIFAR-10.

To run ```SparC-IB``` with compound strategy please use the below code.

```
cd ./SparC-IB-Compound
python3 train_model.py --beta 1 --K 100 --a 2.0 --b 2.0 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'MNIST' --method 'variational_IB'
```

To run ```SparC-IB``` with categorical strategy please use the below code.

```
cd ./SparC-IB-Categorical
python3 train_model.py --beta 1 --K 100 --a 2.0 --b 2.0 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'MNIST'
```

To run ```Drop-VIB``` model please use the below code.

```
cd ./SparC-IB-Compound
python3 train_model.py --beta 1 --K 100 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'MNIST' --method 'drop_VIB'
```

To run ```Intel-VIB``` model please use the below code.

```
cd ./SparC-IB-Compound
python3 train_model.py --beta 1 --K 100 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'MNIST' --method 'intel_VIB'
```

To run ```Fixed-K-VIB``` model please use the below code.

```
cd ./Fixed-K
python3 train_model.py --beta 1 --K 32 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'MNIST' --method 'variational_IB'
```
Note that, the above code will run the ```Fixed K:32``` model. To run for different latent dimension change the ```--K``` argument in the above code.

Upon completion of all the above runs, the results will be stored in ```./results``` directory. 

