# Implementation of SparC-IB

This repository contains scripts to implement ```SparC-IB``` from the paper [Sparsity-Inducing Categorical Prior Improves Robustness of the Information Bottleneck](https://arxiv.org/abs/2203.02592). The codes in this repository is heavily based on the [convexIB](https://github.com/burklight/convex-IB-Lagrangian-PyTorch) implementation.

Please follow the below code blocks to run the models used in the experiments section. Note that, the below codes run the models on MNIST dataset. Please pass ```--dataset 'CIFAR10'``` and the correct hyperparameters from the Appendix to run the models on CIFAR-10.

To run ```SparC-IB``` with compound strategy please use the below code.

```
cd ./SparC-IB-Compound
python3 train_model.py --repl_n 1 --beta 1 --K 100 --a 2.0 --b 2.0 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'mnist' --method 'variational_IB'
```

To run ```SparC-IB``` with categorical strategy please use the below code.

```
cd ./SparC-IB-Categorical
python3 train_model.py --repl_n 1 --beta 1 --K 100 --a 2.0 --b 2.0 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'mnist'
```
Note that, the above codes run ```SparC-IB``` with the hyperparameters ```(a,b)=(2,2)``` for both compound and categorical strategy. For other values, pass different numbers (as floats) in the arguments ```--a``` and ```--b```. 

To run ```Drop-VIB``` model please use the below code.

```
cd ./SparC-IB-Compound
python3 train_model.py --repl_n 1 --beta 1 --K 100 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'mnist' --method 'drop_VIB'
```

To run ```Intel-VIB``` model please use the below code.

```
cd ./SparC-IB-Compound
python3 train_model.py --repl_n 1 --beta 1 --K 100 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'mnist' --method 'intel_VIB'
```

To run ```Fixed-K-VIB``` model please use the below code.

```
cd ./Fixed-K
python3 train_model.py --repl_n 1 --beta 1 --K 32 --verbose --u_func_name 'pow' --eval_rate 1 --same_batch --dataset 'mnist' --method 'variational_IB'
```
Note that, the above code will run the ```Fixed K:32``` model. To run for different latent dimensions please change the ```--K``` argument in the above code. Furthermore, all the above code is for one seed used in the experiments in the paper. To run the models for different seeds change the ```--repl_n``` argument. In the paper, we have used ```--repl_n 1```, ```2```, and ```3``` as three seeds for all the experiments.

Upon completion of all the above runs, the results will be stored in ```./results``` directory.

After running the models, to get the out-of-distribution results from the paper, please run the following lines of code.

**Black box attacks**
```
cd ./Evaluation
python3 Black-box.py --dataset 'mnist'
```
**White box attacks**
```
cd ./Evaluation
python3 White-box.py --dataset 'mnist'
```
**Rotation on MNIST**
```
cd ./Evaluation
python3 Rotation_mnist.py --dataset 'mnist'
```
Running the above codes will save the output in the folder ```./results/figures/mnist/``` under the sub-folders ```PGD``, ```Whitebox```, and ```Rotation``` respectively.
