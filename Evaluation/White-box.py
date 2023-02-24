import torchvision
import torch
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from functions import brier_score
sys.path.insert(0, './SparC-IB-Compound')
from utils import get_data,get_args
from convexIB_7_no_dim_encoder import ConvexIB as Data_Power_VIB_comp   
sys.path.insert(0, './SparC-IB-Categorical')
from convexIB_7_cat_no_dim_encoder import ConvexIB as Data_Power_VIB_cat
from convexIB_2_fixed import ConvexIB as Vanilla_Power_VIB

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

#path_src = '/mnt/home/samadda1/NSF-Project/Simulation-MNIST/src/Kdist_model_v2_compound'
model_base_dir = './results/models/'
figs_base_dir = './results/figures/'
figs_new_dir = 'whitebox/'


########## White box attack
def shot_noise(x, severity=5):
    c = np.linspace(2,0.01,10)[severity - 1]
    x = np.array(x)
    x = np.clip(np.random.poisson(x * c) / float(c), 0, 1)
    return x.astype(np.float32)


# Load the model
args = get_args()
if args.dataset == 'mnist':
    model_base_dir = model_base_dir + args.dataset + '/'
    figs_base_dir = figs_base_dir + args.dataset + '/' + figs_new_dir
elif args.dataset == 'CIFAR10':
    model_base_dir = model_base_dir + args.dataset + '/'
    figs_base_dir = figs_base_dir + args.dataset + '/' + figs_new_dir

corruption_deg = range(1,11,1)
models_list = ['pow_1-0_repl_' + str(args.repl_n) +'_a_1.0_b_3.0_cat_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_2.0_b_2.0_cat_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_1.0_b_3.0_compound_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_2.0_b_2.0_compound_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_2.0_b_2.0_drop_VIB_train_8/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_2.0_b_2.0_intel_VIB_train_8/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'/'
                ]
method = ['variational_IB','variational_IB','variational_IB','variational_IB','drop_VIB'
            ,'intel_VIB','variational_IB','variational_IB','variational_IB']

beta = [np.linspace(0,1,50)[4]]
dim_models = [100,100,100,100,100,100,6,32,128]
Kdist = [True,True,True,True,True,True,False,False,False]
accuracy_data = np.zeros((len(beta),len(models_list),len(corruption_deg)))
brier_loss = np.zeros((len(beta),len(models_list),len(corruption_deg)))
LL = np.zeros((len(beta),len(models_list),len(corruption_deg)))

for b_id,b in enumerate(beta):
    for model_id,model_name in enumerate(models_list):
        for i,d in enumerate(corruption_deg):
            print('Starting corruption ' + str(d) + ' model ' + str(model_id),flush=True)
            #os.chdir(path_src)     # src folder directory
            if args.dataset == 'mnist':
                _, validationset = get_data(args.dataset)
                n_x = 784
                n_y = 10
                network_type = 'mlp_mnist'
                problem_type = 'classification'
                validation_loader = torch.utils.data.DataLoader(validationset, \
                            batch_size=len(validationset),shuffle=False)
                _,testset = next(enumerate(validation_loader))
                testset_x,testset_y = testset
                testset_x = testset_x.cpu().numpy()
                testset_x = shot_noise(testset_x,severity=d)
                testset_x = torch.from_numpy(testset_x).to(dev)    
                testset_y = testset_y.to(dev)
                K = dim_models[model_id]
                if Kdist[model_id]:
                    if model_id == 2 or model_id == 3:
                        model_data_K = Data_Power_VIB_comp(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                    network_type = network_type, K = K, beta = (2-1)/49, dim_pen = args.dim_pen, a=2,b=2, \
                                    logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                    u_func_name = 'pow', hyperparameter = args.hyperparameter,method=method[model_id], \
                                    dataset_name=args.dataset,repl_n = 1)
                        model_data_K.load_state_dict(torch.load(model_base_dir + model_name + 'K-' + str(K) + '-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'))
                        model_data_K.eval()
                        with torch.no_grad():
                            t,gamma,pi,sigma_t,a_x,b_x,mean_t,pi_s = model_data_K.network.encode(testset_x,random=False)
                            tmp = torch.nn.functional.gumbel_softmax(torch.log(pi.repeat(10,1,1)),tau=0.1,hard=True)
                            mask = tmp.cumsum(dim=2)
                            gamma = ((1 - mask) + tmp).to(dev)
                            mean_t_hat = (mean_t[None,:,:].repeat(10,1,1)*gamma).mean(dim=0)
                            logits = model_data_K.network.decode(mean_t_hat)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,model_id,i] = model_data_K.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,model_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,model_id,i] = model_data_K.get_ITY(logits,testset_y)
                    elif model_id == 4:
                        model_data_K = Data_Power_VIB_comp(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                    network_type = network_type, K = K, beta = (2-1)/49, dim_pen = args.dim_pen, a=2,b=2, \
                                    logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                    u_func_name = 'pow', hyperparameter = args.hyperparameter,method=method[model_id], \
                                    dataset_name=args.dataset,repl_n = 1)
                        model_data_K.load_state_dict(torch.load(model_base_dir + model_name + 'K-' + str(K) + '-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'))
                        model_data_K.eval()
                        with torch.no_grad():
                            t,gamma,pi,sigma_t,a_x,b_x,mean_t,pi_s = model_data_K.network.encode(testset_x,random=False)
                            cat_pi = torch.cat(((1-pi)[:,:,None],pi[:,:,None]),dim = -1)
                            tmp = torch.nn.functional.gumbel_softmax(torch.log(cat_pi.repeat(10,mean_t.shape[0],1,1)),tau=0.1,hard=False)
                            tmp1 = tmp[:,:,:,0]
                            gamma = tmp1*(K/(K - pi.sum(dim=1)[None,None,:]))
                            mean_t_hat = (mean_t[None,:,:].repeat(10,1,1)*gamma).mean(dim=0)
                            logits = model_data_K.network.decode(mean_t_hat)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,model_id,i] = model_data_K.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,model_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,model_id,i] = model_data_K.get_ITY(logits,testset_y)
                    elif model_id == 5:
                        model_data_K = Data_Power_VIB_comp(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                    network_type = network_type, K = K, beta = (2-1)/49, dim_pen = args.dim_pen, a=2,b=2, \
                                    logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                    u_func_name = 'pow', hyperparameter = args.hyperparameter,method=method[model_id], \
                                    dataset_name=args.dataset,repl_n = 1)
                        model_data_K.load_state_dict(torch.load(model_base_dir + model_name + 'K-' + str(K) + '-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'))
                        model_data_K.eval()
                        with torch.no_grad():
                            t,gamma,pi,sigma_t,a_x,b_x,mean_t,pi_s = model_data_K.network.encode(testset_x,random=False)
                            pi = model_data_K.network.prob_encoder(mean_t)
                            mean_t_hat = (mean_t*pi)
                            logits = model_data_K.network.decode(mean_t_hat)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,model_id,i] = model_data_K.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,model_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,model_id,i] = model_data_K.get_ITY(logits,testset_y)
                    else:
                        model_data_K = Data_Power_VIB_cat(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                    network_type = network_type, K = K, beta = (2-1)/49, dim_pen = args.dim_pen, a=2,b=2, \
                                    logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                    u_func_name = 'pow', hyperparameter = args.hyperparameter,method='variational_IB', \
                                    dataset_name=args.dataset,repl_n = 1)
                        model_data_K.load_state_dict(torch.load(model_base_dir + model_name + 'K-' + str(K) + '-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'))
                        model_data_K.eval()
                        with torch.no_grad():
                            mean_t,gamma,pi,sigma_t = model_data_K.network.encode(testset_x,random=False)
                            tmp = torch.nn.functional.gumbel_softmax(torch.log(pi.repeat(10,1,1)),tau=0.1,hard=True)
                            mask = tmp.cumsum(dim=2)
                            gamma = ((1 - mask) + tmp).to(dev)
                            mean_t_hat = (mean_t[None,:,:].repeat(10,1,1)*gamma).mean(dim=0)
                            logits = model_data_K.network.decode(mean_t_hat)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,model_id,i] = model_data_K.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,model_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,model_id,i] = model_data_K.get_ITY(logits,testset_y)
                else:
                    model_K = Vanilla_Power_VIB(n_x = n_x, n_y = 10, problem_type = 'classification', \
                    network_type = network_type, K = K, beta = (2-1)/49, \
                    logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                    u_func_name = 'pow', hyperparameter = args.hyperparameter,method='variational_IB', \
                    dataset_name=args.dataset)
                    model_K.load_state_dict(torch.load(model_base_dir + model_name + 'K-' + str(K) + '-B-' + str(round(b,3)).replace('.', '-') + '-Tr-False-model'))
                    model_K.eval()
                    with torch.no_grad():
                        t,_ = model_K.network.encode(testset_x,random=False)
                        mean_t_hat = t
                        logits = model_K.network.decode(mean_t_hat)
                        logits = logits[None,:,:]
                        accuracy_data[b_id,model_id,i] = model_K.evaluate(logits,testset_y).cpu().item()
                        brier_loss[b_id,model_id,i] = brier_score(logits,testset_y).cpu().item()
                        LL[b_id,model_id,i] = model_K.get_ITY(logits,testset_y)

os.makedirs(figs_base_dir) if not os.path.exists(figs_base_dir) else None
os.chdir(figs_base_dir)   # figure directory

np.save("Accuracy_" + str(args.repl_n),accuracy_data)
np.save("Brier_score_" + str(args.repl_n),brier_loss)
np.save("ECE_" + str(args.repl_n),ECE)
np.save("LL_" + str(args.repl_n),LL)


