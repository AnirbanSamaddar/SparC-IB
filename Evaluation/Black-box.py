import torchvision
import torch
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from functions import brier_score
sys.path.insert(0, '../SparC-IB-Compound')
from utils import get_data,get_args
from convexIB import ConvexIB as Data_Power_VIB_comp
sys.path.remove('../SparC-IB-Compound')   
sys.path.insert(0, '../SparC-IB-Categorical')
from convexIB_cat import ConvexIB as Data_Power_VIB_cat
sys.path.remove('../SparC-IB-Categorical')
sys.path.insert(0, '../Fixed-K')
from convexIB_fixed import ConvexIB as Vanilla_Power_VIB
from torchattacks import PGD

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

model_base_dir = '../results/models/'
figs_base_dir = '../results/figures/'
figs_new_dir = 'PGD/'


# Load the model
args = get_args()
if args.dataset == 'mnist':
    model_base_dir = model_base_dir + args.dataset + '/'
    figs_base_dir = figs_base_dir + args.dataset + '/' + figs_new_dir
elif args.dataset == 'CIFAR10':
    model_base_dir = model_base_dir + args.dataset + '/'
    figs_base_dir = figs_base_dir + args.dataset + '/' + figs_new_dir

models_list = ['pow_1-0_repl_' + str(args.repl_n) +'_a_1.0_b_3.0_cat_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_2.0_b_2.0_cat_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_1.0_b_3.0_compound_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_2.0_b_2.0_compound_train_8_no_dim_encoder/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_1.0_b_1.0_drop_VIB_train_8/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'_a_1.0_b_1.0_intel_VIB_train_8/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'/'
                ,'pow_1-0_repl_' + str(args.repl_n) +'/'
                ]
method = ['variational_IB','variational_IB','variational_IB','variational_IB','drop_VIB'
            ,'intel_VIB','variational_IB','variational_IB','variational_IB']

beta = [np.linspace(0,1,50)[1]]
dim_models = [100,100,100,100,100,100,6,32,128]
Kdist = [True,True,True,True,True,True,False,False,False]
corruption_deg = np.linspace(0,0.3,7,dtype=float) + 1e-7
accuracy_data = np.zeros((len(beta),len(models_list),len(corruption_deg)))
brier_loss = np.zeros((len(beta),len(models_list),len(corruption_deg)))
LL = np.zeros((len(beta),len(models_list),len(corruption_deg)))


for dim_id,dim in enumerate(dim_models):
    for i,d in enumerate(corruption_deg):
        print('Starting corruption ' + str(i) + ' model ' + str(dim_id),flush=True)
        for b_id,b in enumerate(beta):
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
                testset_x = testset_x.to(dev)
                testset_y = testset_y.to(dev)
                if Kdist[dim_id]:
                    if dim_id == 2 or dim_id == 3:
                        class Modified_model(Data_Power_VIB_comp):
                            def __init__(self):
                                super().__init__(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                network_type = network_type, K = dim, beta = b, dim_pen = args.dim_pen, a=2,b=2, \
                                logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                u_func_name = 'pow', hyperparameter = args.hyperparameter,method='variational_IB', \
                                dataset_name=args.dataset,repl_n = 1)
                            def forward(self,x):
                                mean_t,gamma,pi,sigma_t,_,_,_,_ = self.network.encode(x,random=False)
                                tmp = torch.nn.functional.gumbel_softmax(torch.log(pi.repeat(10,1,1)),tau=0.1,hard=True)
                                mask = tmp.cumsum(dim=2)
                                gamma = ((1 - mask) + tmp).to(dev)
                                mean_t_hat = (mean_t[None,:,:].repeat(10,1,1)*gamma).mean(dim=0)
                                logits = self.network.decode(mean_t_hat)
                                return logits
                        model = Modified_model()        
                        model_name = models_list[dim_id] + 'K-100-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'
                        model.load_state_dict(torch.load(model_base_dir + model_name))
                        model = model.eval().to(dev)
                        attack_Linf = PGD(model,eps = d,steps=10)
                        testset_x_Linf = attack_Linf(testset_x,testset_y)
                        with torch.no_grad():
                            ### Linf PGD
                            logits = model(testset_x_Linf)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,dim_id,i] = model.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,dim_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,dim_id,i] = model.get_ITY(logits,testset_y)
                    elif dim_id == 4:
                        class Modified_model(Data_Power_VIB_comp):
                            def __init__(self):
                                super().__init__(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                network_type = network_type, K = dim, beta = b, dim_pen = args.dim_pen, a=2,b=2, \
                                logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                u_func_name = 'pow', hyperparameter = args.hyperparameter,method=method[dim_id], \
                                dataset_name=args.dataset,repl_n = 1)
                            def forward(self,x):
                                t,gamma,pi,sigma_t,_,_,mean_t,_ = self.network.encode(x,random=False)
                                cat_pi = torch.cat(((1-pi)[:,:,None],pi[:,:,None]),dim = -1)
                                tmp = torch.nn.functional.gumbel_softmax(torch.log(cat_pi.repeat(10,mean_t.shape[0],1,1)),tau=0.1,hard=False)
                                tmp1 = tmp[:,:,:,0]
                                gamma = tmp1*(self.K/(self.K - pi.sum(dim=1)[None,None,:]))
                                mean_t_hat = (mean_t[None,:,:].repeat(10,1,1)*gamma).mean(dim=0)
                                logits = self.network.decode(mean_t_hat)
                                return logits
                        model = Modified_model()        
                        model_name = models_list[dim_id] + 'K-100-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'
                        model.load_state_dict(torch.load(model_base_dir + model_name))
                        model = model.eval().to(dev)
                        attack_Linf = PGD(model,eps = d,steps=10)
                        testset_x_Linf = attack_Linf(testset_x,testset_y)
                        with torch.no_grad():
                            ### Linf PGD
                            logits = model(testset_x_Linf)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,dim_id,i] = model.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,dim_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,dim_id,i] = model.get_ITY(logits,testset_y)
                    elif dim_id == 5:
                        class Modified_model(Data_Power_VIB_comp):
                            def __init__(self):
                                super().__init__(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                network_type = network_type, K = dim, beta = b, dim_pen = args.dim_pen, a=2,b=2, \
                                logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                u_func_name = 'pow', hyperparameter = args.hyperparameter,method=method[dim_id], \
                                dataset_name=args.dataset,repl_n = 1)
                            def forward(self,x):
                                t,gamma,pi,sigma_t,_,_,mean_t,_ = self.network.encode(x,random=False)
                                pi = self.network.prob_encoder(mean_t)
                                mean_t_hat = (mean_t*pi)
                                logits = self.network.decode(mean_t_hat)
                                return logits
                        model = Modified_model()        
                        model_name = models_list[dim_id] + 'K-100-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'
                        model.load_state_dict(torch.load(model_base_dir + model_name))
                        model = model.eval().to(dev)
                        attack_Linf = PGD(model,eps = d,steps=10)
                        testset_x_Linf = attack_Linf(testset_x,testset_y)
                        with torch.no_grad():
                            ### Linf PGD
                            logits = model(testset_x_Linf)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,dim_id,i] = model.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,dim_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,dim_id,i] = model.get_ITY(logits,testset_y)
                    else:
                        class Modified_model(Data_Power_VIB_cat):
                            def __init__(self):
                                super().__init__(n_x = n_x, n_y = n_y, problem_type = problem_type, \
                                network_type = network_type, K = dim, beta = b, dim_pen = args.dim_pen, a=2,b=2, \
                                logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                                u_func_name = 'pow', hyperparameter = args.hyperparameter,method='variational_IB', \
                                dataset_name=args.dataset,repl_n = 1)
                            def forward(self,x):
                                mean_t,gamma,pi,sigma_t = self.network.encode(x,random=False)
                                tmp = torch.nn.functional.gumbel_softmax(torch.log(pi.repeat(10,1,1)),tau=0.1,hard=True)
                                mask = tmp.cumsum(dim=2)
                                gamma = ((1 - mask) + tmp).to(dev)
                                mean_t_hat = (mean_t[None,:,:].repeat(10,1,1)*gamma).mean(dim=0)
                                logits = self.network.decode(mean_t_hat)
                                return logits
                        model = Modified_model()        
                        model_name = models_list[dim_id] +  'K-100-B-' + str(round(b,3)).replace('.', '-') + '_dim_pen_1-0-Tr-False-model'
                        model.load_state_dict(torch.load(model_base_dir + model_name))
                        model = model.eval().to(dev)
                        attack_Linf = PGD(model,eps = d,steps=10)
                        testset_x_Linf = attack_Linf(testset_x,testset_y)
                        with torch.no_grad():
                            ### Linf PGD
                            logits = model(testset_x_Linf)
                            logits = logits[None,:,:]
                            accuracy_data[b_id,dim_id,i] = model.evaluate(logits,testset_y).cpu().item()
                            brier_loss[b_id,dim_id,i] = brier_score(logits,testset_y).cpu().item()
                            LL[b_id,dim_id,i] = model.get_ITY(logits,testset_y)
                else:
                    class Modified_model(Vanilla_Power_VIB):
                        def __init__(self):
                            super().__init__(n_x = 784, n_y = 10, problem_type = 'classification', \
                            network_type = network_type, K = dim, beta = b, \
                            logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, \
                            u_func_name = 'pow', hyperparameter = args.hyperparameter,method='variational_IB', \
                            dataset_name=args.dataset)
                        def forward(self,x):
                            t,sigma = self.network.encode(x,random=False)
                            mean_t_hat = t
                            logits = self.network.decode(mean_t_hat)
                            return logits
                    model = Modified_model()
                    model_name = models_list[dim_id] + 'K-' + str(dim) + '-B-' + str(round(b,3)).replace('.', '-') + '-Tr-False-model'
                    model.load_state_dict(torch.load(model_base_dir + model_name))
                    model = model.eval().to(dev)
                    attack_Linf = PGD(model,eps = d,steps=10)
                    testset_x_Linf = attack_Linf(testset_x,testset_y)
                    with torch.no_grad():
                        ### Linf PGD    
                        logits = model(testset_x_Linf)
                        logits = logits[None,:,:]
                        accuracy_data[b_id,dim_id,i] = model.evaluate(logits,testset_y).cpu().item()
                        brier_loss[b_id,dim_id,i] = brier_score(logits,testset_y).cpu().item()
                        LL[b_id,dim_id,i] = model.get_ITY(logits,testset_y)


os.makedirs(figs_base_dir) if not os.path.exists(figs_base_dir) else None
os.chdir(figs_base_dir)   # figure directory

np.save("Accuracy_data_PGD_10_"+str(args.repl_n),accuracy_data)
np.save("Brier_score_PGD_10_"+str(args.repl_n),brier_loss)
np.save("LL_PGD_10_"+str(args.repl_n),LL)

