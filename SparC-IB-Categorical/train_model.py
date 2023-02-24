from convexIB_cat import ConvexIB
from utils import get_data
from utils import get_args
import torch
import os
import numpy as np
import gc

gc.collect()

torch.cuda.empty_cache()
torch.set_num_threads(16)

# Obtain the arguments
args = get_args()

# Obtain the data
dataset_name = args.dataset
if dataset_name == 'mnist':
    trainset, validationset = get_data(dataset_name)
    n_x = 784
    n_y = 10
    network_type = 'mlp_mnist'
    maxIXY = np.log2(10)
    problem_type = 'classification'
    TEXT = None
    deterministic = True
elif dataset_name == 'CIFAR10':
    trainset, validationset = get_data(dataset_name)
    n_x = (3,32,32)
    n_y = 10
    network_type = 'mlp_CIFAR10'
    maxIXY = np.log2(n_y)
    problem_type = 'classification'
    TEXT = None
    deterministic = True


# Create the base folders
args.logs_dir = os.path.join(args.logs_dir,dataset_name) + '/'
args.figs_dir = os.path.join(args.figs_dir,dataset_name) + '/'
args.models_dir = os.path.join(args.models_dir,dataset_name) + '/'
os.makedirs(args.logs_dir) if not os.path.exists(args.logs_dir) else None
os.makedirs(args.figs_dir) if not os.path.exists(args.figs_dir) else None
os.makedirs(args.models_dir) if not os.path.exists(args.models_dir) else None

# Create specific folders for the function chosen
if args.method == 'variational_IB': 
    args.logs_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-')  + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_cat_train_8_no_dim_encoder" + '/'
    args.figs_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_cat_train_8_no_dim_encoder" + '/'
    args.models_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_cat_train_8_no_dim_encoder" + '/'
else:
    args.logs_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-')  + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_" + args.method + "_train_8" + '/'
    args.figs_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_" + args.method +"_train_8" + '/'
    args.models_dir += args.u_func_name + '_' + str(round(args.hyperparameter,2)).replace('.', '-') + "_repl_" + str(args.repl_n) + "_a_"+ str(args.a)+"_b_"+str(args.b) + "_" + args.method +"_train_8" + '/'
    
os.makedirs(args.logs_dir) if not os.path.exists(args.logs_dir) else None
os.makedirs(args.models_dir) if not os.path.exists(args.models_dir) else None

# Train the network
args.beta = np.linspace(0,1,50)[int(args.beta)] 
convex_IB = ConvexIB(n_x = n_x, n_y = n_y, problem_type = problem_type, network_type = network_type, K = args.K, beta = args.beta, dim_pen = args.dim_pen, a= args.a,b=args.b, logvar_t = args.logvar_t, logvar_kde = args.logvar_kde, train_logvar_t = args.train_logvar_t, u_func_name = args.u_func_name, hyperparameter = args.hyperparameter,method=args.method,dataset_name=dataset_name,repl_n = args.repl_n)
convex_IB.fit(trainset, validationset, n_epochs = args.n_epochs, learning_rate = args.learning_rate, learning_rate_drop = args.learning_rate_drop, learning_rate_steps = args.learning_rate_steps, sgd_batch_size = args.sgd_batch_size, mi_batch_size = args.mi_batch_size, same_batch = args.same_batch, eval_rate = args.eval_rate, optimizer_name = args.optimizer_name, verbose = args.verbose, visualization = args.visualize, logs_dir = args.logs_dir, figs_dir = args.figs_dir, models_dir=args.models_dir)

# Save the network
name_base = "K-" + str(args.K) + "-B-" + str(round(args.beta,3)).replace('.', '-') \
    + "_dim_pen_" + str(args.dim_pen).replace('.', '-') +  "-Tr-" + str(bool(args.train_logvar_t)) + '-'
torch.save(convex_IB.state_dict(), args.models_dir + name_base + 'model')
