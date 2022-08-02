import torch
import math
import autograd
import scipy.optimize
import scipy.special as sc
from progressbar import progressbar
import numpy as np
import random
from network import nlIB_network
from common import SMALL, kl_divergence, kl_discrete


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class ConvexIB(torch.nn.Module):

    def __init__(self,n_x,n_y,problem_type,network_type,K,beta,dim_pen,a=1,b=1,logvar_t=-1.0,logvar_kde=-1.0,\
        train_logvar_t=False, u_func_name='pow', hyperparameter=1.0, TEXT=None, compression_level = 1.0, method = 'nonlinear_IB', dataset_name = 'mnist',repl_n=1):
        super(ConvexIB,self).__init__()
        
        # Set seed
        self.repl_n = repl_n
        torch.manual_seed(self.repl_n)

        self.HY = np.log(n_y) # in natts
        self.maxIXY = self.HY # in natts
        self.varY = 0 # to be updated with the training dataset
        self.IXT = 0 # to be updated
        self.ITY = 0 # to be
        self.n_x = n_x
        self.a = a
        self.b = b
        
        

        self.u_func_name = u_func_name
        self.compression_level = compression_level
        if self.u_func_name == 'pow':
            self.u_func = lambda r: r ** (1+hyperparameter)
        elif self.u_func_name == 'exp':
            self.u_func = lambda r: torch.exp(hyperparameter*r)
        elif self.u_func_name == 'shifted-exp':
            self.u_func = lambda r: torch.exp((r-compression_level)*hyperparameter)*hyperparameter
        else:
            self.u_func = lambda r: r

        def hist_laxis(data, n_bins, range_limits):
            # Setup bins and determine the bin location for each element for the bins
            R = range_limits
            N = data.shape[-1]
            bins = torch.linspace(R[0],R[1],n_bins+1).to(dev)
            data2D = data.reshape(-1,N)
            idx = torch.searchsorted(bins, data2D,right = True)-1

            # Some elements would be off limits, so get a mask for those
            bad_mask = (idx==-1) | (idx==n_bins)

            # We need to use bincount to get bin based counts. To have unique IDs for
            # each row and not get confused by the ones from other rows, we need to 
            # offset each row by a scale (using row length for this).
            scaled_idx = n_bins*(torch.arange(data2D.shape[0])[:,None] ).to(dev)+ idx

            # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
            limit = n_bins*data2D.shape[0]
            scaled_idx[bad_mask] = limit

            # Get the counts and reshape to multi-dim
            counts = torch.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
            counts = counts.reshape(data.shape[:-1] + (n_bins,)).type('torch.FloatTensor')
            return counts,(bins[1]-bins[0])

        def sparse_loss_relative_entropy(dim_rate):
            dim_rate = dim_rate / dim_rate.sum(dim=1)[:,None]
            mean_each_entropy=(- dim_rate*torch.log(dim_rate+1e-7)).sum(dim = 1)
            mean_prob = dim_rate.mean(dim=0)[None,:]
            mean_prob_entropy = (- mean_prob * torch.log(mean_prob + 1e-7)).sum(dim=1)
            return -mean_prob_entropy + mean_each_entropy.mean(dim=0) 
        

        self.hist_laxis = hist_laxis
        self.sparse_loss_relative_entropy = sparse_loss_relative_entropy
        self.method = method
        
        self.K = K
        self.beta = beta
        self.dim_pen = dim_pen
        self.network = nlIB_network(K,n_x,n_y,a,b,logvar_t,train_logvar_t,network_type,self.method,TEXT).to(dev)
        self.dataset_name = dataset_name
        self.problem_type = problem_type 
        if self.problem_type == 'classification':
            self.ce = torch.nn.CrossEntropyLoss()
        else:
            self.mse = torch.nn.MSELoss()


    def get_IXT(self,t,mean_t,sigma_t,gamma,a_x,b_x,pi=1,pi_s=1,pi_prior = 1,a=1,b=1,Test=True,datasetsize=50000):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t (Tensor) : deterministic transformation of the input
        '''

        if self.method == 'variational_IB':
            tmp = -0.5*(1+2*torch.log(sigma_t)-mean_t.pow(2)-sigma_t**2).cumsum(1)
            tmp = tmp*pi
            pi1 = pi.clone()
            pi1[pi1 == 0] = 1/1e7
            pi = pi1 
            tmp3 = (torch.log(pi)*pi) - (torch.log(pi_prior)*pi)
            self.IXT_1 = tmp.sum(1).mean().div(math.log(2)) 
            self.IXT_2 = tmp3.sum(1).mean().div(math.log(2))
            self.IXT = self.IXT_1.to(dev) + self.IXT_2.to(dev)
        elif self.method == 'drop_VIB':
            with torch.no_grad():
                n_bins = 32
                range_limits = torch.tensor([torch.min(mean_t).item(),torch.max(mean_t).item()]).to(dev)
                counts,width = self.hist_laxis(mean_t.transpose(0,1),n_bins,range_limits)
                width = width
                counts = counts.transpose(0,1).to(dev)
                density = counts/(mean_t.shape[0])#*(width+1e-7))
                bin_entropy_est = (- density * torch.log(density+1e-7)).sum(dim = 0)[None,:]# + density * torch.log(width+1e-7)).sum(dim = 0)[None,:]

            self.IXT = ((1-pi) * bin_entropy_est).sum(dim=1).div(math.log(2)) #((1-pi) * bin_entropy_est).mean(dim=1).div(math.log(2))
            self.IXT_1 = self.IXT
            self.IXT_2 = torch.tensor([0.0]).to(dev)
        elif self.method == 'intel_VIB':
            tmp = -0.5*(1+2*torch.log(sigma_t)-mean_t.pow(2)-sigma_t**2)
            t_mapped = t*pi
            self.IXT_2 = self.sparse_loss_relative_entropy(torch.abs(t_mapped)).div(math.log(2))     ### From line 340 of experiment.py but in the paper: sparse_loss is f(DS(t))
            self.IXT_1 = tmp.sum(1).mean().div(math.log(2)) 
            self.IXT = self.IXT_1 + self.IXT_2

        # NaNs and exploding gradients control
        with torch.no_grad():
            if self.u_func_name == 'shifted-exp':
                if self.IXT > self.compression_level:
                    self.IXT -= (self.IXT - self.compression_level - 0.01)
            if self.u_func(torch.Tensor([self.IXT])) == float('inf'):
                if self.u_func_name == 'exp':
                    self.IXT = torch.log(torch.Tensor([1e5]))
                else:
                    self.IXT = torch.Tensor([1e5])

        return self.IXT_1, self.IXT_2

    def get_ITY(self,logits_y,y):
        '''
        Obtains the mutual information between the bottleneck variable and the output.
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        if self.problem_type == 'classification':
            tmp = logits_y.clone()           
            tmp = tmp.reshape((logits_y.shape[0]*logits_y.shape[1],logits_y.shape[2]))
            y = y.repeat(logits_y.shape[0])            
            logits_y = tmp
            HY_given_T = self.ce(logits_y,y)
            self.ITY = (self.HY - HY_given_T) / np.log(2) # in bits

            return self.ITY
        else: 
            MSE = self.mse(logits_y.view(-1),y)
            ITY = 0.5 * torch.log(self.varY / MSE) / np.log(2) # in bits
            return ITY , (self.HY - MSE) / np.log(2) # in bits

    def evaluate(self,logits_y,y):
        '''
        Evauluates the performance of the model
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        with torch.no_grad():
            if self.problem_type == 'classification':
                tmp = logits_y.clone()
                tmp = tmp.reshape((logits_y.shape[0]*logits_y.shape[1],logits_y.shape[2]))
                y = y.repeat(logits_y.shape[0])                
                logits_y = tmp
                y_hat = y.eq(torch.max(logits_y,dim=1)[1])
                accuracy = torch.mean(y_hat.float())
                return accuracy
            else: 
                mse = self.mse(logits_y.view(-1),y) 
                return mse 

    def class_evaluate(self,logits_y,y,n_class):
        '''
        Evauluates the performance of the model
        Parameters:
        - logits_y (Tensor) : deterministic transformation of the bottleneck variable
        - y (Tensor) : labels of the data
        '''

        with torch.no_grad():
            tmp = logits_y.clone()
            tmp = tmp.reshape((logits_y.shape[0]*logits_y.shape[1],logits_y.shape[2]))
            y = y.repeat(logits_y.shape[0])
            logits_y = tmp
                
            a = torch.transpose(torch.nn.functional.one_hot(y,num_classes=n_class).squeeze(),0,1)
            b = torch.transpose(torch.nn.functional.one_hot(torch.max(logits_y,dim=1)[1],num_classes=n_class).squeeze(),0,1)
            class_acc = torch.sum((a*b).float(),dim=1)/torch.sum(a.float(),dim=1)
            
            return class_acc
            


    def fit(self,trainset,validationset,n_epochs=200,learning_rate=0.0001,\
        learning_rate_drop=0.6,learning_rate_steps=10, sgd_batch_size=128,mi_batch_size=1000, \
        same_batch=True,eval_rate=20,optimizer_name='adam',verbose=True,visualization=True,
        logs_dir='.',figs_dir='.',models_dir='.'):
        '''
        Trains the model with the training set and evaluates with the validation one.
        Parameters:
        - trainset (PyTorch Dataset) : Training dataset
        - validationset (PyTorch Dataset) : Validation dataset
        - n_epochs (int) : number of training epochs
        - learning_rate (float) : initial learning rate
        - learning_rate_drop (float) : multicative learning decay factor
        - learning_rate_steps (int) : number of steps before decaying the learning rate
        - sgd_batch_size (int) : size of the SGD mini-batch
        - mi_batch_size (int) : size of the MI estimation mini-batch
        - same_batch (bool) : if True, SGD and MI use the same mini-batch
        - eval_rate (int) : the model is evaluated every eval_rate epochs
        - verbose (bool) : if True, the evaluation is reported
        - visualization (bool) : if True, the evaluation is shown
        - logs_dir (str) : path for the storage of the evaluation
        - figs_dir (str) : path for the storage of the images of the evaluation
        '''

        
        # Definition of the training and validation losses, accuracies and MI
        report = 0
        n_reports = math.floor(n_epochs / eval_rate) + 1
        train_IXT_1 = np.zeros(n_reports)
        test_IXT_1 = np.zeros(n_reports)
        train_IXT_2 = np.zeros(n_reports)
        test_IXT_2 = np.zeros(n_reports)
        train_loss = np.zeros(n_reports)
        validation_loss = np.zeros(n_reports)
        train_performance = np.zeros(n_reports)     
        validation_performance = np.zeros(n_reports)
        train_IXT = np.zeros(n_reports)
        train_ITY = np.zeros(n_reports)
        validation_IXT = np.zeros(n_reports)
        validation_ITY = np.zeros(n_reports)
        epochs = np.zeros(n_reports)
        
  
        if self.problem_type == 'classification' and (self.dataset_name == 'mnist' or self.dataset_name == 'fashion_mnist' or self.dataset_name == 'CIFAR10'):
            train_performance_class = np.zeros((10,n_reports))
            validation_performance_class = np.zeros((10,n_reports))
            validation_label = np.zeros((10000,n_reports))
            validation_label_pred = np.zeros((10,10000,n_reports))
            dim_prob_evolve = np.zeros((101*500,self.K))
        

        # If regression we update the variance of the output 
        if self.problem_type == 'regression':
            self.varY = torch.var(trainset.targets)
            self.HY = 0.5 * math.log(self.varY.item()*2.0*math.pi*math.e) # in natts
            self.maxIXY = 0.848035293483288 # approximation for California Housing (just train with beta = 0 and get the value of I(T;Y) after training)
                                            # only for visualization purposes

        # Set Data Loader seed
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed) 


        g = torch.Generator()
        g.manual_seed(self.repl_n)


        # Data Loader
        n_sgd_batches = math.floor(len(trainset) / sgd_batch_size)
        sgd_train_loader = torch.utils.data.DataLoader(trainset, \
            batch_size=sgd_batch_size,shuffle=True,worker_init_fn=seed_worker,generator=g)
        if not same_batch:
            n_mi_batches = math.floor(len(trainset) / mi_batch_size)
            mi_train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=mi_batch_size,shuffle=True,worker_init_fn=seed_worker,generator=g)
            mi_train_batches = enumerate(mi_train_loader)
        validation_loader = torch.utils.data.DataLoader(validationset, \
                batch_size=len(validationset),shuffle=False)
 
        # Prior specifcation
        if self.method == "variational_IB":
            def fun(i):
                i = i.item()
                tmp = sc.comb((self.K-1),i,exact=False)*sc.beta((self.a+i),(self.b+self.K-(i+1)))/sc.beta(self.a,self.b) ## i = k-1
                return tmp
            
            A = np.arange(self.K,dtype=int)
            A = np.reshape(A,(1,len(A)))
            B = np.apply_along_axis(fun,0,A)
            B = B/np.sum(B)
            B[B==0] = 1/1e7
            pi_prior = torch.from_numpy(B).to(dev)
        else:
            pi_prior = 1
        

        # Prepare name for logs
        name_base = "K-" + str(self.K) + "-B-" + str(round(self.beta,3)).replace('.', '-') \
            + "_dim_pen_" + str(self.dim_pen).replace('.', '-') + '-'

        # Definition of the optimizer
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.network.parameters(),lr=learning_rate,weight_decay=5e-4)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = torch.optim.Adadelta(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = torch.optim.Adagrad(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)
        elif optimizer_name == 'asgd':
            optimizer = torch.optim.ASGD(self.network.parameters(),lr=learning_rate)
            
        learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, \
            step_size=learning_rate_steps,gamma=learning_rate_drop)
        
        if verbose:
            bar = progressbar
        else:
            def bar(_):
                return _

        # For all the epochs
        for epoch in range(n_epochs+1):
            if verbose:
                print("Epoch #{}/{}".format(epoch,n_epochs))

            # Randomly sample a mini batch for the SGD
            for idx_sgd_batch, sgd_batch in enumerate(bar(sgd_train_loader)):

                sgd_train_x, sgd_train_y = sgd_batch
                sgd_train_x = sgd_train_x.to(dev)
                sgd_train_y = sgd_train_y.to(dev)

                # Skip the last batch
                if idx_sgd_batch == n_sgd_batches - 1:
                    break

                # If we are not using the same batch for SGD and MI...
                if not same_batch:

                    # Randomly sample a mini batch for the MI estimation
                    idx_mi_batch, (mi_train_x, mi_train_y) = next(mi_train_batches)
                    mi_train_x = mi_train_x.to(dev)
                    mi_train_y = mi_train_y.to(dev)

                    # Prepare the MI loader again when finished
                    if (idx_mi_batch == n_mi_batches - 1):
                        mi_train_batches = enumerate(mi_train_loader)

                # If we use the same batch for SGD and MI...
                else:

                    mi_train_x, mi_train_y = sgd_train_x, sgd_train_y
                    mi_train_x = mi_train_x.to(dev)
                    mi_train_y = mi_train_y.to(dev)


                # Gradient descent
                optimizer.zero_grad()
                sgd_train_logits_y = self.network(sgd_train_x)
                if self.problem_type == 'classification':
                    sgd_train_ITY = self.get_ITY(sgd_train_logits_y,sgd_train_y)
                else: 
                    sgd_train_ITY, sgd_train_ITY_lower = self.get_ITY(sgd_train_logits_y,sgd_train_y)
                mi_train_t,mi_train_gamma,mi_train_pi,mi_sigma_t,mi_train_a,mi_train_b,mi_train_mean_t,mi_train_pi_s = self.network.encode(mi_train_x,random=False)
                with torch.no_grad():
                    if epoch<=49:
                        dim_prob_evolve[(epoch*500+idx_sgd_batch),:] = mi_train_pi.mean(0).cpu()

                mi_train_IXT_1,mi_train_IXT_2 = self.get_IXT(mi_train_t,mi_train_mean_t,mi_sigma_t,mi_train_gamma,mi_train_a,mi_train_b,mi_train_pi,mi_train_pi_s,pi_prior,self.a,self.b)
                if self.problem_type == 'classification':
                    loss = - 1.0 * (sgd_train_ITY - self.beta * self.u_func(mi_train_IXT_1 + self.dim_pen * mi_train_IXT_2))
                else: 
                    loss = - 1.0 * (sgd_train_ITY_lower - self.beta * self.u_func(mi_train_IXT_1 + self.dim_pen * mi_train_IXT_2))
                loss.backward()
                

                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

            # Update learning rate
            learning_rate_scheduler.step()

            # Report results
            if epoch % eval_rate == 0:
                with torch.no_grad():
                    epochs[report] = epoch

                    for _, train_batch in enumerate(sgd_train_loader):
                        train_x, train_y = train_batch
                        train_x = train_x.to(dev)
                        train_y = train_y.to(dev)
                        train_logits_y = self.network(train_x)
                        train_t,train_gamma,train_pi,train_sigma_t,train_a,train_b,train_mean_t,train_pi_s = self.network.encode(train_x,random=False)
                        tmp1,tmp2 =  self.get_IXT(train_t,train_mean_t,train_sigma_t,train_gamma,train_a,train_b,train_pi,train_pi_s,pi_prior,self.a,self.b)
                        train_IXT[report] += self.IXT.item()/n_sgd_batches
                        train_IXT_1[report] += tmp1.item() / n_sgd_batches
                        train_IXT_2[report] += tmp2.item() / n_sgd_batches
                        if self.problem_type == 'classification':
                            train_ITY[report] += self.get_ITY(train_logits_y,train_y).item() / n_sgd_batches
                        else: 
                            tmp_train_ITY, _ = self.get_ITY(train_logits_y,train_y)
                            train_ITY[report] += tmp_train_ITY.item() / n_sgd_batches 
                        train_loss[report] = - 1.0 * (train_ITY[report] - \
                            self.beta * train_IXT[report])
                        train_performance[report] += self.evaluate(train_logits_y,train_y).item() / n_sgd_batches
                        if self.problem_type == 'classification' and (self.dataset_name == 'mnist' or self.dataset_name == 'fashion_mnist' or self.dataset_name == 'CIFAR10'):
                            n_class = 10
                            train_performance_class[:,report] += self.class_evaluate(train_logits_y,train_y,n_class).cpu().numpy()/ n_sgd_batches

                    _, validation_batch = next(enumerate(validation_loader))
                    validation_x, validation_y = validation_batch
                    validation_x = validation_x.to(dev)
                    validation_y = validation_y.to(dev)
                    validation_logits_y = self.network(validation_x)
                    validation_t,validation_gamma,validation_pi,validation_sigma_t,validation_a,validation_b,validation_mean_t,validation_pi_s = self.network.encode(validation_x,random=False)
                    tmp1,tmp2 = self.get_IXT(validation_t,validation_mean_t,validation_sigma_t,validation_gamma,validation_a,validation_b,validation_pi,validation_pi_s,pi_prior,self.a,self.b)
                    validation_IXT[report] = self.IXT.item()
                    test_IXT_1[report] = tmp1.item()
                    test_IXT_2[report] = tmp2.item()
                    if self.problem_type == 'classification':
                        validation_ITY[report] = self.get_ITY(validation_logits_y,validation_y).item()
                    else: 
                        tmp_validation_ITY, _ = self.get_ITY(validation_logits_y,validation_y) 
                        validation_ITY[report] = tmp_validation_ITY.item()
                    validation_loss[report] = - 1.0 * (validation_ITY[report] - \
                        self.beta * validation_IXT[report])
                    validation_performance[report] = self.evaluate(validation_logits_y,validation_y).item()
                    if self.problem_type == 'classification' and (self.dataset_name == 'mnist' or self.dataset_name == 'fashion_mnist' or self.dataset_name == 'CIFAR10'):
                        n_class = 10
                        validation_performance_class[:,report] = self.class_evaluate(validation_logits_y,validation_y,n_class).cpu().numpy()
                        validation_label[:,report] = validation_y.cpu().numpy()
                        validation_label_pred[:,:,report] = torch.max(validation_logits_y,dim=2)[1].cpu().numpy()
                        

                if verbose:
                    print("\n",flush=True)
                    print("\n** Results report **",flush=True)
                    print("- I(X;T) = " + str(train_IXT[report]),flush=True)
                    print("- I(T;Y) = " + str(train_ITY[report]),flush=True)
                    if self.problem_type == 'classification':
                        print("- Training accuracy: " + str(train_performance[report]),flush=True)
                        print("- Validation accuracy: " + str(validation_performance[report]),flush=True)
                    else:
                        print("- Training MSE: " + str(train_performance[report]),flush=True)
                        print("- Validation MSE: " + str(validation_performance[report]),flush=True)
                    print("\n",flush=True)

                report += 1

                # Save results
                #if self.K == 2:
                with torch.no_grad():
                    _, (visualize_x,visualize_y) = next(enumerate(validation_loader))
                    visualize_x = visualize_x.to(dev)
                    visualize_y = visualize_y.to(dev)
                    visualize_t,visualize_gamma,visualize_pi,visualize_sigma,visualize_a,visualize_b,visualize_t_mean,visualize_pi_s = self.network.encode(visualize_x,random=False)
                    visualize_t_rand,visualize_gamma,visualize_pi,_,_,_,_,_ = self.network.encode(visualize_x,random=True)
                    
                np.save(logs_dir + name_base + 'hidden_dim_encoder', visualize_gamma.cpu())
                np.save(logs_dir + name_base + 'hidden_dim_probabilities', visualize_pi.cpu())
                np.save(logs_dir + name_base + 'post_a', visualize_a.cpu())
                np.save(logs_dir + name_base + 'post_b', visualize_b.cpu())
                np.save(logs_dir + name_base + 'post_pi_samples', visualize_pi_s.cpu())
                np.save(logs_dir + name_base + 'dim_probabilities_evolution', dim_prob_evolve)   
                np.save(logs_dir + name_base + 'hidden_variables_random', visualize_t_rand.cpu())
                np.save(logs_dir + name_base + 'hidden_variables_mean', visualize_t_mean.cpu())
                np.save(logs_dir + name_base + 'hidden_variables_sigma', visualize_sigma.cpu())
                np.save(logs_dir + name_base + 'train_accuracy_class', train_performance_class)
                np.save(logs_dir + name_base + 'validation_accuracy_class', validation_performance_class)
                np.save(logs_dir + name_base + 'validation_labels', validation_label)
                np.save(logs_dir + name_base + 'validation_labels_pred', validation_label_pred)
                np.save(logs_dir + name_base + 'train_IXT_1', train_IXT_1)
                np.save(logs_dir + name_base + 'validation_IXT_1', test_IXT_1)
                np.save(logs_dir + name_base + 'train_IXT_2', train_IXT_2)
                np.save(logs_dir + name_base + 'validation_IXT_2', test_IXT_2)
                np.save(logs_dir + name_base + 'train_IXT', train_IXT)
                np.save(logs_dir + name_base + 'validation_IXT', validation_IXT)
                np.save(logs_dir + name_base + 'train_ITY', train_ITY)
                np.save(logs_dir + name_base + 'validation_ITY', validation_ITY)
                np.save(logs_dir + name_base + 'train_loss', train_loss)
                np.save(logs_dir + name_base + 'validation_loss', validation_loss)
                if self.problem_type == 'classification':
                    np.save(logs_dir + name_base + 'train_accuracy', train_performance)
                    np.save(logs_dir + name_base + 'validation_accuracy', validation_performance)
                else: 
                    np.save(logs_dir + name_base + 'train_mse', train_performance)
                    np.save(logs_dir + name_base + 'validation_mse', validation_performance) 
                np.save(logs_dir + name_base + 'epochs', epochs)
            
            if epoch in [100,150,200,250,300,350]:
                checkpoint = { 
                    'epoch': epoch,
                    'model': self.network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': learning_rate_scheduler.state_dict()}
                torch.save(checkpoint, models_dir + name_base + 'model_epoch_' + str(epoch))
