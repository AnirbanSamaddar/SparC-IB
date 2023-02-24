import torch#, torchtext
import math
import autograd
import scipy.optimize
from progressbar import progressbar
import numpy as np
import random
from network_fixed import nlIB_network

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvexIB(torch.nn.Module):

    def __init__(self,n_x,n_y,problem_type,network_type,K,beta,logvar_t=-1.0,logvar_kde=-1.0,\
        train_logvar_t=False, u_func_name='pow', hyperparameter=1.0, TEXT=None, compression_level = 1.0, method = 'nonlinear_IB', dataset_name = 'mnist',repl_n=1):
        super(ConvexIB,self).__init__()


        self.repl_n = repl_n
        torch.manual_seed(self.repl_n)

        self.HY = np.log(n_y) # in natts
        self.maxIXY = self.HY # in natts
        self.varY = 0 # to be updated with the training dataset
        self.IXT = 0 # to be updated
        self.ITY = 0 # to be

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
        self.method = method
        
        self.K = K
        self.beta = beta
        self.network = nlIB_network(K,n_x,n_y,logvar_t,train_logvar_t,network_type,TEXT).to(dev)
        self.dataset_name = dataset_name

        self.problem_type = problem_type 
        if self.problem_type == 'classification':
            self.ce = torch.nn.CrossEntropyLoss()
        else:
            self.mse = torch.nn.MSELoss()


    def get_IXT(self,mean_t,sigma_t):
        '''
        Obtains the mutual information between the iput and the bottleneck variable.
        Parameters:
        - mean_t,sigma_t (Tensor) : deterministic transformation of the input
        '''

        if self.method == 'variational_IB':
            self.IXT = -0.5*(1+2*torch.log(sigma_t)-mean_t.pow(2)-sigma_t**2).sum(1).mean().div(math.log(2))

        # NaNs and exploding gradients control
        with torch.no_grad():
            if self.u_func_name == 'shifted-exp':
                if self.IXT > self.compression_level:
                    self.IXT -= (self.IXT - self.compression_level - 0.01)
            if self.u_func(torch.Tensor([self.IXT])) == float('inf'):
                self.IXT = torch.Tensor([1e5])

        return self.IXT.to(dev)

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
            #LEN = y.size()[0]
            #LEN2 = logits_y.size()[0]
            #print("y: "+str(LEN))
            #print("logits: "+str(LEN2))
            #print("y_device: "+str(y.get_device()))
            #print("logits_device: "+str(logits_y.get_device()))
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
        logs_dir='.',figs_dir='.'):
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
        train_loss = np.zeros(n_reports)
        validation_loss = np.zeros(n_reports)
        train_performance = np.zeros(n_reports)     
        validation_performance = np.zeros(n_reports)
        train_IXT = np.zeros(n_reports)
        train_ITY = np.zeros(n_reports)
        validation_IXT = np.zeros(n_reports)
        validation_ITY = np.zeros(n_reports)
        epochs = np.zeros(n_reports)
        n_sgd_batches = math.floor(len(trainset) / sgd_batch_size)
        if self.problem_type == 'classification' and (self.dataset_name == 'mnist' or self.dataset_name == 'fashion_mnist' or self.dataset_name == 'CIFAR10'):
            train_performance_class = np.zeros((10,n_reports))
            validation_performance_class = np.zeros((10,n_reports))
            validation_label = np.zeros((10000,n_reports))
            validation_label_pred = np.zeros((10,10000,n_reports))
            #train_label = np.zeros((128,n_reports,(n_sgd_batches+1)))
            #train_label_pred = np.zeros((128,n_reports,(n_sgd_batches+1)))

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
        sgd_train_loader = torch.utils.data.DataLoader(trainset, \
            batch_size=sgd_batch_size,shuffle=True,worker_init_fn=seed_worker,generator=g)
        if not same_batch:
            n_mi_batches = math.floor(len(trainset) / mi_batch_size)
            mi_train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=mi_batch_size,shuffle=True,worker_init_fn=seed_worker,generator=g)
            mi_train_batches = enumerate(mi_train_loader)
        validation_loader = torch.utils.data.DataLoader(validationset, \
                batch_size=len(validationset),shuffle=False)


        # Prepare name for figures and logs
        name_base = "K-" + str(self.K) + "-B-" + str(round(self.beta,3)).replace('.', '-') \
             + '-'

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
                mi_train_mean_t,mi_sigma_t = self.network.encode(mi_train_x,random=False)
                mi_train_IXT = self.get_IXT(mi_train_mean_t,mi_sigma_t)
                if self.problem_type == 'classification':
                    loss = - 1.0 * (sgd_train_ITY - self.beta * self.u_func(mi_train_IXT)) 
                else: 
                    loss = - 1.0 * (sgd_train_ITY_lower - self.beta * self.u_func(mi_train_IXT))
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
                        train_mean_t,train_sigma_t = self.network.encode(train_x,random=False)
                        train_IXT[report] += self.get_IXT(train_mean_t,train_sigma_t).item() / n_sgd_batches
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
                            train_performance_class[:,report] += self.class_evaluate(train_logits_y,train_y,n_class).cpu().numpy()/n_sgd_batches
                            

                    _, validation_batch = next(enumerate(validation_loader))
                    validation_x, validation_y = validation_batch
                    validation_x = validation_x.to(dev)
                    validation_y = validation_y.to(dev)
                    validation_logits_y = self.network(validation_x)
                    validation_mean_t,validation_sigma_t = self.network.encode(validation_x,random=False)
                    validation_IXT[report] = self.get_IXT(validation_mean_t,validation_sigma_t).item()
                    if self.problem_type == 'classification':
                        validation_ITY[report] = self.get_ITY(validation_logits_y,validation_y).item()
                    else: 
                        tmp_validation_ITY, _ = self.get_ITY(validation_logits_y,validation_y) 
                        validation_ITY[report] = tmp_validation_ITY.item()
                    validation_loss[report] = - 1.0 * (validation_ITY[report] - \
                        self.beta * train_IXT[report])
                    validation_performance[report] = self.evaluate(validation_logits_y,validation_y).item()
                    if self.problem_type == 'classification' and (self.dataset_name == 'mnist' or self.dataset_name == 'fashion_mnist'or self.dataset_name == 'CIFAR10'):
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
                    visualize_t,visualize_sigma = self.network.encode(visualize_x,random=True)
                np.save(logs_dir + name_base + 'hidden_variables', visualize_t.cpu())
                np.save(logs_dir + name_base + 'hidden_variables_sigma', visualize_sigma.cpu())
                np.save(logs_dir + name_base + 'train_accuracy_class', train_performance_class)
                np.save(logs_dir + name_base + 'validation_accuracy_class', validation_performance_class)
                np.save(logs_dir + name_base + 'validation_labels', validation_label)
                np.save(logs_dir + name_base + 'validation_labels_pred', validation_label_pred)
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
