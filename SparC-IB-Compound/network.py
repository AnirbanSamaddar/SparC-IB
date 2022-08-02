import torch
import torchvision
import numpy as np
import scipy.special as sc
from derivatives import Beta

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Deterministic_encoder(torch.nn.Module):
    '''
    Probabilistic encoder of the network.

    '''

    def __init__(self,K,n_x,network_type):
        super(Deterministic_encoder,self).__init__()

        self.K = K
        self.n_x = n_x
        self.network_type = network_type

        if self.network_type == 'mlp_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,2*self.K+2))
            self.f_theta = torch.nn.Sequential(*layers)       

        elif self.network_type == 'mlp_CIFAR10':            
            model = torchvision.models.vgg16()
            self.f_theta_conv = torch.nn.Sequential(*(list(model.children())[:-1]))

            self.f_theta_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, (2*self.K+2))
            )

    def forward(self,x):

        if self.network_type == 'mlp_mnist':
            x = x.view(-1,self.n_x)
            mean_t = self.f_theta(x)
        elif self.network_type == 'mlp_CIFAR10':
            mean_t_conv = self.f_theta_conv(x)
            mean_t_conv = mean_t_conv.flatten(1)
            mean_t = self.f_theta_lin(mean_t_conv)

        return mean_t


class IBP_encoder(torch.nn.Module):
    '''
    Encoder network to be used for Intel-VIB

    '''

    def __init__(self,K,n_x,network_type):
        super(IBP_encoder,self).__init__()

        self.K = K
        self.n_x = n_x
        self.network_type = network_type

        if self.network_type == 'mlp_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,2*self.K))
            self.f_theta = torch.nn.Sequential(*layers)       

        elif self.network_type == 'mlp_CIFAR10':
           
            model = torchvision.models.vgg16()
            self.f_theta_conv = torch.nn.Sequential(*(list(model.children())[:-1]))

            self.f_theta_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 2*self.K)
            )

    def forward(self,x):

        if self.network_type == 'mlp_mnist':
            x = x.view(-1,self.n_x)
            mean_t = self.f_theta(x)
        elif self.network_type == 'mlp_CIFAR10':
            mean_t_conv = self.f_theta_conv(x)
            mean_t_conv = mean_t_conv.flatten(1)
            mean_t = self.f_theta_lin(mean_t_conv)

        return mean_t


class Feature_extractor(torch.nn.Module):
    '''
    Encoder network to be used for Drop-VIB fature extractor

    '''

    def __init__(self,K,n_x,network_type):
        super(Feature_extractor,self).__init__()

        self.K = K
        self.n_x = n_x
        self.network_type = network_type

        if self.network_type == 'mlp_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,self.K))
            self.f_shi = torch.nn.Sequential(*layers)       

        elif self.network_type == 'mlp_CIFAR10':            
            model = torchvision.models.vgg16()
            self.f_shi_conv = torch.nn.Sequential(*(list(model.children())[:-1]))

            self.f_shi_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, self.K)
            )


    def forward(self,x):

        if self.network_type == 'mlp_mnist':
            x = x.view(-1,self.n_x)
            mean_t = self.f_shi(x)
        elif self.network_type == 'mlp_CIFAR10':
            mean_t_conv = self.f_shi_conv(x)
            mean_t_conv = mean_t_conv.flatten(1)
            mean_t = self.f_shi_lin(mean_t_conv)

        return mean_t



class Probability_encoder(torch.nn.Module):

    def __init__(self,K):
        super(Probability_encoder,self).__init__()

        self.K = K

        layers = []
        layers.append(torch.nn.Linear(self.K,10))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(10,10))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(10,self.K))
        layers.append(torch.nn.Sigmoid())
        self.prob_theta = torch.nn.Sequential(*layers)


    def forward(self,y):
        pi = self.prob_theta(y)

        return pi





class Deterministic_decoder(torch.nn.Module):
    '''
    Deterministic decoder of the network.

    '''

    def __init__(self,K,n_y,network_type):
        super(Deterministic_decoder,self).__init__()

        self.K = K
        self.network_type = network_type

        if network_type == 'mlp_mnist' or network_type == 'conv_net_fashion_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.K,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,n_y))
            self.g_theta = torch.nn.Sequential(*layers)
        elif network_type == 'mlp_CIFAR10':
            self.g_theta = torch.nn.Sequential(torch.nn.Linear(self.K,n_y))


    def forward(self,t,gamma=1.0):

        logits_y =  self.g_theta(t*gamma)
        return logits_y

class nlIB_network(torch.nn.Module):
    '''
    Nonlinear Information Bottleneck network.
    - We use the one in Kolchinsky et al. 2017 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
        · n_y (int) : dimensionality of the output variable (number of classes)
        · train_logvar_t (bool) : if true, logvar_t is trained
    '''

    def __init__(self,K,n_x,n_y,a_prior,b_prior,logvar_t=-1.0,train_logvar_t=False,network_type='mlp_mnist',method='nonlinear_IB',TEXT=None):
        super(nlIB_network,self).__init__()

        self.network_type = network_type
        self.method = method
        self.K = K
        
        if self.method == 'variational_IB':
            self.beta = Beta.apply
            self.encoder = Deterministic_encoder(K,n_x,self.network_type)
        elif self.method == 'drop_VIB':
            self.encoder = Feature_extractor(K,n_x,self.network_type)
            self.pi_dash = torch.nn.Parameter(data=torch.FloatTensor(1,K).uniform_(-2,1).to(dev),requires_grad=True)
        elif self.method == 'intel_VIB':
            self.encoder = IBP_encoder(K,n_x,self.network_type)
            self.prob_encoder = Probability_encoder(K)
        self.decoder = Deterministic_decoder(K,n_y,self.network_type)

    def encode(self,x,random=True):

        if self.method == 'variational_IB':
            m = torch.nn.Softplus()
            tmp1 = self.encoder(x) 
            mean_t = tmp1[:,0:self.K]
            sigma_t = m(tmp1[:,self.K:(tmp1.shape[1]-2)])
            ab = m(tmp1[:,(tmp1.shape[1]-2):tmp1.shape[1]])
            a = ab[:,0][:,None]
            b = ab[:,1][:,None]
            dim = np.arange(self.K,dtype=int)[None,:]
            fun1 = lambda r: sc.comb(self.K-1,r.item(),exact=False)
            combs = np.apply_along_axis(fun1,0,dim)[None,:]
            dim = torch.from_numpy(dim).to(dev)
            pi = torch.from_numpy(combs).to(dev) * (self.beta(a+dim,b+self.K-(dim+1)))/(self.beta(a,b))
            pi = pi.type('torch.FloatTensor').to(dev)            
            pi1 = pi.clone()
            pi1[pi1 == 0] = 1/1e7
            pi = pi1
            pi_s = torch.tensor([0.0]).to(dev)
        elif self.method == 'drop_VIB':
            mean_t = self.encoder(x)
            s = torch.nn.Sigmoid() 
            pi = s(self.pi_dash)
            pi1 = pi.clone()
            pi1[pi1 == 0] = 1e-7
            pi1[(1-pi1) == 0] = 1e-7 
            pi = pi1
            cat_pi = torch.cat(((1-pi)[:,:,None],pi[:,:,None]),dim = -1)
            sigma_t = torch.tensor([0.0]).to(dev)
            a = torch.tensor([0.0]).to(dev)
            b = torch.tensor([0.0]).to(dev)
            pi_s = torch.tensor([0.0]).to(dev)
        elif self.method == 'intel_VIB':
            m = torch.nn.Softplus()
            tmp1 = self.encoder(x) 
            mean_t = tmp1[:,0:self.K]
            sigma_t = m(tmp1[:,self.K:tmp1.shape[1]])
            a = torch.tensor([0.0]).to(dev)
            b = torch.tensor([0.0]).to(dev)
            pi_s = torch.tensor([0.0]).to(dev)


        if random:
            if self.method == 'variational_IB':
                t = mean_t.repeat(10,1,1) + sigma_t.repeat(10,1,1) * torch.randn_like(mean_t.repeat(10,1,1)).to(dev)
                tmp = torch.nn.functional.gumbel_softmax(torch.log(pi.repeat(10,1,1)),tau=0.1,hard=True)
                mask = tmp.cumsum(dim=2)
                gamma = (1 - mask) + tmp
            elif self.method == 'drop_VIB':
                t = mean_t.repeat(10,1,1)
                tmp = torch.nn.functional.gumbel_softmax(torch.log(cat_pi.repeat(10,mean_t.shape[0],1,1)),tau=0.1,hard=False)
                tmp1 = tmp[:,:,:,0]
                gamma = tmp1*(self.K/(self.K - pi.sum(dim=1)[None,None,:]))
            elif self.method == 'intel_VIB':
                t = mean_t.repeat(10,1,1) + sigma_t.repeat(10,1,1) * torch.randn_like(mean_t.repeat(10,1,1)).to(dev)
                pi = self.prob_encoder(t)
                gamma = pi

        else:
            if self.method == 'intel_VIB':
                t = mean_t + sigma_t * torch.randn_like(mean_t).to(dev)
                pi = self.prob_encoder(t)
                gamma = pi
            else:
                t = mean_t
                gamma = pi

        return t,gamma,pi,sigma_t,a,b,mean_t,pi_s

    def decode(self,t,gamma=1):

        logits_y = self.decoder(t,gamma)
        return logits_y

    def forward(self,x):

        t,gamma,_,_,_,_,_,_ = self.encode(x)
        logits_y = self.decode(t,gamma)
        return logits_y
