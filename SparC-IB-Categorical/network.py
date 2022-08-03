import torch
import torchvision

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
            layers.append(torch.nn.Linear(800,3*self.K))
            self.f_theta = torch.nn.Sequential(*layers)       

        elif self.network_type == 'mlp_CIFAR10':
            
            model = torchvision.models.vgg16()
            self.f_theta_conv = torch.nn.Sequential(*(list(model.children())[:-1]))

            self.f_theta_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 3*self.K)
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




class Deterministic_decoder(torch.nn.Module):
    '''
    Deterministic decoder of the network.

    '''

    def __init__(self,K,n_y,network_type):
        super(Deterministic_decoder,self).__init__()

        self.K = K
        self.network_type = network_type

        if network_type == 'mlp_mnist':
            layers = []
            layers.append(torch.nn.Linear(self.K,800))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(800,n_y))
            self.g_theta = torch.nn.Sequential(*layers)
        elif network_type == 'mlp_CIFAR10':
            self.g_theta = torch.nn.Sequential(torch.nn.Linear(self.K,n_y))

    def forward(self,t,gamma=1):

        logits_y =  self.g_theta(t*gamma)
        return logits_y

class nlIB_network(torch.nn.Module):
    '''
    Nonlinear Information Bottleneck network.

    '''

    def __init__(self,K,n_x,n_y,logvar_t=-1.0,train_logvar_t=False,network_type='mlp_mnist',method='nonlinear_IB',TEXT=None):
        super(nlIB_network,self).__init__()

        self.network_type = network_type
        self.method = method
        self.K = K
        self.encoder = Deterministic_encoder(K,n_x,self.network_type)
        self.decoder = Deterministic_decoder(K,n_y,self.network_type)

    def encode(self,x,random=True):

        m = torch.nn.Softplus()
        s = torch.nn.Softmax(dim=1)
        tmp1 = self.encoder(x) 
        mean_t = tmp1[:,0:self.K]
        sigma_t = m(tmp1[:,self.K:(tmp1.shape[1]-self.K)])
        pi = s(tmp1[:,(tmp1.shape[1]-self.K):tmp1.shape[1]])
        if self.method == 'variational_IB':
            pi1 = pi.clone()
            pi1[pi1 == 0] = 1/1e7
            pi = pi1
        else:
            pi = 1        
        if random:
            t = mean_t.repeat(10,1,1) + sigma_t.repeat(10,1,1) * torch.randn_like(mean_t.repeat(10,1,1)).cuda()
            tmp = torch.nn.functional.gumbel_softmax(torch.log(pi.repeat(10,1,1)),tau=0.1,hard=True)
            mask = tmp.cumsum(dim=2)
            gamma = (1 - mask) + tmp

        else:
            t = mean_t
            gamma = pi
        return t,gamma,pi,sigma_t

    def decode(self,t,gamma=1):

        logits_y = self.decoder(t,gamma)
        return logits_y

    def forward(self,x):

        t,gamma,pi,_ = self.encode(x)
        logits_y = self.decode(t,gamma)
        return logits_y
