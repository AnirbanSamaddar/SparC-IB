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
            layers.append(torch.nn.Linear(800,2*self.K))
            self.f_theta = torch.nn.Sequential(*layers)       
        elif self.network_type == 'mlp_CIFAR10':
            
            model = torchvision.models.vgg16()
            self.f_theta_conv = torch.nn.Sequential(*(list(model.children())[:-1]))

            self.f_theta_lin = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 2*self.K)
            )
        elif self.network_type == 'mlp_ImageNet':
            layers = []
            layers.append(torch.nn.Linear(self.n_x,1024))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(1024,1024))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(1024,(2*self.K)))
            self.f_theta = torch.nn.Sequential(*layers)


    def forward(self,x):

        if self.network_type == 'mlp_mnist' or self.network_type == 'mlp_ImageNet':
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
        elif network_type == 'mlp_CIFAR10' or network_type == 'mlp_ImageNet':
            self.g_theta = torch.nn.Sequential(torch.nn.Linear(self.K,n_y))


    def forward(self,t):

        logits_y =  self.g_theta(t)
        return logits_y

class nlIB_network(torch.nn.Module):

    def __init__(self,K,n_x,n_y,logvar_t=-1.0,train_logvar_t=False,network_type='mlp_mnist',method='nonlinear_IB',TEXT=None):
        super(nlIB_network,self).__init__()

        self.K = K
        self.network_type = network_type
        self.method = method
        self.encoder = Deterministic_encoder(K,n_x,self.network_type)
        self.decoder = Deterministic_decoder(K,n_y,self.network_type)

    def encode(self,x,random=True):

        m = torch.nn.Softplus()
        tmp1 = self.encoder(x) 
        mean_t = tmp1[:,0:self.K]
        sigma_t = m(tmp1[:,self.K:tmp1.shape[1]])
   
        if random:
            t = mean_t.repeat(10,1,1) + sigma_t.repeat(10,1,1) * torch.randn_like(mean_t.repeat(10,1,1)).cuda()
        else:
            t = mean_t
        return t,sigma_t

    def decode(self,t):

        logits_y = self.decoder(t)
        return logits_y

    def forward(self,x):

        t,_ = self.encode(x)
        logits_y = self.decode(t)
        return logits_y
