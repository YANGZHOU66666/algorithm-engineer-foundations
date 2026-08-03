from torch import nn
import torch

class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # (batch_size, num_channels, width, height)
        if self.training:
            batch_mean = x.mean(dim=(0,2,3)) #(num_channels,)
            batch_var = x.var(dim=(0,2,3), unbias=False)

            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*batch_mean
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*batch_var

            mean_to_use = batch_mean.view(1,-1,1,1)
            var_to_use = batch_var.view(1,-1,1,1)
        else:
            mean_to_use = self.running_mean.view(1,-1,1,1)
            var_to_use = self.running_var.view(1,-1,1,1)
        
        norm = (x-mean_to_use)/(var_to_use+self.epsilon) # (batch_size, num_channels, width, height)
        reshaped_gamma = self.gamma.view(1,-1,1,1)
        reshaped_beta = self.beta.view(1,-1,1,1)
        return reshaped_gamma*norm + reshaped_beta
        
class LayerNorm(nn.Module):
    def __init__(self, norm_shape, epsilon):
        super().__init__()
        if isinstance(norm_shape, int):
            norm_shape = (norm_shape,)
        self.norm_shape = norm_shape
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(norm_shape))
        self.beta = nn.Parameter(torch.zeros(norm_shape))

    def forward(self, x):
        # (batch_size, ...)
        norm_dims = tuple(range(-len(self.norm_shape),0))
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = x.var(dim=norm_dims, keepdim=True, unbias=False)

        norm = (x-mean)/torch.sqrt(var+self.epsilon)
        return self.gamma*norm+self.beta
        

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, epsilon=1e-5):
        super.__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):
        normalized_dims = tuple(range(-len(normalized_dims),0))
        batch_size, seq_len, embed_dim = x.shape
        
        squares = x**2
        means = squares.mean(dim=normalized_dims, keepdim=True)
        sqrt_means = torch.sqrt(means+self.epsilon)
        x_hats = x/sqrt_means
        outs = x_hats*self.gamma
        return outs
