import torch
import torch.nn as nn

class AddBias(nn.Module):
    def __init__(self, num_features, threshold=8):
        super(AddBias, self).__init__()
        self.num_features = num_features
        self.threshold = threshold
        self.register_buffer('count', torch.tensor(0))
    
    def forward(self, x):
        if self.count < self.threshold:
            mu = x.mean(dim=self.dim, keepdim=True).detach()
            self.init_mean += (mu - self.init_mean) / (self.count + 1)
            self.count += 1
            return x.add(self._bias.mul(1e-2).sub(self.init_mean))
        return x.add(self._bias.sub(self.init_mean))

    def extra_repr(self):
        return 'num_features={}, threshold={}'.format(
            self.num_features, self.threshold)

class Bias1D(AddBias):
    def __init__(self, num_features, threshold=8):
        super(Bias1D, self).__init__(num_features, threshold)
        self._bias = nn.Parameter(torch.zeros(1, num_features))
        self.register_buffer('init_mean', torch.zeros(1, num_features))
        self.dim = 0
    
class Bias2D(AddBias):
    def __init__(self, num_features, threshold=8):
        super(Bias2D, self).__init__(num_features, threshold)
        self._bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('init_mean', torch.zeros(1, num_features, 1, 1))
        self.dim = (0,2,3)
    
class Bias3D(AddBias):
    def __init__(self, num_features, threshold=8):
        super(Bias3D, self).__init__(num_features, threshold)
        self._bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.register_buffer('init_mean', torch.zeros(1, num_features, 1, 1, 1))
        self.dim = (0,2,3,4)