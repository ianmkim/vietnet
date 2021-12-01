import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class SpikingNeuronLayerRNN(nn.Module):
    def __init__(self, device, n_inputs=28*28, n_hidden=100,
                decay_multiplier=0.9, threshold=2.0, penalty_threshold=2.5):
        super(SpikingNeuronLayerRNN, self).__init__()
        
        self.device = device
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.decay_multiplier = decay_multiplier
        self.threshold = threshold
        self.penalty_threshold = penalty_threshold
        
        self.fc = nn.Linear(n_inputs, n_hidden)
        
        self.init_parameters() 
        self.reset_state()
        self.to(self.device)
    
    def init_parameters(self):
        for param in self.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
    
    def reset_state(self):
        self.prev_inner = torch.zeros([self.n_hidden]).to(self.device)
        self.prev_outer = torch.zeros([self.n_hidden]).to(self.device)
        
    def forward(self, x):
        if self.prev_inner.dim() == 1:
            batch_size = x.shape[0]
            self.prev_inner = torch.stack(batch_size * [self.prev_inner])
            self.prev_outer = torch.stack(batch_size * [self.prev_outer])
            
        input_excitation = self.fc(x)
        
        inner_excitation = input_excitation + self.prev_inner * self.decay_multiplier
        outer_excitation = F.relu(inner_excitation - self.threshold)
        
        do_penalize_gate = (outer_excitation > 0).float()
        inner_excitation = inner_excitation - do_penalize_gate * (self.penalty_threshold/self.threshold * inner_excitation)
        
        delayed_return_state = self.prev_inner 
        delayed_return_output = self.prev_outer
        self.prev_inner = inner_excitation
        self.prev_outer = outer_excitation
        return delayed_return_state, delayed_return_output
    
class InputDataToSpikingPerceptronLayer(nn.Module):
    def __init__(self, device):
        super(InputDataToSpikingPerceptronLayer, self).__init__()
        self.device = device 
        self.reset_state()
        self.to(self.device)
        
    def reset_state(self):
        pass
    
    def forward(self, x, is_2D=True):
        if is_2D:
            x = x.view(x.size(0), -1)
        random_activation_perceptron = torch.rand(x.shape).to(self.device)
        return random_activation_perceptron * x
    
class OutputDataToSpikingPerceptronLayer(nn.Module):
    def __init__(self, average_output=True):
        super(OutputDataToSpikingPerceptronLayer, self).__init__()
        if average_output:
            self.reducer = lambda x, dim: x.sum(dim=dim)
        else:
            self.reducer = lambda x, dim: x.mean(dim=dim)
            
    def forward(self, x):
        if type(x) == list:
            x = torch.stack(x)
        return self.reducer(x, 0)
