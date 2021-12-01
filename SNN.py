import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from SNNComponents import SpikingNeuronLayerRNN
from SNNComponents import OutputDataToSpikingPerceptronLayer
from SNNComponents import InputDataToSpikingPerceptronLayer

class SpikingNet(nn.Module):
    def __init__(self, device, n_time_steps, begin_eval):
        super(SpikingNet, self).__init__()
        assert( 0 <= begin_eval and begin_eval < n_time_steps)
        self.deice = device 
        self.n_time_steps = n_time_steps
        self.begin_eval = begin_eval
        
        self.input_conversion = InputDataToSpikingPerceptronLayer(device)
        
        self.layer1 = SpikingNeuronLayerRNN(
            device, n_inputs=28*28, n_hidden=100, decay_multiplier=0.9, threshold=1.0, penalty_threshold=1.5)
        
        self.layer2 = SpikingNeuronLayerRNN(
            device, n_inputs=100, n_hidden=10, decay_multiplier=0.9, threshold=1.0, penalty_threshold=1.5)
        
        self.output_conversion = OutputDataToSpikingPerceptronLayer(average_output=False)
        
        self.to(device)
        
    def forward_through_time(self, x):
        self.input_conversion.reset_state()
        self.layer1.reset_state()
        self.layer2.reset_state()
        
        out = []
        
        all_layer1_states = []
        all_layer1_outputs = []
        
        all_layer2_states = []
        all_layer2_outputs = []
        
        for _ in range(self.n_time_steps):
            xi = self.input_conversion(x)
            
            layer1_state, layer1_output = self.layer1(xi)
            layer2_state, layer2_output = self.layer2(layer1_output)
            
            all_layer1_states.append(layer1_state)
            all_layer1_outputs.append(layer1_output)
            
            all_layer2_states.append(layer2_state)
            all_layer2_outputs.append(layer2_output)
            out.append(layer2_state)
        
        out = self.output_conversion(out[self.begin_eval:])
        return out, [[all_layer1_states, all_layer1_outputs], 
                     [all_layer2_states, all_layer2_outputs]]
 

    def forward(self, x):
        out, _ = self.forward_through_time(x)
        return F.log_softmax(out, dim=-1)
    
    
    def visualize_all_neurons(self, x):

        assert x.shape[0] == 1 and len(x.shape) == 4, (

            "Pass only 1 example to SpikingNet.visualize(x) with outer dimension shape of 1.")
        _, layers_state = self.forward_through_time(x)
 
        for i, (all_layer_states, all_layer_outputs) in enumerate(layers_state):
            layer_state  =  torch.stack(all_layer_states).data.cpu(
                ).numpy().squeeze().transpose()
            layer_output = torch.stack(all_layer_outputs).data.cpu(
                ).numpy().squeeze().transpose()
 

            self.plot_layer(layer_state, title="Inner state values of neurons for layer {}".format(i))
            self.plot_layer(layer_output, title="Output spikes (activation) values of neurons for layer {}".format(i))

    def visualize_neuron(self, x, layer_idx, neuron_idx):
        assert x.shape[0] == 1 and len(x.shape) == 4, (
            "Pass only 1 example to SpikingNet.visualize(x) with outer dimension shape of 1.")
        _, layers_state = self.forward_through_time(x)

        all_layer_states, all_layer_outputs = layers_state[layer_idx]
        layer_state  =  torch.stack(all_layer_states).data.cpu(
            ).numpy().squeeze().transpose()
        layer_output = torch.stack(all_layer_outputs).data.cpu(
            ).numpy().squeeze().transpose()

        self.plot_neuron(
            layer_state[neuron_idx], 
            title="Inner state values neuron {} of layer {}".format(neuron_idx, layer_idx))
        self.plot_neuron(
            layer_output[neuron_idx], 
            title="Output spikes (activation) values of neuron {} of layer {}".format(neuron_idx, layer_idx))
 
    def plot_layer(self, layer_values, title):
        """
        This function is derived from:
            https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
        Which was released under the MIT License.
        """
        width = max(16, layer_values.shape[0] / 8)
        height = max(4, layer_values.shape[1] / 8)
        plt.figure(figsize=(width, height))
        plt.imshow(
           layer_values,
            interpolation="nearest",
            cmap=plt.cm.rainbow
        )
        
        plt.title(title)
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Neurons of layer")
        plt.show()
 
    def plot_neuron(self, neuron_through_time, title):
        width = max(16, len(neuron_through_time) / 8)
        height = 4
        plt.figure(figsize=(width, height))
        plt.title(title)
        plt.plot(neuron_through_time)
        plt.xlabel("Time")
        plt.ylabel("Neuron's activation")
        plt.show() 
