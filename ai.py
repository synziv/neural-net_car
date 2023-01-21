

from ctypes import sizeof
import torch
from fastbook import *
import copy


class AI:
    def __init__(self, input_size, hidden_size, output_size, simple_net=None):
        if(simple_net == None):
            self.simple_net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        else:
            self.simple_net = copy.deepcopy(simple_net)
        #print(self.simple_net)    


        self.simple_net.apply(self.init_weights)
        #w,b = self.simple_net[0].parameters()
       

# then applies the desired changes to the weights
    def init_weights(self, m):
        #remove this to have same initial weights every time
        torch.manual_seed(time.time() * random.randint(0, 100000))
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


# use the modules apply function to recursively apply the initialization
    

    def read_outputs(self, inputs):
        #print(len(inputs))
        data = inputs.type(torch.FloatTensor)
        outputMatrices = self.simple_net(data)

        max_idx = torch.argmax(outputMatrices)

        return max_idx
    def mutate(self, mutation_rate):
        for layer in self.simple_net:
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    shape = layer.weight.shape
                    mutation_matrix =  np.random.uniform(-1,1,shape) * np.random.choice([0, 1], size=shape, p=[1-mutation_rate, mutation_rate])
                    #print(layer.weight.data[0])
                    #print("------------------")
                    #modified_weights = layer.weight + torch.from_numpy(mutation_matrix).float()
                    modified_weights = np.where(((mutation_matrix !=0)), mutation_matrix, layer.weight.data)
                    #print(modified_weights[0])
                    layer.weight = nn.Parameter(torch.tensor(modified_weights).float())
                    
                    shape = layer.bias.shape
                    mutation_matrix =  np.random.uniform(-1,1,shape) * np.random.choice([0, 1], size=shape, p=[1-mutation_rate, mutation_rate])
                    modified_bias = np.where(((mutation_matrix !=0)), mutation_matrix, layer.bias.data)
                    layer.bias = nn.Parameter(torch.tensor(modified_bias).float())
                #print("mutation")
                #layer.weight = self.simple_net[0].weight + mutation_matrix

    def crossover(self, partner):
        #take a random point in the weights and split the weights of the two parents
        for i, layer in enumerate(self.simple_net):
            if isinstance(layer, nn.Linear):
                middle_point = np.random.randint(0, len(layer.weight))
                # print(layer.weight[middle_point:])
                # print("middle point", middle_point)
                # print("Start:",len(layer.weight[:middle_point]))
                # print("End:",len(layer.weight[middle_point:]))
                layer.weight = nn.Parameter(torch.cat((layer.weight[:middle_point], partner.simple_net[i].weight[middle_point:])).float())
                middle_point = np.random.randint(0, len(layer.bias))
                layer.bias = nn.Parameter(torch.cat((layer.bias[:middle_point], partner.simple_net[i].bias[middle_point:])))
            
        
    

def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

def sigmoid(x): return 1/(1+torch.exp(-x))
