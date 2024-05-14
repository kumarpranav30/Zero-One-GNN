import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx
import random

emb_size = 8

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(emb_size, 128)  # Assuming node features are now a vector of size 5
        self.conv2 = GCNConv(128, 1)  # Output a single feature per node
        self.initialize_weights()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Mean pooling
        return torch.sigmoid(x)  # Sigmoid activation at the output

    def initialize_weights(self):
        # Initialize weights for each GCNConv layer
        for m in self.modules():
            if isinstance(m, GCNConv):
                torch.nn.init.uniform_(m.lin.weight, -1, 1)  # Correct attribute for weight
                if m.lin.bias is not None:
                    torch.nn.init.uniform_(m.lin.bias, -1, 1)

def generate_graph(n, p=0.5):
    G = nx.erdos_renyi_graph(n, p)
    for _, node_data in G.nodes(data=True):
        node_data['x'] = torch.tensor([random.uniform(0,1) for x in range(emb_size)])  # Example feature vector
    data = from_networkx(G)
    # print(data.edge_index)
    return data

iterations = 20
plt.figure(figsize=(10, 6))
for num_itr in range(iterations):
    print(f'ITERATION {num_itr+1}')
    # Simulate for different n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    model.eval()  # Set the model to evaluation mode
    outputs = []
    ns = range(1, 101, 3)
    samples = 15
    for n in ns:
        print(n)
        cnt = 0
        for itr in range(samples):
            if n > 0:
                graph_data = generate_graph(n).to(device)
                output = model(graph_data)
                if output.item() > 0.5: cnt += 1
                # outputs.append(output.item())
        outputs.append(cnt/samples * 100)
        
    # Plot the results
    plt.plot(ns, outputs, label=f'Iteration {num_itr + 1}')
    # plt.plot(range(len(outputs)), outputs, marker = 'o')

plt.grid(True)
plt.title("Testing Zero-One Laws")
plt.xlabel("Number of Nodes (n)")
plt.ylabel("Graphs classified as 1 (%)")
plt.legend()
plt.show()
