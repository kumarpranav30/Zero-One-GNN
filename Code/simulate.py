import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx
import random
import time

node_emb_size = 128

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_emb_size, 128) 
        self.conv2 = GCNConv(128, 32)  # Output a single feature per node
        self.fc1 = torch.nn.Linear(32, 16)  # First MLP layer
        self.fc2 = torch.nn.Linear(16, 1)
        self.initialize_weights()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Mean pooling
        x = torch.tanh(self.fc1(x))  # First MLP layer with tanh
        x = self.fc2(x)  # Second MLP layer
        return torch.sigmoid(x)  # Sigmoid activation at the output

    def initialize_weights(self):
    # Initialize weights for each layer
        for m in self.modules():
            if isinstance(m, (GCNConv, torch.nn.Linear)):
                if hasattr(m, 'lin'):
                    torch.nn.init.uniform_(m.lin.weight, -1, 1)  # Set random weights
                    if m.lin.bias is not None:
                        torch.nn.init.uniform_(m.lin.bias, -1, 1)  # Set random biases
                elif hasattr(m, 'att'):
                    torch.nn.init.uniform_(m.att.weight, -1, 1)  # Set random weights
                    if m.att.bias is not None:
                        torch.nn.init.uniform_(m.att.bias, -1, 1)  # Set random biases

def generate_graph(n, p=0.5):
    G = nx.erdos_renyi_graph(n, p)
    for _, node_data in G.nodes(data=True):
        node_data['x'] = torch.tensor([random.uniform(0,1) for x in range(node_emb_size)])  # Set random node-embeddings
    data = from_networkx(G)
    return data

iterations = 10 # Number of random GNNs to test the law on
plt.figure(figsize=(10, 6))

ns = range(1, 5002, 500)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
for num_itr in range(iterations):
    start_itr = time.time()
    print(f'ITERATION {num_itr+1}')
    model = GCN().to(device)
    model.eval()  # Set the model to evaluation mode
    outputs = []
    samples = 20

    for n in ns:
        print(f'Nodes = {n}', end = ", ")
        cnt = 0
        start_node = time.time()
        for itr in range(samples):
            if n > 0:
                graph_data = generate_graph(n).to(device)
                output = model(graph_data)
                if output.item() >= 0.5: cnt += 1
        outputs.append(cnt/samples * 100)
        print(f"Time taken: {(time.time() - start_node):.2f}s", end = ", ")
        print(f"Output: {outputs[-1]}%")
    end_itr = time.time()
    total_tme = end_itr - start_itr
    print(f"TOTAL TIME: {(total_tme)//60} mins {(total_tme)%60:.2f} seconds")
    # Plot the results
    plt.plot(ns, outputs, label=f'Iteration {num_itr + 1}')

plt.grid(True)
plt.title("Testing Zero-One Laws")
plt.xlabel("Number of Nodes (n)")
plt.ylabel("Graphs classified as 1 (%)")
plt.legend()
plt.show()
