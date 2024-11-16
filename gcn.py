# Graph Convolutional Networks (GCN)
# Framework: PyTorch Geometric

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import KarateClub

## 1) GRAPH DATA INSPECTION

# Import graph dataset from PyTorch Geometric
# KarateClub dataset: relationships formed within a karae club. It is a kind of social network.
# Nodes -> club member; Edges -> ineractions occurred outside club environment
# In this scenario, member are split in 4 groups. 
# Task -> assign each member to the correct group (node classificaion) based on the interactions.
dataset = KarateClub()

# Print information
print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}') #features per node
print(f'Number of classes: {dataset.num_classes}') #class of each node (our 4 groups)

# Dataset is a clooection of data (graph) objects.
# Print first element
# x (node feature matrix)-> We have 34 nodes ,each with 34 features.
# edge_index (graph connectivity) -> is the graph connectivity (how nodes are connected) with
# shape (2, number of directed edges). 
# y -> is the node ground-trouth label. In this case, each node is a class (group),
# so we have one vlue per node.
# train_mask -> optional attribute. Tells which node should ne for training with a list of True, False. 
print(f'Graph: {dataset[0]}') 

data = dataset[0]

print(f'x = {data.x.shape}')
print(data.x)

# The edge_index is one such data structure, 
# where the graph’s connections are stored in two lists (156 directed edges, which equate to 78 bidirectional edges). 
# The reason for these two lists is that one list stores the source nodes, 
# while the second one identifies the destination nodes.
# A more intuitive manner to represent graph is through adjacency matrix.
# In this each element A_ij specifies the presence or absence of an edge from node i to node j in the graph.
print(f'edge_index = {data.edge_index.shape}')
print(data.edge_index)

from torch_geometric.utils import to_dense_adj

# The adjacency matrix can be inferred from the edge_index with a utility function to_dense_adj().
# We see the adjacency matrix is sparse (more efficient to store in COO format).
A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print(f'A = {A.shape}')
print(A)

# Ground-routh labels of groups (0,1,2,3).
print(f'y = {data.y.shape}')
print(data.y)

# Nodes that should be used for training. The remaining are the test set.
print(f'train_mask = {data.train_mask.shape}')
print(data.train_mask)

# Other utility functions of the Data object.
# is_directed() -> tells if graph is directed (if it is, adjacecy matrix is not symmetric).
print(f'Edges are directed: {data.is_directed()}') 
# has_isolated_nodes() -> if some nodes are no connected to the rest of the graph (these nodes might be challenging for classification).
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
# has_self_loops() -> if graph has at least one node connected to itself.
# This is differen from a loop (a path that starts and ends at the same node, traversing other nodes in between). 
print(f'Graph has loops: {data.has_self_loops()}')

# to_networkx -> uses library NetworkX, useful to visualize graphs with matplotlib.
from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12,12))
plt.axis('off')
nx.draw_networkx(G,
                pos=nx.spring_layout(G, seed=0),
                with_labels=True,
                node_size=800,
                node_color=data.y,
                cmap="hsv",
                vmin=-2,
                vmax=3,
                width=0.8,
                edge_color="grey",
                font_size=14
                )
plt.show()


## 2) GRAPH CONVOLUTIONAL NETWORK (GCN) INTRODUCTION

# With graph data, the connections between nodes mustbe accounted for.
#  typically, in networks, it’s assumed that similar nodes are more likely to be linked 
# to each other than dissimilar ones, a phenomenon known as network homophily.

# Unlike filters in Convolutional Neural Networks (CNNs), 
# our weight matrix W is unique and shared among every node. 
# But there is another issue: nodes do not have a fixed number of neighbors like pixels do.

# To ensure a similar range of values for all nodes and comparability between them, 
# we can normalize the result based on the degree of nodes, 
# where degree refers to the number of connections a node has.
# Features from nodes with numerous neighbors propagate much more easily than those from more isolated nodes. 
# To offset this effect, they suggested assigning bigger weights to features from nodes with fewer neighbors, 
# thus balancing the influence across all nodes. 

## 3) IMPLEMENTING THE GCN

# Graph Convolutional Layer -> implemented through GCNConv function from PyTorch Geometric.

from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(dataset.num_features, 3)
        self.out = Linear(3, dataset.num_classes) # yields 4 values, one for each class

    def forward(self, x, edge_index):
        h = self.gcn(x, edge_index).relu()
        z = self.out(h)
        return h, z

model = GCN()
print(model)

# If we added a second GCN layer, 
# our model would not only aggregate feature vectors from the neighbors of each node, 
# but also from the neighbors of these neighbors.

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

# Data for animations
embeddings = []
losses = []
accuracies = []
outputs = []

# Training loop
for epoch in range(201):
    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    h, z = model(data.x, data.edge_index)

    # Calculate loss function
    loss = criterion(z, data.y)

    # Calculate accuracy
    acc = accuracy(z.argmax(dim=1), data.y)

    # Compute gradients
    loss.backward()

    # Tune parameters
    optimizer.step()

    # Store data for animations
    embeddings.append(h)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(z.argmax(dim=1))

    # Print metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')



from IPython.display import HTML
from IPython.display import display
from matplotlib import animation
'''
plt.rcParams["animation.bitrate"] = 3000

def animate(i):
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G,
                    pos=nx.spring_layout(G, seed=0),
                    with_labels=True,
                    node_size=800,
                    node_color=outputs[i],
                    cmap="hsv",
                    vmin=-2,
                    vmax=3,
                    width=0.8,
                    edge_color="grey",
                    font_size=14
                    )
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=20)

fig = plt.figure(figsize=(12, 12))
plt.axis('off')

anim = animation.FuncAnimation(fig, animate, \
            np.arange(0, 200, 10), interval=500, repeat=True)
html = HTML(anim.to_html5_video())
display(html)
'''
# By aggregating features from neighboring nodes, 
# the GNN learns a vector representation (or embedding) of every node in the network.
# However, embeddings are the real products of GNNs. 

# Print embeddings
print(f'Final embeddings = {h.shape}')
print(h)

# As you can see, embeddings do not need to have the same dimensions as feature vectors. 
# Here, I chose to reduce the number of dimensions from 34 (dataset.num_features) to three to get a nice visualization in 3D.

# Get first embedding at epoch = 0
embed = h.detach().cpu().numpy()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.patch.set_alpha(0)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)

plt.show()

# We see every node from Zachary’s karate club with their true labels (and not the model’s predictions). 
# For now, they’re all over the place since the GNN is not trained yet. 
# But if we plot these embeddings at each step of the training loop, 
# we’d be able to visualize what the GNN truly learns.

def animate(i):
    embed = embeddings[i].detach().cpu().numpy()
    ax.clear()
    ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, c=data.y, cmap="hsv", vmin=-2, vmax=3)
    plt.title(f'Epoch {i} | Loss: {losses[i]:.2f} | Acc: {accuracies[i]*100:.2f}%',
              fontsize=18, pad=40)

fig = plt.figure(figsize=(12, 12))
plt.axis('off')
ax = fig.add_subplot(projection='3d')
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

anim = animation.FuncAnimation(fig, animate, \
              np.arange(0, 200, 10), interval=800, repeat=True)
html = HTML(anim.to_html5_video())

display(html)

# Our Graph Convolutional Network (GCN) has effectively learned embeddings that group similar nodes into distinct clusters. 
# This enables the final linear layer to distinguish them into separate classes with ease.