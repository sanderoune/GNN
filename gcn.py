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


## 2) GRAPH CONVOLUTIONAL NETWORK

