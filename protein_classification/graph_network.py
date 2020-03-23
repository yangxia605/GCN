# Step 1: install the deep Graph Library
import matplotlib.animation as animation
import  matplotlib.pyplot as plt
import dgl #graph library in Python
import networkx as nx # Visualize the graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Step 2:Creating  a toy graph in DGL
def build_toy_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph;nodes are labeled from 0-33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edges_list = [(1,0),(2,0),(2,1),(3,0),(3,1),(3,2),
                  (4,0),(5,0),(6,0),(6,4),(6,5),(7,0),(7,1),
                  (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0),(10,4),
                  (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                  (13, 3), (15, 5), (16, 6),(17,0), (17, 1), (19, 0), (19, 1),
                  (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                  (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                  (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                  (32, 14), (32, 15), (32, 10), (32, 20), (32, 22), (32, 23),
                  (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                  (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                  (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                  (33, 31), (33, 32)]
    # add edges two list of nodes:arc and dst
    src, dst = tuple(zip(*edges_list))
    print("src===============",src)
    print("dst===============",dst)
    g.add_edges(src,dst)
    # edges are directional in DGL;make them bi-directional
    g.add_edges(dst,src)
    return g
G = build_toy_graph()
print("we have %d nodes." % G.number_of_nodes())
print("we have %d edges." % G.number_of_edges())

# since the actual graph is undirected, we convert it for visualization
# purpose
nx_G = G.to_networkx().to_undirected()
# Kamada-kawaii layout usually looks prety for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G,pos,with_labels=True,node_color=[[.7, .7, .7]])

# Step 3:Assign features to nodes or edges
G.ndata['feat'] = torch.eye(34)
# print out node 2's input feature
print(G.nodes[2].data['feat'])
#print out node 10 and 11's input feature
print(G.nodes[[10,11]].data['feat'])

# Step 4: Define a Graph Convolutional Network(GCN)
# Define the message and reduce function
# NOTE: We ignore the GCN's normalization content c_ij for this tutorial

def gcn_message(edges):
    # The argument is a bath of edges.
    # This computes a(batch of) message called 'msg' using the source node's feature 'h
    return {'msg': edges.src['h']}
def gcn_reduce(nodes):
    # The argument is a bath of nodes.
    # This computes the new 'h' features by summing received 'msg' in each nodes 'mailbox'
    return {'h': torch.sum(nodes.mailbox['msg'],dim=1)}
# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats,out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats,out_feats)

    def forward(self, g, inputs):
        # g is the graph and is the input node features
        # first set the node feature
        g.ndata['h'] = inputs
        # trigger message passing on all edges
        g.send(g.edges(), gcn_message)
        # trigger aggregation at all nodes
        g.recv(g.nodes(), gcn_reduce)
        # get the result node feature
        h = g.ndata.pop('h')
        # perform linear transformation
        return self.linear(h)
# define a deeper GCN model contains two GCN layers:
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h
# The first layer transform input features of size of 34 to a hidden size of 5
# The second layer transform the hidden layer and produces output
# features of size 2, corresponding to the two groups of the protein network

net = GCN(34, 5, 2)

# Step 5: Data Preparation
inputs = torch.eye(34)
labeled_nodes = torch.tensor([0,33])
labels = torch.tensor([0,1])

# Step 6: Train the Graph Network

optimizer = torch.optim.Adam(net.parameters(),lr=0.1)
all_logits = []
for eopch in range(30):
    logits = net(G, inputs)
    # we save the logits for visulization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # We only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (eopch, loss.item()))
# Step 7: Visualize
# def draw (i):
#     cls1color = '#00FFFF'
#     cls2color = '#FF00FF'
#     pos = {}
#     colors = []
#     for v in range(34):
#         pos[v] = all_logits[i][v].numpy()
#         cls = pos[v].argmax()
#         colors.append(cls1color if cls else cls2color)
#     ax.cla()
#     ax.axis('off')
#     ax.set_title('Epoch: %d' % i)
#     nx.draw_networkx(nx_G.to_undirected(), pos, node_color = colors,with_labels=True,node_size = 300,ax=ax)
# fig = plt.figure(dpi=150)
# fig.clf()
# ax = fig.subplots()
# draw(0) # draw the prediction of the first epoch
# plt.close()
last_epoch = all_logits[29].detach().numpy()
predicated_class = np.argmax(last_epoch,axis=1)
color = np.where(predicated_class==0,'c','r')

nx.draw_networkx(nx_G,pos,node_color=color,with_labels=True,node_size=300)