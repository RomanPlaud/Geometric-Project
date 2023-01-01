import torch
from torch_geometric.nn import global_mean_pool, GATConv
from torch.nn import Linear
import torch.nn.functional as F


class Net(torch.nn.Module):
  def __init__(self, num_features, num_classes, nhid, concat=False):
    super(Net, self).__init__()

    self.num_features = num_features
    self.num_classes = num_classes
    self.nhid = nhid
    self.concat = concat

    self.conv1 = GATConv(self.num_features, self.nhid * 2)
    self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

    self.fc1 = Linear(self.nhid * 2, self.nhid)

    if self.concat:
      self.fc0 = Linear(self.num_features, self.nhid)
      self.fc1 = Linear(self.nhid * 2, self.nhid)

    self.fc2 = Linear(self.nhid, self.num_classes)


  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch

    x = F.selu(self.conv1(x, edge_index))
    x = F.selu(self.conv2(x, edge_index))
    embedding = x
    x = F.selu(global_mean_pool(x, batch))
    x = F.selu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)

    if self.concat:
      news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
      news = F.relu(self.fc0(news))
      x = torch.cat([x, news], dim=1)
      x = F.relu(self.fc1(x))

    x = F.log_softmax(self.fc2(x), dim=-1)

    return embedding, x

