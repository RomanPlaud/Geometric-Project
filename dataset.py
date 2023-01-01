import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset



class MergedDataset(Dataset):
    def __init__(self, datasets, mapping):
        self.datasets = datasets
        self.mapping = mapping

    def __len__(self):
        # Assume all datasets have the same length
        return len(self.datasets[0])

    def __getitem__(self, index):
        # Retrieve the item from all datasets
        items = [dataset[index].x for dataset in self.datasets]

        # Merge the attributes as needed
        merged_data = Data(x = torch.cat(items, dim = 1), edge_index = self.datasets[0][index].edge_index, y = self.datasets[0][index].y, id = self.mapping[index])

        return merged_data

class RandomDataset(Dataset):
    def __init__(self, dataset, mapping, dim_features = 10):
        self.dataset = dataset
        self.dim_features = dim_features
        self.mapping = mapping

    def __len__(self):
        # Assume all datasets have the same length
        return len(self.dataset)

    def __getitem__(self, index):
        # Retrieve the item from all datasets

        # Merge the attributes as needed
        merged_data = Data(x = torch.randn((self.dataset[index].x.shape[0], self.dim_features)), edge_index = self.dataset[index].edge_index, y = self.dataset[index].y, id=self.mapping[index])

        return merged_data