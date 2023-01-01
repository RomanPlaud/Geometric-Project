from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
from torch.utils.data import ConcatDataset
from dataset import RandomDataset, MergedDataset
import numpy as np
from math import sqrt


def get_ids(mapping, train_idx, val_idx, test_idx):
    ## Function that production the user mapping for the given datasets (train, test, val) knowing how it was initially shuffled thanks to train_idx, val_idx and test_idx

    all_ids=list()
    i=0
    ids = []
    for id in mapping:
        if 'gos' in mapping[id] or (id == len(mapping)-1):
            if len(ids)!=0:
                all_ids.append(ids)
                i+=1
            ids = [mapping[id]]
        else:
            ids.append(mapping[id])

        train_ids = []
        val_ids = []
        test_ids = []

    for idx in train_idx:
        train_ids.append(all_ids[idx])
    for idx in val_idx:
        val_ids.append(all_ids[idx])
    for idx in test_idx:
        test_ids.append(all_ids[idx])


    return [train_ids, val_ids, test_ids]


def get_dataset(liste_features, dataset_name, all_ids, path=''):
  datasets = []
  for feature in liste_features:
    if feature == 'random':
      train_dataset = UPFD(path, dataset_name, 'profile', 'train', ToUndirected())
      val_dataset = UPFD(path, dataset_name, 'profile', 'val', ToUndirected())
      dataset = ConcatDataset([train_dataset, val_dataset]) 
      datasets.append(RandomDataset(dataset))
    else : 
      train_dataset = UPFD(path, dataset_name, feature, 'train', ToUndirected())
      val_dataset = UPFD(path, dataset_name, feature, 'val', ToUndirected())
      dataset = ConcatDataset([train_dataset, val_dataset])
      datasets.append(dataset)
  return MergedDataset(datasets, np.concatenate(all_ids[:2]))

def get_dataset_v2(liste_features, dataset_name, all_ids, path=''):
  datasets_train = []
  datasets_val = []
  for feature in liste_features:
    if feature == 'random':
      train_dataset = UPFD(path, dataset_name, 'profile', 'train', ToUndirected())
      val_dataset = UPFD(path, dataset_name, 'profile', 'val', ToUndirected())
      datasets_train.append(RandomDataset(train_dataset))
      datasets_val.append(RandomDataset(val_dataset))

    else : 
      train_dataset = UPFD(path, dataset_name, feature, 'train', ToUndirected())
      val_dataset = UPFD(path, dataset_name, feature, 'val', ToUndirected())
      datasets_train.append(RandomDataset(train_dataset))
      datasets_val.append(RandomDataset(val_dataset))
  return MergedDataset(datasets_train, all_ids[0]), MergedDataset(datasets_val, all_ids[1])

def compute_num_features(liste_features):
  num_features = { 'random' : 10, 'profile' : 10, 'content': 310, 'spacy' : 300, 'bert' : 768}
  return sum([num_features[feat] for feat in liste_features])


def plot_confidence_interval(ax, x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    mean = np.mean(values, axis =1)
    stdev = np.std(values, axis = 1)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    ax.plot([x, x], [top, bottom], color=color)
    ax.plot([left, right], [top, top], color=color)
    ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(x, mean, 'o', color='#f44336')