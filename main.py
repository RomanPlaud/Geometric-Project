import argparse
from utils import get_dataset, get_dataset_v2, get_ids, compute_num_features
from sklearn.model_selection import KFold
import torch
from torch.utils.data import SubsetRandomSampler, Subset
import pickle
import os
import numpy as np
from torch_geometric.data import DataLoader
from model import Net
from train import train
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="K_fold cross_validation")

parser.add_argument('--type_of_training', type=list, default='k_fold')
parser.add_argument('--listes_features', type=list, default = [['random'], ['profile'], ['content'], ['spacy'], ['bert'], ['content', 'spacy'], ['content', 'bert'], ['content', 'spacy', 'bert']])
parser.add_argument('--features', type=list, default=['spacy'])
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--dataset_name', type=str, default='gossipcoc')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--path_mapping', type=str, default= "/content/drive/MyDrive/geometric/gos_id_twitter_mapping.pkl")
parser.add_argument('--path_idx', type=str, default= "/content/drive/MyDrive/geometric")
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('-path_save_results', type=str, default="/content/drive/MyDrive/geometric/results_all_go.pickle")
parser.add_argument('--visualizing_embedding', type=bool, default=False)
parser.add_argument('--name_fig', type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mapping=pickle.load(open(args.path_mapping,'rb'))

train_idx = np.load(os.path.join(args.path_idx, "train_idx.npy"))
val_idx = np.load(os.path.join(args.path_idx, "val_idx.npy"))
test_idx = np.load(os.path.join(args.path_idx, "test_idx.npy"))

all_ids = get_ids(mapping, train_idx, val_idx, test_idx)

if args.type_of_training == 'k_fold':
  results = list()
  results = []

  for liste_features in args.listes_features:

    dataset = get_dataset(liste_features, args.dataset_name, all_ids)
    splits=KFold(n_splits=5,shuffle=True,random_state=42)

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):


      print('Fold {}'.format(fold + 1))

      train_sampler = SubsetRandomSampler(train_idx)
      val_sampler = SubsetRandomSampler(val_idx)

      train_dataset = Subset(dataset, train_sampler)
      test_dataset = Subset(dataset, val_sampler)

      train_loader = DataLoader(dataset, batch_size=args.batch_size) 
      val_loader = DataLoader(dataset, batch_size=args.batch_size)
      model = Net(compute_num_features(liste_features), 2, args.nhid)

      model = model.to(device)
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      result = train(model, optimizer, train_loader, val_loader, args.epochs, printing=True)
      results.append(result)

  # Enregistrement de la liste dans le fichier
  pickle.dump(results, open(args.path_save_results, 'wb'))


if args.type_of_training == 'standard':
  train_dataset, val_dataset = get_dataset_v2(args.features, args.dataset_name, all_ids)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size) #, sampler=train_sampler)
  val_loader = DataLoader(test_dataset, batch_size=args.batch_size) #, sampler=test_sampler)
  model = Net(compute_num_features(args.features), 2, args.nhid)
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  result = train(model, optimizer, train_loader, val_loader, args.epochs, printing=True)
  pickle.dump(result, open(args.path_save_results, 'wb'))

  if args.visualizing_embedding: 
    users = list(set(mapping.values()))
    mapping_inverse = dict(zip(users, range(len(users))))
    embeddings = []
    credibility = np.zeros((len(mapping_inverse), 2))
    for data in tqdm(train_loader):
        data = data.to(device)
        
        embedding, out = model(data)
        y = data.y
        for id in (data.id[0]):
          credibility[mapping_inverse[id], y] = credibility[mapping_inverse[id], y] + 1

        embeddings.append(embedding.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', n_iter=1000).fit_transform(embeddings)
    all_users = (np.concatenate(all_ids[0]+all_ids[1]))
    users_id = [mapping_inverse[user] for user in all_users]
    credibility_clipped = (credibility[:, 1] )/(credibility.sum(axis=1))
    fig, ax = plt.subplots(figsize = (16,10))
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c = credibility_clipped[users_id],cmap ='jet', s = 1)
    plt.colorbar()
    fig.savefig(args.name_fig)
