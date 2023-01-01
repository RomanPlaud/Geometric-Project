import time
import torch
from tqdm import tqdm
import torch.nn.functional as F

from eval_functions import eval_deep, compute_test

def train(model, train_loader, val_loader, epochs=50, printing = True):
  device = model.device()
  t = time.time()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
  model.train()
  for epoch in tqdm(range(epochs)):
    out_log = []
    loss_train = 0.0
    for i, data in enumerate(train_loader):
      optimizer.zero_grad()
      data = data.to(device)
      _, out = model(data)
      y = data.y
      loss = F.nll_loss(out, y)
      loss.backward()
      optimizer.step()
      # scheduler.step()
      loss_train += loss.item()
      out_log.append([F.softmax(out, dim=1), y])
    _,_, acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
    [_,_, acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader, model)
    if printing : 
      print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
          f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
          f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
          f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')
  return compute_test(val_loader, model, verbose=False)


