import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from experiments.dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
from experiments.selector.BADGE import *

def train_function(model, train_dataloader, n_epochs, criterion, optimizer, device, hide_bar = True, loss_plot_name = None):
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        with tqdm(train_dataloader, disable = hide_bar) as tepoch:
            for data, label in tepoch:
                tepoch.set_description(f'Epoch: {epoch}')

                model.train()
                optimizer.zero_grad()
                
                data = data.to(device)
                #print(label)
                target = label.to(device)
                #print(data.shape, label.shape)

                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                tepoch.set_postfix(loss=epoch_loss)
        losses.append(epoch_loss/len(train_dataloader.dataset))
        '''if epoch%5 == 0:
            torch.save(model.state_dict(), f"/pgeoprj/godeep/fjqv/ckpt/AL1/CBAMUnetDiverseAL_ckpt_{epoch+1}.pt")'''
    if loss_plot_name is not None:
        plt.plot(losses)
        plt.savefig(loss_plot_name)
        plt.close()

#TO DO
def evaluate_function():
    pass

def active_learning(list_1, list_2, batch_size, n_epochs, model, optimizer, data_path, num_of_rounds=100, criterion = nn.BCEWithLogitsLoss()):
    datalist_1 = list_1
    datalist_2 = list_2

    device = next(model.parameters()).device

    #TO DO:
    # metric_1_list = []
    # metric_2_list = []
    # metric_3_list = []
    # ...
    
    for i in range(num_of_rounds):
        dataloader_1 = DataLoader(ArtiFactBadge(datalist_1), batch_size)
        dataloader_2 = DataLoader(ArtiFactBadge(datalist_2), batch_size)
        
        if i == 0:
            selector = BadgeSelection(dataloader2, model, optimizer, batch_size, "fc.weight")
            pick = selector.select()
            pick = [os.path.join(data_path, img_name) for img_name in pick]
            for element in pick:
                datalist_1.append(element)
                datalist_2.remove(element)
        else:
            train_function(model, dataloader_1, n_epochs, criterion, optimizer, device)
            selector = BadgeSelection(dataloader2, model, optimizer, batch_size, "fc.weight")
            pick = selector.select()
            pick = [os.path.join(data_path, img_name) for img_name in pick]
            for element in pick:
                datalist_1.append(element)
                datalist_2.remove(element)
            #TO DO:
            #metric_1, metric_2, metric_3 ... = evaluate_function()
            #metric_1_list.append(metric_1)
            #metric_2_list.append(metric_2)
            #metric_3_list.append(metric_3)
            #...
    return #PUT ALL METRIC LISTS HERE!
            
            
    
    