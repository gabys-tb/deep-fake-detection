import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

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
                print(label)
                label = torch.tensor(label).to(device)
                #print(data.shape, label.shape)

                output = model(data)
                loss = criterion(prediction = output, target = label)
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

