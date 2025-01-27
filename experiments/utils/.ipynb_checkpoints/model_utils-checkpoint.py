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
    if loss_plot_name is not None:
        plt.plot(losses)
        plt.savefig(loss_plot_name)
        plt.close()

def get_metrics(output, label):
    # Flatten the 3D arrays to 1D for metric calculation
    output_flat = output.flatten()
    label_flat = label.flatten()

    # Ensure that the output and label are binary arrays (0s and 1s)
    output_flat = (output_flat > 0).astype(int)
    label_flat = (label_flat > 0).astype(int)

    # Calculate metrics
    p = precision_score(label_flat, output_flat, zero_division=1)
    r = recall_score(label_flat, output_flat, zero_division=1)
    f1 = f1_score(label_flat, output_flat, zero_division=1)
    a = accuracy_score(label_flat, output_flat)

    return p, r, f1, a

def evaluate_function(model, device, criterion, dataloader=None):
    """
    Evaluate the model on a list of cube names and their corresponding targets.

    Parameters:
    - model: The PyTorch model to evaluate.
    - device: The device (CPU/GPU) to run the model on.
    - get_metrics: Function to calculate the precision, recall, F1, and accuracy for each example.
    - dataloader: To load the data/label.

    Returns:
    - avg_p, avg_r, avg_f1, avg_a: Averaged Precision, Recall, F1-score, and Accuracy metrics.
    """
    model.eval()  # Set the model to evaluation mode
    all_results = []  # List to store tuples (seismic, label, prediction)

    total_p, total_r, total_f1, total_a = 0, 0, 0, 0  # Initialize metrics

    total_loss = 0
    
    with torch.no_grad():
        for img, label in dataloader:
            # Convert sis and label to tensors and move to device
            img = img.to(device)
            label = label.to(device)

            # Get model prediction
            output = model(img)
            loss = criterion(prediction = output, target = label)
            total_loss += loss.item()

            # Convert outputs to numpy for evaluation and plotting
            output = output.squeeze().cpu().numpy()
            label = label.squeeze().cpu().numpy()

            threshold = 0.5
            output = output > threshold

            # Calculate metrics for this example
            p, r, f1, a = data_utils.get_metrics(output, label)
            
            # Sum up the metrics for averaging later
            total_p += p
            total_r += r
            total_f1 += f1
            total_a += a
    
    # Average the metrics across all samples
    avg_p = total_p / num_samples
    avg_r = total_r / num_samples
    avg_f1 = total_f1 / num_samples
    avg_a = total_a / num_samples
    avg_loss = total_loss / num_samples

    return avg_p, avg_r, avg_f1, avg_a, avg_loss

def active_learning(list_1, list_2, evaluate_list, batch_size, n_epochs, model, optimizer, data_path, num_of_rounds=100, criterion = nn.BCEWithLogitsLoss()):
    datalist_1 = list_1
    datalist_2 = list_2

    dataloader_evaluate = DataLoader(ArtifactBadge(evaluate_list), 1)

    device = next(model.parameters()).device

    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    loss_list = []
    
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
            p, r, f, a, l = evaluate_function(model, device, criterion, dataloader=dataloader_evaluate)
            precision_list.append(p)
            recall_list.append(r)
            f1_list.append(f)
            accuracy_list.append(a)
            loss_list.append(l)

    return precision_list, recall_list, f1_list, accuracy_list, loss_list
            
            
    
    
