import os
import torch
import numpy as np
from experiments.utils.kcenters import KCenter
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn

class BadgeSelection():
    def __init__(self, dataloader, model, optimizer, B=5, loss_layer_name = None, verbose=False):
        """
        Initialize the BADGE class.

        Parameters:
        model: The model used to process the data.
        dataloader: The dataloader that provides the data, which includes the label.
        B (int): Number of points to select.
        optimizer: optimizer used to train the model.
        criterion: loss used to train the model.
        """
        self.model = model
        self.B = B
        self.device = next(model.parameters()).device
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_layer_name = loss_layer_name
        self.verbose = verbose
        
    def get_latent(self):
        """
        Extracts latent representation from the data using the model.
        For this method, we try to use the loss.

        Returns:
        points (np.array): Array of latent points.
        cube_name (list): List of names for each data point.
        """

        points = []
        image_name = []

        for inp, name in tqdm(self.dataloader):
            
            inp = inp.to(self.device)
        
            for batch in range(inp.shape[0]):
                new_inp = inp[batch].unsqueeze(0)
                #print(new_inp.shape)
                self.model.eval()
                self.optimizer.zero_grad()
        
                # forward pass
                output = self.model(inp)
                #print(output.shape)
                #print(output)
                loss = self.criterion(output, output)
        
                loss.backward()
            
                named_params = dict(self.model.named_parameters())
                if self.loss_layer_name is None:
                    raise ValueError("Especifique uma camada válida para obter a loss")
                layer_name = self.loss_layer_name
                gradients = named_params[layer_name].grad.cpu().numpy()[0]
                #print(gradients.shape)
                pick = np.sort(gradients)[:100]
                points.append(pick)
                image_name.append(name[batch])
                
        return np.array(points), image_name
    
    def select(self):
        """
        Select B points using k-means++ clustering method.

        Returns:
        selected_volumes (list): List of selected data point names.
        """
        list_of_points, list_of_names = self.get_latent()
        clustering = KCenter(list_of_points)
        selected_centers = clustering.kmeans_plus(self.B)
        selected_images = []

        for center in selected_centers:
            selected_images.append(list_of_names[center])

        return selected_images