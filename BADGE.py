# imports

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from kcenters import KCenter
from tqdm import tqdm
#import dataset here
# import Dataset....
from torch.optim import Adam
import torch.nn.BCELoss as BCE

class BadgeSelection():
    def __init__(self, data_path, model, optimizer, criterion, B=5, loss_layer_name = None, verbose=False):
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
        self.device = next(model.parameters()).device
        # Declare the dataset here:
        # ds = Dataset( Dataset Parameters)
        
        # Declare the dataloader here:
        self.dataloader = DataLoader(ds, self.B, shuffle=True, drop_last = False)
        self.optimizer = optimizer
        self.criterion = BCE()
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
        cube_name = []

        for inp, name in tqdm(self.dataloader, disable = not self.verbose):

            self.model.eval()
            inp = inp.to(self.device)

            for i in range(inp.shape[0]):
                batch_inp = inp[i].unsqueeze(0)

                self.optimizer.zero_grad()

                #forwarding

                out = self.model(batch_inp)
                loss = self.criterion(out, out)

                #backwards
                loss.backward()

                #pick loss gradients

                layer_name = self.loss_layer_name
                if layer_name is None:
                    raise ValueError("Erro: O nome da layer para obter a loss não foi especificado. Por favor, especifique um válido!")
                named_params = dict(self.model.named_parameters())
                weights = named_params[layer_name].grad.cpu()
                weights = np.abs(weights)
                weights_shape = weights.shape
                #print(weights_shape)
                weights = weights.sum(dim=(-1, -2)) / (weights_shape[-1] * weights_shape[-2])
                weights = weights.flatten().numpy()
                points.append(weights)
                cube_name.append(name[i])
        
        return np.array(points), cube_name
    
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