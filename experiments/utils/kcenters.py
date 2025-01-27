import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy

class KCenter():
    def __init__(self, data_X):
        self.X = data_X
        self.points = list(range(len(self.X)))
        #vectorized distance matrix calculator
        self.distance_matrix_1 = np.abs(self.X[:, np.newaxis] - self.X).sum(axis=2)
        diff = self.X[:, np.newaxis] - self.X
        self.distance_matrix_2 = np.linalg.norm(diff, axis=2)

    def kmeans_plus(self, k):
        """
        Input:
            k: number of centers to look for.

        Return:
            C: List of centers selected, from 0 to N-1, according to the data_X vector order and a probability approach.
        """

        dist_matrix = self.distance_matrix_2
        S = deepcopy(self.points)

        if k > len(S):
            return S
        else:
            pick = np.random.choice(S)
            S.remove(pick)
            C = [pick]

        while len(C) < k:
            #print(C)
            distance_list = [min([(dist_matrix[center][point], point) for center in C]) for point in S]
            just_distances = np.array([dist for (dist, point) in distance_list])
            just_points = [point for (dist, point) in distance_list]
            #print(distance_list)
            just_distances = just_distances - max(just_distances)
            prob_p = np.exp(just_distances) / np.sum(np.exp(just_distances)) #softmax function
            point_pick = np.random.choice(just_points, size=1, p=prob_p)[0]
            #print(point_pick)
            #print(max(distance_list))
            S.remove(point_pick)
            C.append(point_pick)
        
        return C