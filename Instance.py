import numpy as np
from typing import List
from copy import deepcopy
N = 10
MIN_COST = np.inf

W,D = np.zeros((N,N)), np.zeros((N,N))



class Instance:
    def __init__(self):
        self.perm_matrix = np.zeros((N,N), dtype = float)
        self.permutation = np.zeros((N,), dtype= int)
        self.PSO_min = np.zeros((N,N), dtype = float) # argument dla minimalnej wartości jaką osiągnęła dana instancja
        self.taboo_lst = []
        self.taboo_size = 10
        self.taboo_longterm_lst = np.zeros((N,N-1)) # convention smaler first
        pass

    def PSO_step(self, p_d):
        #update pojedynczej cząstki
        pass

    def fuzzy_matrix_to_permutation(self):
        temp = deepcopy()
        for i in range(N):

        pass

    def PSO_to_taboo(self):
        pass

    def taboo_step(self):
        pass

    def mutation():
        pass

    def GA_to_PSO(self, T_mean):
        pass