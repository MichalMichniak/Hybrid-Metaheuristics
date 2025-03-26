import numpy as np
from typing import List

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

    def PSO_to_taboo(self):
        pass

    def taboo_step(self):
        pass

    def mutation():
        pass

    def GA_to_PSO(self, T_mean):
        pass


def PSO(population_lst : List[Instance], M_PSO = 5):
    pass

def Taboo(population_lst : List[Instance], M_Taboo = 5):
    pass

def GA(population_lst : List[Instance], idx : List[int]):
    pass

class Island:
    def __init__(self, population_lst : List[Instance], idx : List[int]):
        self.idx : List[int] = idx # index of instances of that island
        self.population_lst : List[Instance] = population_lst

    def count_mean_transformation(self):
        self.T_mean = np.zeros((N,N), dtype=float)
        pass

    def run(self):
        # wykonanie GA i zamiana na formę PSO
        self.offsprings : List[Instance]= GA(self.population_lst, self.idx, self.population_lst)
        # (tu mogą być problemy z multiprocessing)

        for i in self.idx:
            self.population_lst[i].GA_to_PSO(self.T_mean)
        
        for i in range(len(self.offsprings)):
            self.offsprings[i].GA_to_PSO(self.T_mean)
        self.population_lst.extend(self.offsprings)


def initialization() -> List[Instance]:
    pass

def split_population(population_lst : List[Instance], M_species = 3) -> List[List[int]]:
    pass

def run(M_PSO = 5, M_TABOO = 5, M_species = 3, max_it = 10):
    population_lst : List[Instance] = initialization()
    for i in range(max_it):
        PSO(population_lst, M_PSO)
        Taboo(population_lst, M_TABOO)

        split_idx : List[List[int]] = split_population(population_lst, M_species )

        for i in range(M_species):
            island = Island(population_lst, split_idx[i])
            island.count_mean_transformation()
            island.run()
    print("finished :)")
    


    

if __name__ == '__main__':
    pass