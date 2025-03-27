import numpy as np
from typing import List, Tuple
from copy import deepcopy
N = 20
MIN_COST = np.inf
LONG_TERM_CONST = 2
W = np.random.random((N,N))
D = np.random.random((N,N))*(np.ones((N,N)) - np.eye(N))


class Instance:
    def __init__(self):
        self.perm_matrix = np.zeros((N,N), dtype = float)
        self.best_matrix = np.zeros((N,N), dtype = float)
        self.best_cost = np.inf
        self.velocity_matrix = np.zeros((N,N), dtype = float)
        self.permutation = np.zeros((N,), dtype= int)
        self.PSO_min = np.zeros((N,N), dtype = float) # argument dla minimalnej wartości jaką osiągnęła dana instancja
        self.taboo_lst = []
        self.taboo_size = 10
        self.taboo_longterm_lst = np.zeros((N,N-1)) # convention smaler first

        ### PSO ###
        self.omega = 0.8
        self.c_1 = 2.8
        self.c_2 = 2.8
        self.B1 = 5
        self.B2 = 5
        ### END PSO ###

        ### GA -> PSO ###
        self.c_1_GA_PSO = 0.5
        self.c_2_GA_PSO = 0.5
        pass

    def PSO_step(self, p_d):
        #update pojedynczej cząstki
        self.velocity_matrix = self.omega*self.velocity_matrix + np.random.normal(0.2)*self.c_1*(self.best_matrix - self.perm_matrix) + np.random.rand()*self.c_2*(p_d - self.perm_matrix)
        self.perm_matrix = np.minimum(1,np.maximum(0, self.perm_matrix + self.velocity_matrix))
        pass
    
    def inverse_permutation(self,perm):
        temp = np.array(list(range(len(perm)))).reshape((1,len(perm)))
        temp2 = perm.reshape((1,len(perm)))
        temp3 = np.concatenate((temp2,temp),axis=0)
        return temp3[:,np.argsort(temp3[0,:])][1,:]


    def PSO_QAP_cost(self):
        A = W@(self.perm_matrix.T)@D
        sum = 0
        for i in range(len(self.permutation)):
            sum+=A[i][self.permutation[i]]
        return sum

    def PSO_full_QAP_cost(self):
        A = W@(self.perm_matrix.T)@D@(self.perm_matrix)
        
        return np.trace(A)

    def real_QAP_cost(self):
        W_prim = np.zeros((N,N))
        for j in range(N):
            W_prim[:,self.permutation[j]] = W[:,j]
            pass
        A = W_prim@D
        sum = 0
        for i in range(len(self.permutation)):
            sum+=A[i][self.permutation[i]]
        return sum

    def fuzzy_matrix_to_permutation(self):
        idx_col = list(range(N))
        idx_row = list(range(N))
        self.permutation = np.zeros((N,), dtype= int)
        
        for i in range(N):
            curr_max = 0
            curr_x = 0
            curr_y = 0
            for j in range(len(idx_col)):
                for k in range(len(idx_row)):
                    if(curr_max<self.perm_matrix[idx_row[k]][idx_col[j]]):
                        curr_max = self.perm_matrix[idx_row[k]][idx_col[j]]
                        curr_x = j
                        curr_y = k
            self.permutation[idx_row[curr_y]] = idx_col[curr_x]
            del idx_col[curr_x]
            del idx_row[curr_y]
    
    def penality(self):
        return self.B1*np.sum((np.sum(self.perm_matrix,axis=1)-1)**2) + self.B2*np.sum((np.sum(self.perm_matrix,axis=0)-1)**2)

    

    def PSO_to_taboo(self):
        self.fuzzy_matrix_to_permutation()
        for i in range(self.taboo_size):
            ind = np.unravel_index(np.argmax(self.perm_matrix, axis=None), self.perm_matrix.shape)
            self.perm_matrix[ind[0],ind[1]] = 0
            idx2 = np.argmax(self.perm_matrix[ind[0]], axis=None)
            # odtwarzanie listy taboo
            if(ind[1]<idx2):
                self.taboo_lst.append((ind[1],idx2))
            elif(ind[1]>idx2):
                self.taboo_lst.append((idx2,ind[1]))
            # odtwarzanie długoterminowej
            self.taboo_longterm_lst = np.zeros((N,N-1))
            for n1 in range(N):
                for n2 in range(N-1):
                    self.taboo_longterm_lst[n1,n2] = LONG_TERM_CONST * min(self.velocity_matrix[n1,n2],self.velocity_matrix[n2,n1])
        pass

    def taboo_step(self):
        pass

    def mutation():
        pass

    def GA_to_PSO(self, V_mean):
        self.velocity_matrix = self.c_1_GA_PSO*V_mean + self.c_2_GA_PSO*np.random.normal(0,1)
        self.perm_matrix=np.zeros((N,N))
        for i in range(N):
            self.perm_matrix[i,self.permutation[i]] = 1
        pass
    
    def transposition_to_matrix(self, transp : Tuple[int]): # Można szybciej
        T = np.eye(N)
        T[transp[0],transp[0]] = 0
        T[transp[1],transp[1]] = 0
        T[transp[1],transp[0]] = 1
        T[transp[0],transp[1]] = 1
        return T



    def V_prev(self, V_mean):
        transformation = np.eye(N)
        for i in list(range(len(self.taboo_lst)))[::-1]:
            transformation = transformation@self.transposition_to_matrix(self.taboo_lst[i])
        X=np.zeros((N,N))
        for i in range(N):
            X[i,self.permutation[i]] = 1
        X_prev = X@transformation.T
        return X-X_prev

def PSO(population_lst : List[Instance], M_PSO = 5):
    p_d = population_lst[0].best_matrix    # TODO: lepsza inicializacja najlepszego
    best_cost = np.inf
    for i in range(len(population_lst)):
        population_lst[i].fuzzy_matrix_to_permutation()
        cost1 = population_lst[i].PSO_full_QAP_cost()
        cost2 = population_lst[i].PSO_QAP_cost()
        cost3 = population_lst[i].real_QAP_cost()
        
        penality = population_lst[i].penality()
        cost = cost3 + penality #cost1+penality #TODO: chose cost func
        if(best_cost>cost):
            best_cost = cost
            p_d = deepcopy(population_lst[i].perm_matrix)
            print(f"inst: {i}, PSO {cost1}, PSO_half {cost2}, real {cost3}, cost {cost}")

    for it in range(M_PSO):
        for i in range(len(population_lst)):
            population_lst[i].PSO_step(p_d)

        
        for i in range(len(population_lst)):
            population_lst[i].fuzzy_matrix_to_permutation()
            cost1 = population_lst[i].PSO_full_QAP_cost()
            cost2 = population_lst[i].PSO_QAP_cost()
            cost3 = population_lst[i].real_QAP_cost()
            
            penality = population_lst[i].penality()
            cost = cost3 + penality #cost1+penality #TODO: chose cost func
            if(best_cost>cost):
                best_cost = cost
                p_d = deepcopy(population_lst[i].perm_matrix)
                print(f"inst: {i}, PSO {cost1}, PSO_half {cost2}, real {cost3}, cost {cost}")
            if(population_lst[i].best_cost>cost):
                population_lst[i].best_cost = cost
                population_lst[i].best_matrix = deepcopy(population_lst[i].perm_matrix)
    print("PSO -> Taboo")
    for i in range(len(population_lst)):
        population_lst[i].PSO_to_taboo()
    pass

def initialization(M_start = 100) -> List[Instance]:
    instance_lst = [Instance() for i in range(M_start)]
    for i in range(len(instance_lst)):
        instance_lst[i].perm_matrix = np.random.rand(N,N)
        instance_lst[i].velocity_matrix = np.random.rand(N,N)
    return instance_lst





def test1():
    ins = Instance()
    # ins.perm_matrix = np.array([[0.25,0.5,0],
    #                             [0,0.1,0.9],
    #                             [0.5,0.4,0.1]],dtype=float)
    ins.perm_matrix = np.random.random((N,N))
    ins.perm_matrix = (N)* ins.perm_matrix/np.sum(ins.perm_matrix)
    ins.fuzzy_matrix_to_permutation()
    # print(ins.permutation)
    # ins.inverse_permutation(np.array([0,2,5,1,6,3,4]))
    W = np.random.random((N,N))
    D = np.random.random((N,N))*(np.ones((N,N)) - np.eye(N))
    print(ins.real_QAP_cost())
    print(ins.PSO_QAP_cost())
    print(ins.PSO_full_QAP_cost())

def test2():
    instance_lst = initialization(M_start = 100)
    PSO(instance_lst, M_PSO = 50)
    

if __name__ == '__main__':
    test2()
    # test1()

    
    
