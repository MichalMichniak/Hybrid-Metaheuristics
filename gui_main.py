import numpy as np
from typing import List
import numpy as np
from typing import List, Tuple
from copy import deepcopy
import random
import sys
import time
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QCoreApplication, QMetaObject, QObject, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QTextCursor
from ui import Ui_MainWindow
from constants import *

# N = 50
# TABOO_NEIGHBORS = 30
# global MIN_COST
MIN_COST = np.inf
BEST_PERM = []
# LONG_TERM_CONST = 2
W = np.random.random((N,N))
D = np.random.random((N,N))*(np.ones((N,N)) - np.eye(N))
D = (D+D.T)/2
# SURV_PART = 0.6
# MUTATION_PROB = 0.3
        
        
class Instance:
    def __init__(self):
        self.perm_matrix = np.zeros((N,N), dtype = float)
        self.best_matrix = np.zeros((N,N), dtype = float)
        self.best_cost = np.inf
        self.velocity_matrix = np.zeros((N,N), dtype = float)
        self.permutation = np.zeros((N,), dtype= int)

        self.permutation_prev_PSO = np.zeros((N,), dtype= int)
        self.counter = 0
        self.max_counter = 5

        self.PSO_min = np.zeros((N,N), dtype = float) # argument dla minimalnej wartości jaką osiągnęła dana instancja
        self.taboo_lst = []
        self.taboo_size = 10
        self.taboo_longterm_lst = np.zeros((N,N-1)) # convention smaler first

        ### PSO ###
        self.omega = OMEGA
        self.c_1 = C_1
        self.c_2 = C_2
        self.B1 = B1
        self.B2 = B2
        ### END PSO ###

        ### islands ###

        self.histogram = np.zeros((N,), dtype=int)

        ### end islands ###


        ### GA -> PSO ###
        self.c_1_GA_PSO = C_1_GA_PSO
        self.c_2_GA_PSO = C_2_GA_PSO

        pass
    
    def make_hist(self):
        self.histogram = np.zeros((N,), dtype=int)
        for i in range(len(self.taboo_lst)):
            a,b = self.taboo_lst[i]
            self.histogram[a] += 1
            self.histogram[b] += 1
            



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

    def taboo_QAP_cost(self, solution=None):
        if solution is None:
            solution = self.permutation
        total_cost = 0
        for i in range(N):
            for j in range(N):
                total_cost += W[i][j] * D[solution[i]][solution[j]]
        return total_cost

    def taboo_step(self):
        best_neighbor_cost = np.inf
        for i in range(TABOO_NEIGHBORS): # częściowe przeszukanie sąsiedztwa
            neighbor = deepcopy(self.permutation)
            a, b = random.sample(range(1, N+1), 2) # wybór dwóch indeksów losowych elementów do zamiany
            move = sorted((a - 1, b - 1))
            if move not in self.taboo_lst:
                neighbor[a - 1], neighbor[b - 1] = neighbor[b - 1], neighbor[a - 1]
                neighbor_cost = self.taboo_QAP_cost(neighbor)
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor_cost = neighbor_cost
                    self.taboo_lst.append(move)
                    self.permutation = neighbor
                    if len(self.taboo_lst) > self.taboo_size:
                        self.taboo_lst.pop(0)

                # if len(self.taboo_lst) != 10:
                #     print(len(self.taboo_lst))

        return best_neighbor_cost
        pass

    def mutation(self):
        a, b = random.sample(range(0, N), 2)
        move = sorted((a, b))
        max_it = self.taboo_size
        it = 0
        while(it < max_it and move in self.taboo_lst ):
            a, b = random.sample(range(0, N), 2)
            move = sorted((a, b))
            it+=1
        self.permutation[a], self.permutation[b] = self.permutation[b], self.permutation[a]
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



    def V_prev(self):
        transformation = np.eye(N)
        for i in list(range(len(self.taboo_lst)))[::-1]:
            transformation = transformation@self.transposition_to_matrix(self.taboo_lst[i])
        X=np.zeros((N,N))
        for i in range(N):
            X[i,self.permutation[i]] = 1
        X_prev = X@transformation.T
        return X-X_prev

def PSO(callback, population_lst : List[Instance], M_PSO = 5):
    p_d = population_lst[0].best_matrix    # TODO: lepsza inicializacja najlepszego
    best_cost = np.inf
    for i in range(len(population_lst)):
        population_lst[i].fuzzy_matrix_to_permutation()

        population_lst[i].permutation_prev_PSO = population_lst[i].permutation
        population_lst[i].counter = 0

        cost1 = population_lst[i].PSO_full_QAP_cost()
        cost2 = population_lst[i].PSO_QAP_cost()
        cost3 = population_lst[i].real_QAP_cost()
        
        penality = population_lst[i].penality()
        cost = cost3# + penality #cost1+penality #TODO: chose cost func
        global MIN_COST, BEST_PERM
        if cost < MIN_COST:
            MIN_COST = cost
            BEST_PERM = population_lst[i].permutation

        if(best_cost>cost):
            best_cost = cost
            p_d = deepcopy(population_lst[i].perm_matrix)
            best_cost = cost
    print(f"best PSO input: {best_cost}")

    for it in range(M_PSO):
        iteration_best = np.inf
        print(f"start iteration PSO{it}")   
        for i in range(len(population_lst)):
            population_lst[i].PSO_step(p_d)
        for i in range(len(population_lst)):
            population_lst[i].fuzzy_matrix_to_permutation()
            if ((population_lst[i].permutation_prev_PSO == population_lst[i].permutation).all()):
                population_lst[i].counter += 1
                if population_lst[i].counter > population_lst[i].max_counter:
                    population_lst[i].counter = 0
                    # usun najwieksze
                    curr_max = -np.inf
                    curr_min = np.inf 
                    curr_x = 0
                    curr_y = 0
                    for j in range(N):
                        for k in range(N):
                            if(curr_max<population_lst[i].perm_matrix[k][j]):
                                curr_max = population_lst[i].perm_matrix[k][j]
                                curr_x = j
                                curr_y = k
                            if(curr_min>population_lst[i].perm_matrix[k][j]):
                                curr_min = population_lst[i].perm_matrix[k][j]
                    population_lst[i].perm_matrix[curr_y][curr_x] = curr_min - 0.00001
            else:
                population_lst[i].permutation_prev_PSO = population_lst[i].permutation
            cost1 = population_lst[i].PSO_full_QAP_cost()
            cost2 = population_lst[i].PSO_QAP_cost()
            cost3 = population_lst[i].real_QAP_cost()
            
            penality = population_lst[i].penality()
            cost_main = cost3# + penality #cost1+penality #TODO: chose cost func
            cost = cost3 + penality
            if(best_cost>cost_main):
                best_cost = cost_main
                p_d = deepcopy(population_lst[i].perm_matrix)
                print(f"inst: {i}, PSO {cost1}, PSO_half {cost2}, real {cost3}, cost {cost}")
            if(population_lst[i].best_cost>cost):
                population_lst[i].best_cost = cost
                population_lst[i].best_matrix = deepcopy(population_lst[i].perm_matrix)
            # global MIN_COST
            if cost < MIN_COST:
                MIN_COST = cost
                BEST_PERM = population_lst[i].permutation
            if cost3 < iteration_best:
                iteration_best = cost3
        callback([iteration_best])
    print("PSO -> Taboo")
    for i in range(len(population_lst)):
        population_lst[i].PSO_to_taboo()
    pass

def Taboo(callback, population_lst : List[Instance], M_Taboo = 5):
    best_global_cost = np.inf
    for i in range(len(population_lst)):
        instance_cost = population_lst[i].taboo_QAP_cost()
        if instance_cost < best_global_cost:
            best_global_cost = instance_cost
    for it in range(M_Taboo):
        iteration_best = np.inf
        print(f"start iteration Taboo: {it}")
        for i in range(len(population_lst)):
            best_local_cost = population_lst[i].taboo_step()
            if best_local_cost < best_global_cost:
                best_global_cost = best_local_cost
                print(f"inst: {i}, Taboo {best_global_cost}")
            global MIN_COST, BEST_PERM
            if best_local_cost < MIN_COST:
                MIN_COST = best_local_cost
                BEST_PERM = population_lst[i].permutation
            if best_local_cost < iteration_best:
                iteration_best = best_local_cost
        callback([iteration_best])
    min_cost = np.inf
    for inst in population_lst:
        temp = inst.taboo_QAP_cost()
        if(temp < min_cost):
            min_cost = temp
    print(f"minimalny koszt po taboo: {min_cost}")
    pass

def PMX(inst1 : Instance, inst2 : Instance):
    new = Instance()
    p1 = inst1.permutation
    p2 = inst2.permutation
    offspring = np.zeros(len(p1), dtype=p1.dtype)
    cutoff_1, cutoff_2 = np.sort(np.random.choice(np.arange(len(p1)+1), size=2, replace=False))
    offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]
    for i in np.concatenate([np.arange(0,cutoff_1), np.arange(cutoff_2,len(p1))]):
        candidate = p2[i]
        while candidate in p1[cutoff_1:cutoff_2]: # allows for several successive mappings
            candidate = p2[np.where(p1 == candidate)[0][0]]
        offspring[i] = candidate
    new.permutation = offspring
    return new

def GA(population_lst : List[Instance], idx : List[int]):
    if len(idx) <3:
        return [], idx
    # count cost
    costs = []
    for i in idx:
        costs.append(population_lst[i].real_QAP_cost())
    costs = np.max(np.array(costs)) - np.array(costs)+0.001 #epsilon 
    # selection
    NO_SURV = int(np.ceil(float(len(idx))*SURV_PART))
    NO_OFFSPRING = len(idx)-NO_SURV

    #rulette
    probability = costs/np.sum(costs)
    surv = np.random.choice(len(idx), NO_SURV, replace=False, p=probability)
    surv = np.array([idx[i] for i in surv])

    offsprings = []
    # offspring
    for i in range(NO_OFFSPRING):
        t = np.random.choice(len(idx), 2, replace=False, p=probability)
        offsprings.append(PMX(population_lst[t[0]], population_lst[t[1]]))

    # mutation of survivors
    for i in surv:
        if random.random() < MUTATION_PROB:
            population_lst[i].mutation()

    return offsprings, surv
    pass

class Island:
    def __init__(self, population_lst : List[Instance], idx : List[int]):
        self.idx : List[int] = idx # index of instances of that island
        self.population_lst : List[Instance] = population_lst

    def count_mean_transformation(self):
        self.V_mean = np.zeros((N,N), dtype=float)
        for i in range(len(self.idx)):
            self.V_mean = self.V_mean + self.population_lst[self.idx[i]].V_prev()
        self.V_mean = self.V_mean/len(self.idx)
        pass

    def run(self, callback):
        # wykonanie GA i zamiana na formę PSO
        self.offsprings, self.survivors = GA(self.population_lst, self.idx)# both : List[Instance]
        # (tu mogą być problemy z multiprocessing)
        island_best = np.inf
        for i in self.survivors:
            survivor_cost = self.population_lst[i].taboo_QAP_cost()
            if survivor_cost < island_best:
                island_best = survivor_cost
        for offspring in self.offsprings:
            offspring_cost = offspring.taboo_QAP_cost()
            if offspring_cost < island_best:
                island_best = offspring_cost
        print("GA->PSO")
        for i in self.survivors:
            self.population_lst[i].GA_to_PSO(self.V_mean)
        
        for i in range(len(self.offsprings)):
            self.offsprings[i].GA_to_PSO(self.V_mean)
        self.population_lst.extend(self.offsprings)
        self.del_idx = [idx for idx in self.idx if idx not in self.survivors]
        self.island_best = island_best
        return self.del_idx, self.island_best


def initialization(M_start = 100) -> List[Instance]:
    instance_lst = [Instance() for i in range(M_start)]
    for i in range(len(instance_lst)):
        instance_lst[i].perm_matrix = np.random.rand(N,N)
        instance_lst[i].velocity_matrix = np.random.rand(N,N)
    return instance_lst

def simmilarity_inst(inst1 : Instance, inst2 : Instance):
    return np.sum(np.minimum(inst1.histogram, inst2.histogram))


def split_population(population_lst : List[Instance], M_species = 3) -> List[List[int]]:
    
    for i in range(len(population_lst)):
        population_lst[i].make_hist()
    
    frontiers = []
    first = population_lst[0]
    second = population_lst[0]
    curr_smalest = simmilarity_inst(first,second)
    ## first two
    for i in range(len(population_lst)):
        for j in range(len(population_lst)):
            if simmilarity_inst(population_lst[i], population_lst[j]) < curr_smalest:
                curr_smalest = simmilarity_inst(population_lst[i], population_lst[j])
                first = population_lst[i]
                second = population_lst[j]
    frontiers.append(first)
    frontiers.append(second)


    for time in range(M_species-2):
        temp = population_lst[0]
        curr_smalest = np.sum(np.array([simmilarity_inst(population_lst[0], inst) for inst in frontiers]))
        for i in range(len(population_lst)):
            temp_sim = np.sum(np.array([simmilarity_inst(population_lst[i], inst) for inst in frontiers]))
            if temp_sim < curr_smalest:
                curr_smalest = temp_sim
                temp = population_lst[i]
        frontiers.append(temp)
    
    idx_lst = []
    for i in range(M_species):
        idx_lst.append([])
    
    for i in range(len(population_lst)):
        temp_idx = np.argmax(np.array([simmilarity_inst(population_lst[i], inst) for inst in frontiers]))
        idx_lst[temp_idx].append(i)
    print([len(idx_lst[i]) for i in range(len(idx_lst))])
    return idx_lst#[list(range(len(population_lst)))[0:len(population_lst)//3], list(range(len(population_lst)))[len(population_lst)//3:(2*len(population_lst))//3], list(range(len(population_lst)))[(2*len(population_lst))//3:]]

def run(callback, M_PSO = M_PSO, M_TABOO = M_TABOO, M_species = M_SPECIES, M_start=START, max_it = MAX_ITER):
    population_lst : List[Instance] = initialization(M_start=M_start)
    print(N)
    # print(max_it)
    for i in range(max_it):
        PSO(callback, population_lst, M_PSO)
        Taboo(callback, population_lst, M_TABOO)

        split_idx : List[List[int]] = split_population(population_lst, M_species )
        del_list = []
        best_iteration = []
        for i in range(M_species):
            island = Island(population_lst, split_idx[i])
            island.count_mean_transformation()
            new_del_list, new_island_best = island.run(callback)
            del_list.extend(new_del_list)
            global MIN_COST, BEST_PERM
            if new_island_best < MIN_COST:
                MIN_COST = new_island_best
                BEST_PERM = 'nie wiem jak wyciągnąć'
                

            print("NEW ISLAND BEST: ", new_island_best)
            best_iteration.append(new_island_best)
            print(best_iteration)
        callback(best_iteration)
        del_list.sort(reverse=True)
        for i in del_list:
            population_lst.pop(i)
    print('taki jest najlepszy', MIN_COST, ' dla perm ', BEST_PERM)
        
    print("finished :)")

def load_problem(path):
    f = open(path, 'r')
    data = f.read()
    data = data.split('\n')
    size = int(data[0])
    
    F = []
    D = []
    for row in data[2:size+2]:
        F.append([])
        row = row.split(' ')
        for nums in row:
            if nums:
                F[-1].append(float(nums))

    for row in data[size+3:2*size+3]:
        D.append([])
        row = row.split(' ')
        for nums in row:
            if nums:
                D[-1].append(float(nums))
    return size, np.array(F), np.array(D)
    
    
class Worker(QObject):
    finished = Signal() 
    start_runn = Signal(int, int, int, int, int)
    value_updated = Signal(list)
    def __init__(self):
        super().__init__()
        self.start_runn.connect(self.runn)

    @Slot(int, int, int, int, int)
    def runn(self, m_pso, m_taboo, m_species, m_start, max_it):
        def emit_value(val):
            self.value_updated.emit(val)

        run(emit_value, M_PSO=m_pso, M_TABOO =m_taboo, M_species = m_species, M_start=m_start, max_it=max_it)
        self.finished.emit()
    

class EmittingStream(QObject):
    text_written = Signal(str)

    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        self.original_stream.write(text)
        self.text_written.emit(str(text))

    def flush(self):
        self.original_stream.flush()
    
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Przekierowanie stdout i stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Przekieruj stdout i stderr
        sys.stdout = EmittingStream(self.original_stdout)
        sys.stderr = EmittingStream(self.original_stderr)
        sys.stdout.text_written.connect(self.append_text)
        sys.stderr.text_written.connect(self.append_text)
        self.pushButton.clicked.connect(self.start_thread)

        self.x = []
        self.y1 = []
        self.y2 = []
        self.y3 = []

        self.curve1 = self.plotWidget.plot([], [], pen='r')
        self.curve2 = self.plotWidget.plot([], [], pen='g')
        self.curve3 = self.plotWidget.plot([], [], pen='b')

        self.showing_multiple = False
        self.curve2.hide()
        self.curve3.hide()

        self.counter = 0

    # def update_display(self, value):
    #     print(value)
    #     self.x.append(self.counter)
    #     self.y.append(value)
    #     self.counter += 1
    #     # self.y = self.y[1:] + [value]
    #     self.curve.setData(self.x, self.y)

    def update_display(self, value):
        print("VALUE: ", value)
        self.x.append(self.counter)
        if len(value) == 3:
            self.showing_multiple = True
            self.y1.append(value[0])
            self.y2.append(value[1])
            self.y3.append(value[2])
        else:
            self.y1.append(value[0])
            self.y2.append(value[0])
            self.y3.append(value[0])

        self.counter += 1

        self.curve1.setData(self.x, self.y1)

        if self.showing_multiple:
            self.curve2.show()
            self.curve3.show()
            self.curve2.setData(self.x, self.y2)
            self.curve3.setData(self.x, self.y3)
            self.showing_multiple = False
    
    def handle_new_value(self, value):
        self.y = self.y[1:] + [value]
        self.curve.setData(self.x, self.y)

    # def update_plot_data(self):
    #     self.y = self.y[1:] + [random.randint(0, 100)]  # dodaj nową wartość
    #     self.curve.setData(self.x, self.y)
    
    def append_text(self, text):
        self.textBrowser.moveCursor(QTextCursor.MoveOperation.End)
        self.textBrowser.insertPlainText(text)
        self.textBrowser.ensureCursorVisible()

    # def start_thread(self):
    #     if not self.thread.isRunning():
    #         self.thread.start()
            
    def start_thread(self):
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        # self.worker.start_runn.emit(13)
        self.get_global_vars()
        # self.thread.started.connect(self.worker.runn)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        
        self.thread.started.connect(lambda: self.worker.start_runn.emit(*self.get_values_for_run()))
        self.worker.value_updated.connect(self.update_display)
        self.thread.start()

        
    def get_values_for_run(self):
        m_pso = int(self.m_pso_num.text())
        m_taboo = int(self.m_taboo_num.text())
        m_species = int(self.m_species_num.text())
        m_start = int(self.m_start_num.text())
        max_it = int(self.max_iter_num.text())
        return m_pso, m_taboo, m_species, m_start, max_it
    
    def get_global_vars(self):
        global N, TABOO_NEIGHBORS, MUTATION_PROB, SURV_PART, W, D
        # prev_n = N
        N, W, D = load_problem('qap/Chr20a.txt')
        # N = int(self.N_numer.text())
        # if prev_n != N:
        #     global W, D
        #     W = np.random.random((N,N))
        #     D = np.random.random((N,N))*(np.ones((N,N)) - np.eye(N))
        #     D = (D+D.T)/2
        TABOO_NEIGHBORS = int(self.taboo_neighbours_num.text())
        MUTATION_PROB = float(self.mut_prob_num.text())
        SURV_PART = float(self.surr_part_num.text())
        global LONG_TERM_CONST, OMEGA, C_1, C_2, B1, B2, C_1_GA_PSO, C_2_GA_PSO
        LONG_TERM_CONST = int(self.long_term_cost_num.text())
        OMEGA = float(self.omega_num.text())
        C_1 = float(self.c1_num.text())
        C_2 = float(self.c2_num.text())
        B1 = float(self.b1_num.text())
        B2 = float(self.b2_num.text())
        C_1_GA_PSO = float(self.c1_ga_pso_num.text())
        C_2_GA_PSO = float(self.c2_ga_pso_num.text())
        
        
    # def handle_run(self): 
    #     result = run(max_it=1)
    #     # self.textBrowser.setPlainText("dupa")

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec())
    # run()
    # pass