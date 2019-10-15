# S1) En el juego de ajedrez la reina amenaza a aquellas piezas que se encuentren en su misma fila,
# columna o diagonal. El problema de las 8 reinas (o n-reinas ya que dependen del numero asignado)
# consiste en poner sobre un tablero de ajedrez ocho reinas sin que estas se amenacen entre ellas.

import random
import numpy as np

class Individuo():
    def __init__(self, default_list=[], n_queens=8):

        self.n_queens = n_queens
        if not default_list:
            self.generate_list()
        else:
            self.list = default_list

        self.create_board()
        self.fitness = self.calc_fitness()
        self.feasible = self.is_feasible()

    def get_fitness(self):
        return self.fitness

    def get_feasible(self):
        return self.feasible

    def calc_fitness(self):
        """
        encuentra el numero de cruces en el tablero
        para el cromosoma [2, 3, 2, 7, 1, 6, 3, 7]

        0   0   1   0   0   0   0   0
        0   0   0   1   0   0   0   0
        0   0   1   0   0   0   0   0
        0   0   0   0   0   0   0   1
        0   1   0   0   0   0   0   0
        0   0   0   0   0   0   1   0
        0   0   0   1   0   0   0   0
        0   0   0   0   0   0   0   1

        hay 5 (?) cruces

        """
        cont_cruzan = 0
        for i in range(self.n_queens):
            j = self.list[i]
            m = i + 1
            n = j - 1

            while m < self.n_queens and n >= 0:
                cont_cruzan += self.board[m][n]
                m += 1
                n -= 1

            m = i + 1
            n = j + 1

            while m < self.n_queens and n < self.n_queens:
                cont_cruzan += self.board[m][n]
                m += 1
                n += 1
            m = i - 1
            n = j - 1

            while m >= 0 and n >= 0:
                cont_cruzan += self.board[m][n]
                m -= 1
                n -= 1

            m = i - 1
            n = j + 1

            while m >= 0 and n < self.n_queens:
                cont_cruzan += self.board[m][n]
                m -= 1
                n += 1

        return 1/(1+cont_cruzan) # de forma que el mejor fitness sea cuando hayan 0 cruces
        
    def is_feasible(self):
        return self.fitness == 0

    def generate_list(self):
        self.list = np.random.randint(
            0, self.n_queens, (1, self.n_queens))[0]

    def get_list(self):
        return self.list

    def create_board(self):
        tablero = []

        for _ in range(self.n_queens):
            temp = [0] * self.n_queens
            tablero.append(temp)

        for i in range(self.n_queens):
            tablero[i][self.list[i]] = 1

        self.board = tablero

    def __str__(self):
        return self.list

    def mutar(self, pmuta):
        for _ in range(0, len(self.list)):
            rpmuta = np.random.rand()
            if rpmuta < pmuta:
                pos_1 = np.random.randint(0, len(self.list))
                pos_2 = pos_1
                while(pos_1 == pos_2):
                    pos_2 = np.random.randint(0, len(self.list))

                self.list[pos_1], self.list[pos_2] = self.list[pos_2], self.list[pos_1]

        self.fitness = self.calc_fitness


class Population():
    def __init__(self, default_population=[], tam=5):
        self.individuos = []
        if not default_population:
            self.size = tam
            self.random_population()
        else:
            self.size = len(default_population)
            self.individuos = default_population

        self.total_fitness = sum(
            map(lambda individuo: individuo.get_fitness(), self.individuos))
        self.acumulado = np.cumsum(list(map(lambda individuo: individuo.get_fitness() / self.total_fitness, self.individuos)))

    def random_population(self):
        for _ in range(self.size):
            self.individuos.append(Individuo())
    
    def best_individual(self):
        return max(self.individuos, lambda individuo: individuo.get_fitness())

    def __str__(self):
        return self.individuos
    
    def get_size(self):
        return self.size
    
    def get_individuos(self):
        return self.individuos
    
    def get_acumulado(self):
        return self.acumulado

class GeneticAlgorithm():
    def __init__(self, pmuta=0.1, pcruce=0.9):
        self.population = Population()
        self.pcruce = pcruce
        self.pmuta = pmuta

    def cruce(self, pcruce, p1, p2):
        if pcruce < self.pcruce:
            print("Mas grande", self.pcruce, "que ", pcruce, "-> Si Cruzan")
            corte = np.random.randint(1, len(p1.get_list()))
            temp1 = p1.get_list()[0:corte]  # [i:j] corta desde [i a j)
            temp2 = p1.get_list()[corte: len(p1.get_list())]
            print(temp1, temp2)
            temp3 = p2.get_list()[0:corte]
            temp4 = p2.get_list()[corte:len(p2.get_list())]
            print(temp3, temp4)
            hijo1 = list(temp1)
            hijo1.extend(list(temp4))
            hijo1_individuo = Individuo(hijo1)
            hijo2 = list(temp3)
            hijo2.extend(list(temp2))
            hijo2_individuo = Individuo(hijo2)
        else:
            print("Menor", pcruce, "que ", self.pcruce, "-> No Cruzan")
            hijo1_individuo = p1
            hijo2_individuo = p2

        return hijo1_individuo, hijo2_individuo

    def seleccion(self):
        escoje = np.random.rand()
        print("escoje:      ", escoje)

        acumulado = self.population.get_acumulado()
        individuos = self.population.get_individuos()

        for i in range(0, self.population.get_size()):
            if acumulado[i] > escoje:
                padre = individuos[i]
                break
        return (padre)
