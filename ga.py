# S1) En el juego de ajedrez la reina amenaza a aquellas piezas que se encuentren en su misma fila,
# columna o diagonal. El problema de las 8 reinas (o n-reinas ya que dependen del numero asignado)
# consiste en poner sobre un tablero de ajedrez ocho reinas sin que estas se amenacen entre ellas.

import random
import numpy as np
import copy
import matplotlib.pyplot as plt


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

        8 0   0   0   0   0   0   0   1
        7 0   0   0   1   0   0   0   0
        6 0   0   0   0   0   0   1   0
        5 0   1   0   0   0   0   0   0
        4 0   0   0   0   0   0   0   1
        3 0   0   1   0   0   0   0   0
        2 0   0   0   1   0   0   0   0
        1 0   0   1   0   0   0   0   0
          1   2   3   4   5   6   7   8
        hay 8 cruces diagonales

        """
        cont_cruces = 0
        chromosome = list(self.get_list())
        # cuenta cruces en diagonales
        for i in range(self.n_queens):
            for j in range(self.n_queens):
                if (i != j):
                    cont_cruces += 1 if i - \
                        chromosome[i] == j - chromosome[j] or i + chromosome[i] == j + chromosome[j] else 0
        cont_cruces /= 2
        # cuenta cruces columna, elimina los elementos repetidos del cromosoma
        cont_cruces += self.n_queens - \
            len([ncol for ncol in chromosome if chromosome.count(ncol) == 1])
        cont_cruces /= 2
        # de forma que el mejor fitness sea cuando hayan 0 cruces
        return 1/(1 + cont_cruces)

    def is_feasible(self):
        return self.fitness == 1

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

    def get_board(self):
        for i in range(self.n_queens):
            for j in range(self.n_queens):
                print(self.board[i][j], "\t", end=" ")
            print("\n")

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

        self.fitness = self.calc_fitness()


class Population():
    def __init__(self, default_population=[], tam=20):
        self.individuos = []
        if not default_population:
            self.size = tam
            self.random_population()
        else:
            self.size = len(default_population)
            self.individuos = default_population

        self.total_fitness = sum(
            map(lambda individuo: individuo.get_fitness(), self.individuos))
        self.acumulado = np.cumsum(list(map(
            lambda individuo: individuo.get_fitness() / self.total_fitness, self.individuos)))

    def random_population(self):
        for _ in range(self.size):
            self.individuos.append(Individuo())

    def best_individual(self):
        return max(self.individuos, key=lambda individuo: individuo.get_fitness())

    def __str__(self):
        return f'{self.individuos}'  # self.individuos

    def get_size(self):
        return self.size

    def get_individuos(self):
        return self.individuos

    def get_acumulado(self):
        return self.acumulado

    def get_total_fitness(self):
        return self.total_fitness


class GeneticAlgorithm():
    def __init__(self, pmuta=0.001, pcruce=0.9, elitism=False):
        self.population = Population()
        self.pcruce = pcruce
        self.pmuta = pmuta
        self.elitism = elitism

        self.resume = {
            'fitness_averages': [],
            'populations': [],
            'population_best_solution': None,
            'best_fitness_averages': []
        }

    def cruce(self, pcruce, p1, p2):
        if pcruce < self.pcruce:
            print("Mas grande", self.pcruce, "que ", pcruce, "-> Si Cruzan")
            corte = np.random.randint(1, len(p1.get_list()))
            temp1 = p1.get_list()[0:corte]  # [i:j] corta desde [i a j)
            temp2 = p1.get_list()[corte: len(p1.get_list())]
            print(temp1, temp2)
            temp3 = p2.get_list()[0:corte]
            temp4 = p2.get_list()[corte:len(p2.get_list())]
            print(temp3, temp4, "\n")
            hijo1 = list(temp1)
            hijo1.extend(list(temp4))
            hijo1_individuo = Individuo(hijo1)
            hijo2 = list(temp3)
            hijo2.extend(list(temp2))
            hijo2_individuo = Individuo(hijo2)
        else:
            print("Menor", pcruce, "que ", self.pcruce, "-> No Cruzan\n")
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
        return padre

    def plot(self):
        x = list(range(len(self.resume.get('populations'))))
        y = [population.best_individual().get_fitness()
             for population in self.resume.get('populations')]
        y2 = self.resume.get('fitness_averages')
        y3 = []
        for i in range(0, len(y)):
            if i == 0:
                y3.append(y[0])
            else:
                y3.append((y[i] + y[i-1])/(i+1))
        plt.plot(x, y, label="Curva best-so-far")
        plt.plot(x, y2, label="Curva online")
        plt.plot(x, y3, label="Curva off-line")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

    def run(self, itera=10):
        print("\n iniciales: \n")

        print("Poblacion Inicial Aleatoria      Fitness       Factible")

        for i in range(self.population.get_size()):
            print("\t", self.population.get_individuos()[i].get_list(), "          ", round(self.population.get_individuos()[i].get_fitness(), 2),
                  "       ", self.population.get_individuos()[i].get_feasible())
        print("Total Fitness: ", self.population.get_total_fitness())
        print("Average Fitness: ", self.population.get_total_fitness() /
              self.population.get_size())

        best = self.population.best_individual()
        print("Best Individual: ", best.get_list())
        print("Board Best Individual")
        best.get_board()
        print("\n")

        for _ in range(itera):
            print("Iteracion: ", (_+1), "\n")
            individuos_next_generation = []
            self.resume['populations'].append(copy.deepcopy(self.population))
            self.resume['fitness_averages'].append(
                    self.population.get_total_fitness() / self.population.get_size())
            
            while(True):
                p1 = self.seleccion()
                print("Padre 1: ", p1.get_list())
                p2 = self.seleccion()
                print("Padre 2: ", p2.get_list())

                pcruce = np.random.rand()

                h1, h2 = self.cruce(pcruce, p1, p2)

                h1.mutar(self.pmuta)
                h2.mutar(self.pmuta)

                if h1.get_feasible():
                    print("Hijo1 es factible y es: ", h1.get_list(),
                          "\n---------------------------------------------------")
                individuos_next_generation.append(h1)
                if len(individuos_next_generation) == self.population.get_size():
                    break

                if h2.get_feasible():
                    print("Hijo2 es factible y es: ", h2.get_list())
                individuos_next_generation.append(h2)
                if len(individuos_next_generation) == self.population.get_size():
                    break

            if self.elitism:
                individuos_next_generation[-1] = self.population.best_individual()

            self.population = Population(
                default_population=individuos_next_generation)
            print("          Poblacion Final              Fitness      Factible")

            for i in range(self.population.get_size()):
                print("\t", self.population.get_individuos()[i].get_list(), "          ", round(self.population.get_individuos()[i].get_fitness(), 2),
                      "       ", self.population.get_individuos()[i].get_feasible())
            print("Total Fitness: ", self.population.get_total_fitness())
            print("Average Fitness: ", self.population.get_total_fitness() /
                  self.population.get_size())
            best = self.population.best_individual()
            print("Best Individual: ", best.get_list(), "\n")

        self.resume['population_best_solution'] = copy.deepcopy(max(
            self.resume['populations'], key=lambda population: population.best_individual().get_fitness()))

        print("\n Finales: \n")

        print("        Poblacion Final                Fitness      Factible")

        for i in range(self.population.get_size()):
            print("\t", self.population.get_individuos()[i].get_list(), "          ", round(self.population.get_individuos()[i].get_fitness(), 2),
                  "       ", self.population.get_individuos()[i].get_feasible())
        print("Total Fitness: ", self.population.get_total_fitness())
        print("Average Fitness: ", self.population.get_total_fitness() /
              self.population.get_size())
        best = self.population.best_individual()
        print("Best Individual: ", best.get_list())
        print("Board Best Individual")
        best.get_board()
        print("\n")
        self.plot()
