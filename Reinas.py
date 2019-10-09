# S1) En el juego de ajedrez la reina amenaza a aquellas piezas que se encuentren en su misma fila,
# columna o diagonal. El problema de las 8 reinas (o n-reinas ya que dependen del numero asignado)
# consiste en poner sobre un tablero de ajedrez ocho reinas sin que estas se amenacen entre ellas.

# CREADO NDD Sept 2019
import random
import numpy as np


"""   Comentarios son Una Linea: #
O triple comilla doble: Un bloque"""

"""Si se desea una población inicial no aleatoria
cromosoma1 = [2, 3, 2, 7, 1, 6, 3, 7]
cromosoma2 = [0, 1, 0, 0, 0, 0, 0, 0]
cromosoma3 = [1, 1, 0, 0, 1, 2, 2, 2]
cromosoma4 = [1, 1, 1, 0, 1, 4, 2, 3]
poblInicial = np.array([cromosoma1, cromosoma2, cromosoma3, cromosoma4]
"""

# MEJORA: Tamaño de la Población como parametro
# random.seed(1)
# print("\n","aletorio:", random.randrange(2)) #Entero 0 o 1

# FUNCIONES PARA OPERADORES


#### Parametros #####
x = 8  # numero de reinas: x
n = 5  # individuos en la poblacion - cromosomas: n
Pcruce = 0.9  # Probabilidad de Cruce
Pmuta = 0.1  # Probabilidad de Mutación

fitness = np.empty((n))  # fitness maximo es 28
factible = np.empty((n))
acumulado = np.empty((n))
suma = 0
total = 0

# Individuos, soluciones o cromosomas
tamTablero = 8
poblInicial = np.random.randint(0, tamTablero, (n, x))
tablero = []
# Ingresar los datos del Problema de la Mochila - Peso y Utilidad de los Elementos


def initTablero():
    for i in range(tamTablero):
        temp = [0] * tamTablero
        tablero.append(temp)
    return tablero


def llenarTablero(pos):  # ubica la posicion de cada reina
    tablero = initTablero()
    for i in range(tamTablero):
        tablero[i][poblIt[pos][i]] = 1


def evalua(n, x, poblIt, pos):
    cont_cruzan = 0
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
    for i in range(tamTablero):
        j = poblIt[pos][i]
        m = i+1
        n = j-1
        while m < tamTablero and n >= 0:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m += 1
            n -= 1
        m = i+1
        n = j+1
        while m < tamTablero and n < tamTablero:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m += 1
            n += 1
        m = i-1
        n = j-1
        while m >= 0 and n >= 0:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m -= 1
            n -= 1
        m = i-1
        n = j+1
        while m >= 0 and n < tamTablero:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m -= 1
            n += 1
    return cont_cruzan

#    suma = 0
#    total = 0
#    sumaF = 0
#    for i in range(0, n):
#        for j in range(0, x):
#            suma += poblIt[i, j]*utilidad[j]
#            sumaF += poblIt[i, j]*pesos[j]
#        fitness[i] = suma
#        factible[i] = sumaF
#        total += suma
#        sumaF = 0
#        suma = 0
#    return fitness, total, factible


def imprime(n, total, fitness, poblIt):
    # Tabla de evaluación de la Población
    acumula = 0
    print("\n", 'Tabla Iteración:', "\n")
    for i in range(0, n):
        probab = fitness[i]/total
        acumula += probab
        print([i+1], " ", poblIt[i], "  ", fitness[i], " ", factible[i], " ",
              "{0:.3f}".format(probab), " ",  "{0:.3f}".format(acumula))
        acumulado[i] = acumula
    print("Total Fitness:      ", total)
    return acumulado


def seleccion(acumulado):
    escoje = np.random.rand()
    print("escoje:      ", escoje)

    for i in range(0, n):
        if acumulado[i] > escoje:
            padre = poblIt[i]
            break
    return (padre)


def cruce(a1, p1, p2):
    if a1 < Pcruce:
        print("Mas grande", Pcruce, "que ", a1, "-> Si Cruzan")
        corte = np.random.randint(1, len(p1))
        temp1 = p1[0:corte]  # [i:j] corta desde [i a j)
        temp2 = p1[corte:len(p1)]
        print(temp1, temp2)
        temp3 = p2[0:corte]
        temp4 = p2[corte:len(p2)]
        print(temp3, temp4)
        hijo1 = list(temp1)
        hijo1.extend(list(temp4))
        hijo2 = list(temp3)
        hijo2.extend(list(temp2))

    else:
        print("Menor", Pcruce, "que ", a1, "-> NO Cruzan")
        hijo1 = p1
        hijo2 = p2

    return hijo1, hijo2


def mutacion(individuo):
    for i in range(0, len(individuo)):
        rpmuta = np.random.rand()
        if rpmuta < Pmuta:
            pos_1 = np.random.randint(0, len(individuo))

            while(pos_1 == pos_2):
                pos_2 = np.random.randint(0, len(individuo))

            aux = individuo[pos_1]
            individuo[pos_1] = individuo[pos_2]
            individuo[pos_2] = aux

    return individuo


def es_factible(individuo):
    suma = 0
    for i in range(len(individuo)):
        suma += individuo[i] * pesos[i]
    if suma < mochila:
        return True
    else:
        return False


print("Poblacion inicial Aleatoria:", "\n", poblInicial)
print("\n", "Utilidad:", utilidad)
print("\n", "Pesos", pesos)
poblIt = poblInicial

# FIN DE LOS DATOS INICIALES


# Llama función evalua, para calcular el fitness de cada individuo
fitness, total, factible = evalua(n, x, poblIt)
#####print("\n","Funcion Fitness por individuos",  fitness)
#####print("\n","Suma fitness: ",  total)

# imprime la tabla de la iteracion
imprime(n, total, fitness, poblIt)

# ***************************************
# Inicia Iteraciones
for iter in range(1):
    print("\n", "Iteración ", iter+1)
    poblacion_aux = np.empty((n, x), dtype='int')
    cont = 0
    while cont < 4:
        papa1 = seleccion(acumulado)  # Padre 1
        print("padre 1:", papa1)
        papa2 = seleccion(acumulado)  # Padre 2
        print("padre 2:", papa2)
        hijoA, hijoB = cruce(np.random.rand(), papa1, papa2)
        hijoA = mutacion(hijoA)
        hijoB = mutacion(hijoB)
        factibleA = es_factible(hijoA)
        factibleB = es_factible(hijoB)
        if factibleA:
            print("HijoA es factible y es: ", hijoA)
            poblacion_aux[cont] = hijoA
            cont += 1
            if cont >= 4:
                break
        if factibleB:
            print("HijoB es factible y es: ", hijoB)
            poblacion_aux[cont] = hijoB
            print("hijo2: ", hijoB)
            cont += 1
    poblIt = poblacion_aux
    print("\n", "Poblacion Iteración ", iter+1, "\n", poblIt)
    fitness, total, factible = evalua(n, x, poblIt, utilidad)
    #### print("\n","Funcion Fitness por individuos",  fitness)
    #### print("\n","Suma fitness: ",  total)

    # imprime la tabla de la iteracion
    imprime(n, total, fitness, poblIt)


class Individuo():
    def __init__(self, n_queens=8):
        self.individuo = self.generate_individuo()
        self.fitness = self.calc_fitness()
        self.feasible = self.is_feasible()
        self.n_queens = n_queens
    
    def get_fitness():
        return self.fitness

    def get_feasible():
        return self.feasible
    
    def calc_fitness(self):
        for i in range(tamTablero):
        j = poblIt[pos][i]
        m = i+1
        n = j-1
        while m < tamTablero and n >= 0:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m += 1
            n -= 1
        m = i+1
        n = j+1
        while m < tamTablero and n < tamTablero:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m += 1
            n += 1
        m = i-1
        n = j-1
        while m >= 0 and n >= 0:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m -= 1
            n -= 1
        m = i-1
        n = j+1
        while m >= 0 and n < tamTablero:
            if tablero[m][n] == 1:
                cont_cruzan += 1
            m -= 1
            n += 1
    return cont_cruzan
        

    def is_feasible():
        pass

    def generate_individuo():
        self.individuo = np.random.randint(0, self.n_queens, (1, x))[0]
    
    def get_individuo():
        return self.individuo


class Poblacion():
    def __init__(self, tam):
        self.individuos = []
    
    def random_poblacion():
        self.individuos.append(Individuo())