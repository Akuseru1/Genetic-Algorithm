from Reinas import GeneticAlgorithm, np


solucion = GeneticAlgorithm()
iniciales = solucion.population.get_individuos()
for _ in range(2):

    p1 = solucion.seleccion()
    p2 = solucion.seleccion()
    pcruce = np.random.rand()
    h1, h2 = solucion.cruce(pcruce, p1, p2)
    h1.mutar(solucion.pmuta)
    h2.mutar(solucion.pmuta)
#iniciales
solucion.population.get_individuos()