import time

inicio = time.time()

import numpy as np
from scipy.signal import convolve2d as conv2
from queue import PriorityQueue
import math  

class Estado:
    def __init__(self, pai=None, matriz=None):
        self.pai = pai
        self.matriz = matriz

        self.d = 0  #distanciaa
        self.c = 0
        self.p = 0  #prioridade

    def __eq__(self, other):
        return np.array_equal(self.matriz, other.matriz)  # Comparação correta de arrays (gpt enche o saco)

    def __lt__(self, other):
        return self.p < other.p

    def __hash__(self):
        return hash(tuple(self.matriz.flatten()))  # Transforma matriz em tupla para hash (único número)

    def mostrar(self):
        for i in self.matriz:
            print(i)
        print()


def acoes_permitidas(estado):
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    blank = estado.matriz == 9
    mask = conv2(blank, adj, 'same')
    return estado.matriz[np.where(mask)]


def movimentar(s, c):
    matriz = s.matriz.copy()
    matriz[np.where(s.matriz == 9)] = c
    matriz[np.where(s.matriz == c)] = 9
    return Estado(matriz=matriz)


def dist(t1, t2):
    return np.sum(list(map(lambda i, j: abs(i - j), t1, t2)))


def manhattan(estado):
    obj = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return np.sum([dist(np.where(obj == i), np.where(estado.matriz == i)) for i in range(1, 9)])


def hamming(s):
    obj = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    qtde_fora_lugar = len(s.matriz[np.where(s.matriz != obj)])
    # 9 não pode entrar na conta
    return (qtde_fora_lugar - 1 if qtde_fora_lugar > 0 else 0)


def euclidiana(s):
    obj = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    distancia = 0
    for i in range(1, 9):  
        pos_obj = np.where(obj == i)
        pos_estado = np.where(s.matriz == i)
        distancia += math.sqrt((pos_obj[0][0] - pos_estado[0][0]) ** 2 + (pos_obj[1][0] - pos_estado[1][0]) ** 2)
    return distancia


def temResposta(s, f, o):
    retorno = astar(s, f, o)
    if retorno != o:
        print("Não tem solução")
    else:
        print("O jogo tem solução")


def astar(s, f, o):
    Q = PriorityQueue()
    vetor = set()

    s.p = 0
    Q.put((s.p, s))

    while not Q.empty():
        v = Q.get()[1]

        if v == o:
            return v

        vetor.add(v)

        for a in acoes_permitidas(v):
            u = movimentar(v, a)

            if u not in vetor:
                u.d = v.d + 1
                u.pai = v
                u.p = f(u) + u.d
                vetor.add(u)
                Q.put((u.p, u))
                u.mostrar()

    return s


o = Estado(matriz=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

s = Estado(matriz=np.array([[8, 6, 7], [2, 5, 4], [3, 9, 1]]))

print("Euclidiana")
y = astar(s, euclidiana, o)
y.mostrar()
print("Custo:", y.d)

fim = time.time()
print(f"Tempo de execução: {fim - inicio:.2f} segundos")
