import math
import numpy as np
import pandas as pd
from scipy.stats import cauchy
import random
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import choice as np_choice

random_matrix = pd.DataFrame([[int(random.random() * 100) for _ in range(100)]
                              for _ in range(100)])
random_matrix.to_csv('random_matrix.csv', header=True, index=False)
random_matrix = pd.read_csv('random_matrix.csv')
spisok = random_matrix.values.tolist()


def el_int(el):
    try:
        int(el)
        return True
    except ValueError:
        return False


def el_float(el):
    try:
        float(el)
        return True
    except ValueError:
        return False


def el_complex(el):
    try:
        complex(el)
        return True
    except ValueError:
        return False


def rev_complex(h):
    h_rev = ''
    sep = 0
    if h[0] == '+' or h[0] == '-':
        for _ in range(1, len(h)):
            if h[_] == '+' or h[_] == '-':
                sep = _
                break
        h_rev = h[sep:len(h)] + h[0:sep]
    else:
        for _ in range(0, len(h)):
            if h[_] == '+' or h[_] == '-':
                sep = _
                break
        h_rev = h[sep:len(h)] + '+' + h[0:sep]
    return (h_rev)


def matr(m, n):
    matrix = []
    print('Введите элементы строки матрицы через пробел:')
    for _ in range(0, m):
        a = []
        row = input()
        row = row.split(' ')
        matrix.append(row)
        if len(row) != n:
            print('Некорректное количество элементов в строке матрицы.')
            exit()
        for j in range(0, n):
            el = matrix[_][j]
            k = 0
            while k == 0:
                if el_int(el) is True:
                    matrix[_][j] = int(el)
                    k = 1
                else:
                    if el_float(el) is True:
                        matrix[_][j] = float(el)
                        k = 1
                    else:
                        if el_complex(el) is True:
                            matrix[_][j] = complex(el)
                            k = 1
                        else:
                            if el_complex(rev_complex(el)) is True:
                                matrix[_][j] = complex(rev_complex(el))
                                k = 1
                            else:
                                el = input('Неверный формат ввода. '
                                           'Повторите ввод '
                                           'элемента [{}, {}]: '.format(_, j))
    return (matrix)


def temperatura(k, t):
    return t / k


def coeff_a(xk, tk):
    return xk + tk * cauchy(0, 1).rvs(size=1)


def simulated_annealing(dist, n, t0):
    way = [_ for _ in range(n)]
    rand0 = [_ for _ in range(1, n)]
    tk = 1
    m = 1
    s = 0
    x0 = 0.1
    x = [x0]
    t = t0
    while t > tk:
        sp = 0
        t = temperatura(m, t0)
        x.append(coeff_a(x[m - 1], t))
        way_p = [way[j] for j in range(n)]
        rand = random.sample(rand0, 2)
        way_p[rand[0]], way_p[rand[1]] = way_p[rand[1]], way_p[rand[0]]
        for j in range(n - 1):
            sp = sp + dist[way_p[j]][way_p[j + 1]]
        sp = sp + dist[way_p[0]][way_p[-1]]
        if m == 1 or sp < s:
            s = sp
            way = [way_p[j] for j in range(n)]
        else:
            p = math.exp(-(sp - s) / t)
            if x[m - 1] < p:
                x[m - 1], x[m] = x[m], x[m - 1]
                s = sp
                way = [way_p[j] for j in range(n)]
        m += 1
    way.append(way[0])
    return way, s, m


def file():
    import csv
    matrix_1 = []
    name = input("Введите названи файла. Например, city.csv: ")
    with open(name) as file:
        reader = csv.reader(file, delimiter=';', quotechar=',')
        for row in reader:
            matrix_1.append(row)
    matrix_1 = [[float(matrix_1[i][j]) for j in range(len(matrix_1))]
                for i in range(len(matrix_1))]
    return matrix_1


def random_dist(k):
    d = [[0 if i == j else random.uniform(0, 10) for j in range(k)]
         for i in range(k)]
    for _ in range(k):
        print(d[_])
    return d


def inlet():
    print("Ввод данных")
    length = int(input("Введите: 1 - для считывания файла с устройства, "
                       "2 - для случайной генерации, "
                       "3 - для ввода матрицы с клавиатуры\n"))
    if length == 1:
        dist = file()
    if length == 2:
        k = int(input("Введите количество городов: "))
        dist = random_dist(k)
    if length == 3:
        k = int(input("Введите количество городов: "))
        dist = matr(k, k)
    return dist


class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations,
                 decay, alpha=1, beta=1):
        i = 0
        j = 0
        while i < len(distances):
            while j < len(distances):
                if distances[i][j] == 0:
                    distances[i][j] = np.inf
                    i += 1
                    j += 1
                else:
                    continue

        self.distances = np.array(distances)
        self.pheromone = np.ones(self.distances.shape) / len(self.distances)
        self.all_inds = range(len(self.distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best,
                                  shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone * self.decay
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev],
                                  visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move


def route_conversion(lst):
    result = []
    for _ in range(len(lst)):
        if _ == 0:
            result.append('-'.join([str(lst[_][0]), str(lst[_][1])]))
        else:
            result.append(str(lst[_][1]))
    return '-'.join(result)


def route_con(lst):
    result = []
    for _ in range(len(lst)):
        if _ == 0:
            result.append(lst[_][0])
            result.append(lst[_][1])
        else:
            result.append(lst[_][1])
    return result


def graph(n, way, dist):
    rand = [i for i in range(n)]
    g = nx.Graph()
    g.add_nodes_from(rand)
    for _ in range(n):
        for j in range(_ + 1, n):
            if dist[_][j] != 0:
                g.add_edge(rand[_], rand[j])
    comb = []
    for _ in range(n):
        if rand.index(way[_]) > rand.index(way[_ + 1]):
            comb.append(tuple([way[_ + 1], way[_]]))
        else:
            comb.append(tuple([way[_], way[_ + 1]]))
    edge_colors = ["red" if _ in comb else "blue" for _ in g.edges()]
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos, edge_color=edge_colors)
    plt.title("Алгоритм Отжига")
    plt.show()


def graph_1(n, way, dist):
    rand = [_ for _ in range(n)]
    g = nx.Graph()
    g.add_nodes_from(rand)
    for _ in range(n):
        for j in range(_ + 1, n):
            if dist[_][j] != 0:
                g.add_edge(rand[_], rand[j])
    comb = []
    for _ in range(n):
        if rand.index(way[_]) > rand.index(way[_ + 1]):
            comb.append(tuple([way[_ + 1], way[_]]))
        else:
            comb.append(tuple([way[_], way[_ + 1]]))
    edge_colors = ["red" if _ in comb else "blue" for _ in g.edges()]
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos, edge_color=edge_colors)
    plt.title("Алгоритм Муравьиной Колонии")
    plt.show()


distant = inlet()
len_m = len(distant)
temper = len_m * (len_m - 1) / 2
w, s, q = simulated_annealing(distant, len_m, temper)
print("Длина маршрута: ", s)
print("Маршрут алгоритма имитации отжига: ", w)
print("Количество итераций в маршруте имитации отжига: ", q)
graph(len_m, w, distant)

distance = distant
ant_colony = AntColony(distance, len(distance) * 2, 5, len(distance) * 4,
                       0.95, alpha=1, beta=1)
shortest_path = ant_colony.run()
route = shortest_path[0]
len_m = len(distance)
results = route_con(shortest_path[0])
print("Полученный путь алгоритмом муравьиной колонии:",
      route_conversion(shortest_path[0]))
print("Стоимость пути муравьиной колонии:", shortest_path[1])
graph_1(len_m, results, distance)
