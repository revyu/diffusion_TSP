import os
import numpy as np
import networkx as nx
from deap import base, creator, tools, algorithms

# Параметры
num_graphs = 1000
num_nodes = 30
pop_size = 100  # Размер популяции
num_generations = 50  # Количество поколений

# Директория для хранения файлов
script_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(script_dir, 'data')
os.makedirs(directory, exist_ok=True)

# Инициализация массива весовых матриц и "хороших" путей
weights_matrices = np.zeros((num_graphs, num_nodes, num_nodes))
good_paths = np.zeros((num_graphs, num_nodes), dtype=int)

# Создание функции для вычисления стоимости пути
def evaluate_path(path, graph):
    weight = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
    weight += graph[path[-1]][path[0]]['weight']  # Замкнуть цикл
    return weight,

# Генетический алгоритм для поиска пути
def find_good_path(graph):
    # Настройка параметров GA
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("indices", np.random.permutation, num_nodes)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_path, graph=graph)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Создание начальной популяции
    population = toolbox.population(n=pop_size)
    
    # Запуск алгоритма
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, verbose=False)
    
    # Находим лучший путь
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

# Генерация графов и их весовых матриц
for i in range(num_graphs):
    G = nx.complete_graph(num_nodes)
    
    for (u, v) in G.edges():
        weight = np.random.uniform(1, 10)
        G[u][v]['weight'] = weight
    
    weights_matrix = nx.to_numpy_array(G, weight='weight')
    weights_matrices[i] = weights_matrix
    
    # Нахождение "хорошего" пути с использованием генетического алгоритма
    best_path = find_good_path(G)
    good_paths[i] = best_path

# Сохранение весовых матриц и хороших путей
np.save(os.path.join(directory, 'weights.npy'), weights_matrices)
np.save(os.path.join(directory, 'good_paths.npy'), good_paths)


