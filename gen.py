import numpy as np
import networkx as nx
import os

# Параметры графа
num_graphs = 1000
num_nodes = 30

# Массив для хранения весовых матриц
weights_matrices = np.zeros((num_graphs, num_nodes, num_nodes))

# Генерация графов и их весовых матриц
for i in range(num_graphs):
    # Создаем полный граф
    G = nx.complete_graph(num_nodes)
    
    # Назначаем случайные веса ребрам
    for (u, v) in G.edges():
        weight = np.random.uniform(1, 10)  # Случайный вес от 1 до 10
        G[u][v]['weight'] = weight
    
    # Сохраняем весовую матрицу
    weights_matrix = nx.to_numpy_array(G, weight='weight')
    weights_matrices[i] = weights_matrix

# Получаем директорию, где находится сам скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(script_dir, 'data')
file_path = os.path.join(directory, 'weights.npy')

# Создаем папку, если она не существует
os.makedirs(directory, exist_ok=True)

# Сохраняем файл
np.save(file_path, weights_matrices)
