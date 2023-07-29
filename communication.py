from matplotlib import pyplot as plt
import networkx as nx
import numpy as np



fig = plt.figure(figsize=(10, 10))
G = nx.connected_watts_strogatz_graph(3000, 2000, 1,  seed=1)
nx.draw_networkx(G)
plt.show()
plt.savefig('communication.png', dpi=100)

np.set_printoptions(threshold=np.inf)
dict = nx.to_dict_of_lists(G)
print(dict)
connect_path = './Process-Data/'
file = open("connection.txt", "w")
for key, value in dict.items():
    file.write(str(key) + ": " + str(value) + "\n")