# %%
import pandas as pd
import numpy as np
import igraph
from radiation_revised import Radiation

# %%
# create a random newtork
test_net = igraph.Graph(directed=False)
test_net.add_vertices(5)  # [0,1,2,3,4]
test_net.add_edges([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
test_net.es["weight"] = [10, 20, 5, 20, 10]

# %%
# create inputFile
"""
- network structure:
    0 --10-- 1
    |        |
    20       5
    |        |
    2 --20-- 3
            |
            10
            |
            4

- the network has in total 5 nodes with population attached to each node
- origin nodes: 0, 1, 2
- destination nodes: [1,2,4] for origin node 0,
    [3,4] for origin node 1,
    [3] for origin node 2

"""
inputFile = pd.DataFrame(
    {
        "node_idx": [0, 1, 2, 3, 4],  # all the origin and destination ndoes
        "population": [100, 10, 2, 20, 30],  # population for all nodes
        "tot_outflow": [50, 5, 1, 0, 0],  # indicate the outflow for origins (0, 1, 2)
        "list_of_destinations": [
            [1, 2, 4],  # for origin node 0
            [3, 4],  # for origin node 1 ...
            [3],
            [],
            [],
        ],  # restricted destinations for each origin
    }
)
# %%
np.random.seed(0)
rd_fun = Radiation()

# [origin, destination, flows]
od = rd_fun.generate(
    test_net,
    inputFile,
    tile_id_column="node_idx",
    tot_outflows_column="tot_outflow",
    relevance_column="population",
    list_of_destinations_column="list_of_destinations",
    out_format="flows",
)
print(od)
