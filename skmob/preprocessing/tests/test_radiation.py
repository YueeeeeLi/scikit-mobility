# %%
import pandas as pd
import numpy as np
import igraph
from skmob.models.radiation_revised import Radiation

# %%
# create a random newtork
test_net = igraph.Graph(directed=False)
test_net.add_vertices(5)  # [0,1,2,3,4]
test_net.add_edges([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
test_net.es["weight"] = [10, 20, 5, 20, 10, 1]

# %%
# create inputFile
inputFile = pd.DataFrame(
    {
        "origin_node_idx": [0, 1, 2, 3, 4],
        "population": [100, 10, 2, 20, 30],
        "tot_outflow": [50, 5, 1, 0, 0],
        "list_of_destinations": [[1, 2, 4], [3, 4], [3], [], []],
    }
)
# %%
np.random.seed(0)
rd_fun = Radiation()

# [origin, destination, flows]
od = rd_fun.generate(
    test_net,
    inputFile,
    tile_id_column="origin_node_idx",
    tot_outflows_column="tot_outflow",
    relevance_column="population",
    list_of_destinations_column="list_of_destinations",
    out_format="flows",
)
print(od)
