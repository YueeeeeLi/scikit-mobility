# %%
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import igraph
from radiation_revised import Radiation

# from radiation import Radiation as rd
# from ipfn import ipfn

import warnings

warnings.simplefilter("ignore")

nist_path = Path(r"C:\Oxford\Research\NIST\local\scripts\results\outputs")
data_path = Path(r"C:\Oxford\Research\NIST\local\data\processed")
CONV_METER_TO_MILE = 0.000621371

# %%
nodes = gpd.read_parquet(data_path / "England_road_nodes_with_bridges.gpq")
edges = gpd.read_parquet(nist_path / "edge_flow_2021.gpq")
inputFile = pd.read_parquet(nist_path / "radiation_inputs_2021_revised.pq")
inputFile = inputFile[
    ~inputFile.destinations.apply(lambda x: len(x) == 0)
].reset_index()

# %%
edges["weight_free"] = (
    edges.geometry.length * CONV_METER_TO_MILE / edges.free_flow_speeds
)  # hour
edges["weight_congest"] = edges.geometry.length * CONV_METER_TO_MILE / edges.acc_speed
graph_df = edges[["from_id", "to_id", "e_id", "weight_free", "weight_congest"]]
test_net = igraph.Graph.TupleList(
    graph_df.itertuples(index=False),
    edge_attrs=list(graph_df.columns)[2:],
    directed=False,
)

# Convert node name to node index
name_to_index = {v["name"]: v.index for v in test_net.vs}
index_to_name = {v: k for k, v in name_to_index.items()}
inputFile["node_idx"] = inputFile["node_id"].map(name_to_index)
inputFile["destinations"] = inputFile["destinations"].apply(
    lambda lst: [name_to_index[name] for name in lst]
)
rd_rev = Radiation()
# %%
# Free-flow time (hour) - based on network
np.random.seed(0)
od_free = rd_rev.generate(
    test_net,
    inputFile,
    weight_col="weight_free",
    tile_id_column="node_idx",
    tot_outflows_column="outflow",
    relevance_column="population",
    list_of_destinations_column="destinations",  # [node_idx,...]
    out_format="flows",
)

# %%
