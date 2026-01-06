import numpy as np
import pandas as pd
from tqdm import tqdm
import operator


class Radiation:
    def __init__(self, name="Radiation model"):
        self.name_ = name
        self._inputFile = None
        self._out_format = None

    def _get_flows(
        self,
        origin,
        weight_col,
    ):
        edges = []
        probs = []

        # mi
        origin_relevance = self.pop_dict[origin]
        # Oi
        try:
            origin_outflow = self.tot_outflows[origin]
        except AttributeError:
            origin_outflow = 1

        if origin_outflow > 0.0:  # only for origins with positive outflows
            # find the shortest paths
            list_of_destinations = self.destination_dict[origin]
            shortest_paths = self._network.get_shortest_paths(
                v=origin,
                to=list_of_destinations,
                weights=weight_col,  # weight columns in df
                mode="out",
                output="epath",
            )
            # [origin, destination, path]
            temp_flow_matrix = pd.DataFrame(
                [(origin, list_of_destinations, shortest_paths)],
                columns=["origin", "destination", "path"],
            ).explode(["destination", "path"])
            # temp_flow_matrix = temp_flow_matrix[
            #     temp_flow_matrix.path.apply(lambda x: len(x) > 0)
            # ].reset_index(drop=True)

            destination_relevance = (
                temp_flow_matrix["destination"].apply(lambda x: self.pop_dict.get(x, 0))
            ).sum()

            total_relevance = origin_relevance + destination_relevance
            normalization_factor = 1.0 / (1.0 - origin_relevance / total_relevance)

            # compute the normalization factor
            destinations_and_weights = []
            for _, row in temp_flow_matrix.iterrows():
                destination = row["destination"]
                weight = 0

                if len(row["path"]) == 0:  # if no path is found
                    weight = 1e20  # assign a very large weight if no path is found
                else:
                    for edge in row["path"]:
                        weight += self.edge_weight_dict[edge]
                destinations_and_weights += [(destination, weight)]

            # sort the destinations by distance (from the closest to the farthest)
            destinations_and_weights.sort(key=operator.itemgetter(1))

            sum_inside = 0.0
            for destination, _ in destinations_and_weights:
                destination_relevance = self.pop_dict.get(destination, 0)
                prob_origin_destination = (
                    normalization_factor
                    * (origin_relevance * destination_relevance)
                    / (
                        (origin_relevance + sum_inside)
                        * (origin_relevance + sum_inside + destination_relevance)
                    )
                )

                sum_inside += destination_relevance
                edges += [[origin, destination]]
                probs.append(prob_origin_destination)

            probs = np.array(probs)

            if self._out_format == "flows":
                quantities = np.rint(origin_outflow * probs)
            elif self._out_format == "flows_sample":
                quantities = np.random.multinomial(origin_outflow, probs)
            else:
                quantities = probs

            edges = [edges[i] + [od] for i, od in enumerate(quantities)]

        return edges

    def generate(
        self,
        network,
        inputFile,
        weight_col,
        tile_id_column="origin_node_idx",
        tot_outflows_column="tot_outflow",
        relevance_column="population",
        list_of_destinations_column="list_of_destinations",
        out_format="flows",
    ):
        self._network = network
        self.edge_weight_dict = {k: v[weight_col] for k, v in enumerate(network.es)}
        self._out_format = out_format
        self._tile_id_column = tile_id_column
        self.pop_dict = inputFile.set_index(tile_id_column)[relevance_column]
        self.destination_dict = inputFile.set_index(tile_id_column)[
            list_of_destinations_column
        ].to_dict()

        if "flows" in out_format:
            if tot_outflows_column not in inputFile.columns:
                raise KeyError(
                    "The column %s for the 'tot_outflows' must be present in the tessellation."
                    % tot_outflows_column
                )
            self.tot_outflows = inputFile.set_index(tile_id_column)[
                tot_outflows_column
            ].to_dict()

        # check if arguments are valid
        if out_format not in ["flows", "flows_sample", "probabilities"]:
            raise ValueError(
                'Value of out_format "%s" is not valid. \nValid values: flows, flows_sample, probabilities.'
                % out_format
            )

        all_flows = []
        for origin in tqdm(range(len(inputFile))):  # tqdm print a progress bar
            # calculate relevance
            origin = inputFile.loc[origin, tile_id_column]
            # get the edges for the current origin location
            flows_from_origin = self._get_flows(origin, weight_col)
            if len(flows_from_origin) > 0:
                all_flows += list(flows_from_origin)
        # Always return a FlowDataFrame
        if True:  # 'flows' in out_format:
            return self._from_matrix_to_flowdf(all_flows, inputFile)
        else:
            return all_flows

    def _from_matrix_to_flowdf(self, all_flows, inputFile):
        output_list = [[i, j, flow] for i, j, flow in all_flows if flow > 0.0]
        temp_df = pd.DataFrame(output_list, columns=["origin", "destination", "flows"])
        return temp_df
