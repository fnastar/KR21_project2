import itertools
from Ordering import Ordering
from BayesNet import BayesNet
import pandas as pd
import numpy as np


class MPE:

    def __init__(self):
        self._bn: BayesNet
        self._evidence: dict[str, bool]
        self._ordering = Ordering()
        self._value_dict = {}

    def run(self, bn: BayesNet, evidence: dict[str, bool], order: list):
        # Assign class attributes
        self._bn = bn
        self._evidence = evidence

        # Pruning
        self._edge_pruning()

        # Determine Order
        query = self._bn.get_all_variables()

        # Different Order Heuristic
        # heuristic = order
        # ordered_query = self._ordering.min_degree(bn=self._bn, X=query) if heuristic == 0 \
        #     else self._ordering.min_degree(bn=self._bn, X=query)
        ordered_query = order

        # Get all CPTs of the Network
        cpt_list = list(self._bn.get_all_cpts().values())

        # Start Loop
        for variable in ordered_query:

            # Get all factors containing given variable
            factors = self.all_cpt_containing_var(cpt_list=cpt_list, variable=variable)

            # Store current factors
            headers = [list(cpt.columns) for cpt in factors]

            # Multiply all given factors for given variable
            product = self.multiply(factors=factors)

            # Maximize the product of factors for the given variable
            new_cpt = self.super_max(cpt=product, variable=variable)

            # Change used factors in the list of all CPTs with maximized product of used factors
            cpt_list = [cpt for cpt in cpt_list if list(cpt.columns) not in headers]
            cpt_list.append(new_cpt)

        # Initialize Dataframe
        trivial = pd.DataFrame({'p': [1.0]})

        # Multiply final expressions
        for f in cpt_list:
            trivial.at[0, 'p'] = trivial.iloc[0]['p'] * f['p'].max()

        #print(f'\nThis is the MPE given the evidence {self._evidence}: \n {trivial}')
        return trivial

    def _edge_pruning(self):

        all_edges = [self._bn.get_edges_outgoing_from_var(variable=var) for var in self._evidence]

        for edges_from_var in all_edges:
            self._bn.del_edges(edges=edges_from_var)

        for var, CPT in self._bn.get_all_cpts().items():
            NEW_CPT = self._bn.get_compatible_instantiations_table(instantiation=pd.Series(self._evidence), cpt=CPT)

            for ev_var in self._evidence.keys():
                if ev_var in NEW_CPT and ev_var != var:
                    NEW_CPT = NEW_CPT.drop(ev_var, axis=1)

            self._bn.update_cpt(variable=var, cpt=NEW_CPT)

    @staticmethod
    def all_cpt_containing_var(cpt_list: list[pd.DataFrame], variable: str) -> list[pd.DataFrame]:
        return [cpt for cpt in cpt_list if variable in cpt.columns]

    @staticmethod
    def multiply(factors: list[pd.DataFrame]) -> pd.DataFrame:

        # Initialize merging Dataframe
        merged_df = factors.pop(0)

        # Start merging for loop
        for factor in factors:
            common_vars = np.intersect1d(merged_df.columns[:-1], factor.columns[:-1])
            merged_df = pd.merge(merged_df, factor, on=list(common_vars))
            merged_df['p'] = (merged_df['p_x'] * merged_df['p_y'])
            merged_df.drop(['p_x', 'p_y'], inplace=True, axis=1)

        merged_df['p'] = merged_df['p'].astype(float)
        return merged_df

    @staticmethod
    def super_max(cpt: pd.DataFrame, variable: str):

        cols = [col for col in cpt.columns[:-1] if col != variable]

        if len(cols) != 0:
            cpt = cpt.sort_values(by=['p'])

            cols = [col for col in cpt.columns[:-1] if col != variable]


            unique_vals2 = cpt.drop_duplicates(subset=cols, keep="last")

            unique_vals2 = unique_vals2.drop(columns=variable)
            unique_vals2 = unique_vals2.reset_index(drop=True)

            return unique_vals2

        return cpt
