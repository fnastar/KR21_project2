import itertools

from BayesNet import BayesNet
from typing import Set
import pandas as pd



class MAP:

    def __init__(self):
        pass

    def sum_out(self, bn: BayesNet, node: str) -> None:
        cpt = bn.get_all_cpts()
        dfs_containing_var = [val for key, val in cpt.items() if node in val]
        var_cpt = bn.get_cpt(node)

        # node_df = [dataframe for dataframe in dfs_containing_var if node == dataframe.keys()[-2]][0]

        # Check if DF is solely dependant on the wanted variable
        if len(var_cpt.columns) != 2:
            # Recursively sum out other variables
            for i in range(len(var_cpt.columns) - 2):
                self.sum_out(
                    bn=bn,
                    node=var_cpt.columns[i]
                )

        dfs_containing_var.remove(var_cpt)

        var_value_true = var_cpt.loc[var_cpt[node] is True]['p']
        var_value_false = var_cpt.loc[var_cpt[node] is False]['p']
        for df in dfs_containing_var:
            for index, row in df.iterrows():
                val = var_value_true if row[node] is True else var_value_false
                df.at[index, 'p'] *= val

    def super_max(self, cpt: pd.DataFrame, variable: str):

        s1 = cpt.loc[cpt[variable] == True].max().to_frame()
        s2 = cpt.loc[cpt[variable] == False].max().to_frame()

        cpt = pd.concat([s1, s2], axis=1).reset_index().transpose()

        return cpt










