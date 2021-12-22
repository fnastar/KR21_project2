from typing import Union
from BayesNet import BayesNet
import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
            
            #Graph
            self.G = self.bn.structure
            
            self.variables = self.bn.get_all_variables() 
        else:
            self.bn = net
    
    def see_Graph(self, Graph):
    
        if Graph == 'default':
            self.bn.draw_structure()
        else:
            subax1 = plt.subplot(121)
            nx.draw(Graph, with_labels = True, font_weight='bold')
            plt.show()            
    
    def get_nodes(self, Graph):
        return Graph.nodes()
    
    def find_all_paths(self, graph, X, Y, path = []):
        path = path + [X]
        
        if X == Y:
            return [path]
    
        if X not in graph.nodes():
            return []
        
        paths = []
        
        for node in graph[X]:
            if node not in path:
                newpaths = self.find_all_paths(graph, node, Y, path)
                for newpath in newpaths:
                    paths.append(newpath)
                    
        return paths

    def trigram_path(self, path):
        trigram_paths = []
        
        for i in range(len(path)-2):
            trigram_paths.append(path[i:i+3])
        
        return trigram_paths 
    '''
    Given three sets of variables X, Y , and Z, determine whether X is independent of Y given Z.
    '''
    '''
    D-Seperation
    '''
    def d_separated(self, X, Z, Y):
        Paths = []
        path_state, valve_state = [], []
        
        if type(X) != set or type(Z) != set or type(Y) != set:
            return "X or Y or Z not of type set()" 
        
        if X == {''} or X == {' '} or Y == {''} or Y == {' '}:
            return "X or Y empty"
        
        if Z == {' '} or Z == {''}:
            X_Y_Z = X.union(Y)
        else:
            X_Y_Z = X.union(Z).union(Y)
            
        for node in X_Y_Z:
            if node not in list(self.G.nodes()):
                #print(f'Node: {node} not found')
                print(node)
                return f'Node: {node} not found'
            #return "Warning: Node not found"
        
        #Check for disjointness between X, Y, and Z
        if X.intersection(Z) != set():
            print(f'X:{X} and Z:{Z} are not disjoint sets')
            print(f'{X.intersection(Z)} is(are) the common element(s) found in X and Z')
            return
        elif Z.intersection(Y) != set():
            print(f'Z:{Z} and Y:{Y} are not disjoint sets')
            print(f'{Z.intersection(Y)} is(are) the common element(s) found in Z and Y')
            return
        elif X.intersection(Y) != set():
            print(f'X:{X} and Y:{Y} are not disjoint sets')
            print(f'{X.intersection(Y)} is(are) the common element(s) found in X and Y')
            return        
            
        else:
            #Get interaction graph
            g = nx.to_undirected(self.G)
            
            for y in Y:
                for x in X:
                    # Find all path from X to Y
                    Paths.append([path for path in self.find_all_paths(g, x, y) if len(path) > 1])
                    #If no paths are found exit function
                    if Paths == [[]]:
                        return  
                    
            for paths in Paths:
                for path in paths:
                    #print("Path: ", path)
                    if len(path) == 2:
                        path_state.append(["Active"])
                        break
                    else:
                        for triples in self.trigram_path(path):
                            #Check if:
                            #         (a) Sequential
                            #         (b) Divergent
                            #         (c) Convergent
                        
                            #Look for parent of middle node in the triple
                            #print(triples)
                            parents  = list(self.G.predecessors(triples[1]))
                            children = list(self.G[triples[1]])
                            flag     = 0
                            #<-W->
                            if triples[0] in parents and triples[2] in parents:
                               # print("Convergent")
                                if triples[1] not in Z:
                                    if len(children) == 0:
                                        #print("Inactive")
                                        valve_state.append("Inactive")
                                    else:
                                        for child in children:
                                            if child in Z:
                                                flag = 1
                                        if flag == 1:
                                            #print("Active")
                                            valve_state.append("Active")
                                        else:
                                            #print("Inactive")
                                            valve_state.append("Inactive")
                                
                                else:
                                    #print("Active")
                                    valve_state.append("Active")
                            #->W->
                            elif (triples[0] in parents and triples[2] in children) or (triples[2] in parents and triples[0] in children):
                                #print("Sequential")
                                if triples[1] in Z:
                                    #print("Inactive")
                                    valve_state.append("Inactive")
                                else:
                                    #print("Active")
                                    valve_state.append("Active")
                            #->W<-        
                            elif (triples[0] in children and triples[2] in children) or (triples[2] in children and triples[0] in children):
                                #print("Divergent")
                                if triples[1] in Z:
                                    #print("Inactive")
                                    valve_state.append("Inactive")
                                else:
                                    #print("Active")
                                    valve_state.append("Active")
                        
                    path_state.append(valve_state)
                    valve_state = []
                    #print(path_state)
            
            number_of_inactive_paths = 0
            number_of_paths = len(path_state)
            
            for valve_state in path_state:
                if 'Inactive' in valve_state:
                    number_of_inactive_paths += 1
            
            if number_of_inactive_paths == number_of_paths:
                return True
            else:
                return False
    
    #def summation(self, Parent, child, value):
    
        #cpt = cpt.drop(columns = list(Parent))
        
        #aggregation_functions = {'p': 'sum'}
        #updated_cpt = cpt.groupby(cols).aggregate(aggregation_functions).reset_index()  
        
        #return updated_cpt
    '''
    Given a set of variables X in the Bayesian network, compute a good ordering for
    elimination of X based on the min-degree heuristics (3pts) and the min-fill heuristics (3pts).
    (Hint: you get the interaction graph ”for free” from the BayesNet class )
    '''
    '''
    Ordering:
    (a) Min-Degree
    (b) Min-Fill
    (c) Random-Order
    '''
    '''
    (a) Min-Degree
    '''
    def min_degree(self, Graph, Q, E):
        
        graph = Graph.to_undirected()
        variables = set(graph.nodes()) - set(Q.keys()) - set(E.keys())
        order = []
        var_with_num_of_nei = dict(graph.degree(variables))

        while len(var_with_num_of_nei) > 0:
            minimal = min(var_with_num_of_nei, key = var_with_num_of_nei.get)
            order.append(minimal)
            neighbors = list(graph.neighbors(minimal))
    
            for X in neighbors:
                for Y in neighbors:
                    if X != Y:
                        if not graph.has_edge(X, Y) or not graph.has_edge(Y, X):
                            graph.add_edge(X, Y)
    
            graph.remove_node(minimal)
            var_with_num_of_nei.pop(minimal)
        
        #Order is a queue
        return order        
    '''
    Network Pruning
    '''    
    def network_pruning(self, Q, E):
        
        if type(Q) != dict:
            return "Q not of type Set"
        elif type(E) != dict:
            return "E not of type Dictonary"
        #elif Q == {}:
            #return "Q is empty!"
        else:
            g = deepcopy(self.G)
            for variable in set(Q.keys()).union(set(E.keys())):
                if variable not in set(g.nodes()):
                    return "Q or E not in Graph"
            else:
                '''
                   Node-Pruning
                   -> Find leaf node not part of Q U E in the graph
                   -> Remove the node
                '''
                q_U_e = set(Q.keys()).union(set(E.keys()))
                flag  = True
                
                while flag:
                    leaves = [node for node in g.nodes() if list(g[node]) == []]
                    
                    if set(leaves).intersection(q_U_e) == set(leaves):
                        flag = False
                    
                    for leaf_node in leaves:
                        if leaf_node not in q_U_e:
                            g.remove_node(leaf_node)
                
                if E != {}:
                    e = set(E.keys())
                    children = []
                    
                    for node in e:
                        if list(g[node]) != []:
                            for child in list(g[node]):
                                children.append(child)
                                g.remove_edge(node, child)
                                
                                if E[node] != None:
                                    cpt = self.G.nodes[child]['cpt']
                                    cols = [x for x in list(set(cpt.columns) - {child}) if x != 'p']
                                    
                                    cpt = cpt.drop(cpt[cpt[node] != E[node]].index).reset_index()
                                    cpt = cpt.drop(columns = ['index', node])
                                    
                                    g.nodes[child]['cpt'] = cpt
         
        return g
    
    def multiply(self, Graph_, node_, factor_):
        
        temp_factor = factor_[0]
        p_factor    = 1
        cols        = []
        
        for factor in factor_[1:]:
            temp_factor_cols = set(temp_factor.columns) - {'p'}
            factor_cols = set(factor.columns) - {'p'}
            common_cols = list(temp_factor_cols.intersection(factor_cols))
            temp_factor = pd.merge(left = factor, right = temp_factor, on = common_cols ,how = 'inner')
        
        for col in temp_factor.columns:
            #print(col)
            if len(col) != 1:
                if 'p' in col[0] and '_' in col[1]:
                    cols.append(col)
        
        for col in cols:
            p_factor = p_factor*temp_factor[col]
        
        temp_factor['p'] = p_factor
        temp_factor = temp_factor.drop(columns = cols)
        
        return temp_factor
    
    def factor_multiply(self, df_1, df_2, var):
        
        new_factor = pd.merge(left = df_1, right = df_2, on = var, how = 'inner') 
        
        p_factor   = 1
        cols       = []
        
        for col in new_factor.columns:
            if len(col) != 1:
                if 'p' in col[0] and '_' in col[1]:
                    cols.append(col)
        
        for col in cols:
            p_factor = p_factor*new_factor[col]
        
        new_factor['p'] = p_factor
        new_factor = new_factor.drop(columns = cols)
        
        return new_factor
    
    def summation(self, cpt_, node_):
        cpt = cpt_.drop(columns = node_)
        cpt_cols = set(cpt.columns)
        cpt_cols = list(cpt_cols - {'p'})
        
        aggregation_functions = {'p': 'sum'}
        updated_cpt = cpt.groupby(cpt_cols).aggregate(aggregation_functions).reset_index()
        
        return updated_cpt
        
    
    def marginal_distribution(self, Q, E, order = 'min-fill'):
        
        E_nodes = set(E.keys())
        Q_nodes = set(Q.keys())
        
        if E_nodes.intersection(Q_nodes) != set():
            return "Q and E not disjoint"
        
        G_    = self.network_pruning(Q, E)
        #self.see_Graph(G_)
        order = self.min_degree(G_, Q, E)
        
        flag  = True
        
        if E != {}:
            Evidence_variables = list(E.keys())
            
            for e in Evidence_variables:
                if E[e] != None:
                    cpt = G_.nodes[e]['cpt']
                    cpt = cpt.drop(cpt[cpt[e] != E[e]].index).reset_index()
                    cpt = cpt.drop(columns = ['index'])
                    
                    G_.nodes[e]['cpt'] = cpt            
        
        if Q != {}:
            Query_variables = list(Q.keys())
            for e in Query_variables:
                if Q[e] != None:
                    cpt = G_.nodes[e]['cpt']
                    cpt = cpt.drop(cpt[cpt[e] != Q[e]].index).reset_index()
                    cpt = cpt.drop(columns = ['index'])
                    G_.nodes[e]['cpt'] = cpt
              
        if order == []:
            if Q != {}:
                nodes = list(Q.keys())
            elif E != {}:
                nodes = list(E.keys())

            temp_cpt = G_.nodes[nodes.pop(0)]['cpt']
            temp_cpt_cols = set(temp_cpt.columns) - {'p'}
            
            for node in nodes:
                node_cpt = G_.nodes[node]['cpt']
                node_cpt_cols = set(node_cpt.columns) - {'p'}
                #print(node_cpt)
                common_cols   = list(node_cpt_cols.intersection(temp_cpt_cols))
                if common_cols == []:
                    temp_cpt = pd.merge(left = temp_cpt, right = node_cpt, how = 'cross')
                    p_factor   = 1
                    cols       = []
                    
                    for col in temp_cpt.columns:
                        if len(col) != 1:
                            if 'p' in col[0] and '_' in col[1]:
                                cols.append(col)
                    
                    for col in cols:
                        p_factor = p_factor*temp_cpt[col]
                    
                    temp_cpt['p'] = p_factor
                    temp_cpt = temp_cpt.drop(columns = cols)
                
                else:
                    temp_cpt      = self.factor_multiply(temp_cpt, node_cpt, common_cols)
                    
            return temp_cpt
    
        #print(factor)
        factor = {}
        factor_cpt = []
        visited_cpt = []
        
        for node_to_be_deleted in order:
            factor[node_to_be_deleted] = []
        
        for node_to_be_deleted in order:
            for node in G_.nodes():
                cpt = G_.nodes[node]['cpt']
                cpt_cols = cpt.columns
                if node_to_be_deleted in cpt_cols:
                    if list(cpt_cols) not in visited_cpt:
                        visited_cpt.append(list(cpt_cols))
                        if factor[node_to_be_deleted] == []:
                            factor_cpt.append(cpt)
                        else:
                            factor[node_to_be_deleted].append(cpt)
                            
            factor[node_to_be_deleted] = factor_cpt
            factor_cpt = []       
        
        order_ = deepcopy(order)
        Factor  = []
        visited = []
        print(order)
        for node_to_be_deleted in order:
            visited.append(node_to_be_deleted)
            
            cpt = self.multiply(G_, node_to_be_deleted, factor[node_to_be_deleted])
            cpt = self.summation(cpt, node_to_be_deleted)
            

        new_cpt = G_.nodes[order_.pop(0)]['cpt']
        cols_ = set(new_cpt.columns) - {'p'}
       
        while order_ != []:

            node_to_be_deleted = order_.pop(0)
            cpt = G_.nodes[node_to_be_deleted]['cpt']
            cpt_column = set(cpt.columns) - {'p'}
            
            common_nodes = list(cols_.intersection(cpt_column))
            
            if common_nodes != []:
                new_cpt = self.factor_multiply(new_cpt, cpt, common_nodes)
            else:
                new_cpt = pd.merge(left = new_cpt, right = cpt, how = 'cross')
                p_factor   = 1
                cols       = []
                
                for col in new_cpt.columns:
                    if len(col) != 1:
                        if 'p' in col[0] and '_' in col[1]:
                            cols.append(col)
                
                for col in cols:
                    p_factor = p_factor*new_cpt[col]
                
                new_cpt['p'] = p_factor
                new_cpt = new_cpt.drop(columns = cols)                
                
           
                
        #print(new_cpt)'''
        #return new_cpt
        
sprinkler_problem = '/Users/kms/Desktop/KR/Assignment 2/KR21_project2-main/testing/sprinkler_problem.BIFXML'
markov_problem = '/Users/kms/Desktop/KR/Assignment 2/KR21_project2-main/testing/markov_problem.BIFXML'
BN = BNReasoner(sprinkler_problem)
'''
-> Dictonary X = {node1:evidence, node2:evidence,...}; 
-> Dictonary Z = {node5:evidence, node6:evidence,...}; 
-> evidence = None if no evidence else True/False
'''
x  = BN.marginal_distribution({'wet-grass':None}, {})
print(x)
#BN.see_Graph(x)
#BN.summation('bowel-problem', 'dog-out',True)
#BN.see_Graph('default')
        
