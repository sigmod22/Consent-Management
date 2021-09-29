import concurrent
import itertools
from queue import Queue
from threading import Thread
from BooleanHelpers import BooleanHelper

import pandas as pd
from boolean import boolean

import numpy as np

from Settings import *
from BooleanHelpers import BooleanHelper

class Scenarios:
    def __init__(self, framework:Setting):
        self.framework = framework
        self.listExp_dnf,self.listExp_cnf=self.framework.get_Boolean_Provenance()

    def get_ls_dnf(self):
        return self.listExp_dnf
    def get_ls_cnf(self):
        return self.listExp_cnf
    def experiment_init(self,initial_idx):
        variables_indices = []
        self.variables = self.get_variables_first_time()

        for concept in self.variables:
            transaction_num = int(str(concept).split('v')[-1])
            index_X = self.index_of(self.get_X(), transaction_num)
            variables_indices.append(index_X)

        self.X_pool = ((self.get_X())[list(set(variables_indices).difference(set(initial_idx)))]).astype(int)
        self.y_pool = ((self.get_y())[list(set(variables_indices).difference(set(initial_idx)))]).astype(int)
        self.pool_indices=dict(zip(list(self.X_pool[:,0]),list(range(0,self.X_pool.shape[0]))))


        self.listExp_dnf = self.get_exp_dnf()
        self.listExp_cnf = self.get_exp_cnf()        
        self.variables_expressions_mapper = self.generate_dict_variables_exps(self.listExp_dnf)
        self.variables = self.get_variables()
        amount_of_variables = len(self.variables)
  





    def index_of(self, array, transaction_code):
        if array.shape[0] == self.framework.X.shape[0]:
            return transaction_code - 1
        if array.shape[0] > 1:
            dict_indices = dict(zip(list(array[:, 0]), list(range(0, array.shape[0]))))
            return dict_indices[(transaction_code)]
        
    def convert_variable_to_instance(self, X_pool, y_pool, concept):
        index = self.index_of(X_pool, int(str(concept).split('v')[-1]))
        return X_pool[index], y_pool[index], index

    def convert_variable_to_transaction_id(self, concept):
        df = self.framework.observed_transactions.loc[
            self.framework.observed_transactions['Transaction id'] == int(str(concept).split('v')[-1])]
        rf = df['Transaction id']
        transaction_id = list(dict.fromkeys(list(rf.values)))[0]
        return transaction_id

        return False, None

    def get_value_of_concept(self, concept):
        algebra = boolean.BooleanAlgebra()

        return algebra.parse(str(int(self.framework.dict_variables[int(str(concept).split('v')[-1])])))

    def convert_peer_weight_to_transaction_weight(self, weights_):

        weights_of_transaction = {str(k).replace('v', ''): val for k, val in weights_.items()}

        return weights_of_transaction

    def convert_pool_probabilities_to_variables_probabilities(self, probabilities):
        transactions = list(self.X_pool[:, 0])
        probabilities = list(np.around(probabilities, decimals=2))

        all_concepts_probabilities = dict(zip(transactions, probabilities))

        concept_probabilities = {}
        for sym in self.variables:
            concept_probabilities.update({sym: all_concepts_probabilities[int(str(sym).split('v')[-1])]})
        return concept_probabilities

    def convert_variables_weight_to_pool_weights(self,weights):
        utility_weights = self.convert_peer_weight_to_transaction_weight(weights)
        weights = list(np.zeros(self.X_pool.shape[0]))
        for i, w in utility_weights.items():
            index = self.pool_indices[int(i)]
            if index != -1:
                weights[index] = w
        return np.array(weights)

    def convert_instance_to_variable(self, instance):
        algebra = boolean.BooleanAlgebra()
        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        return symbol("v" + str(int(instance[0])))

    def get_exp_dnf(self):
        return self.listExp_dnf.copy()

    def get_exp_cnf(self):
        return self.listExp_cnf.copy()

    def get_X(self):
        return self.framework.X

    def get_y(self):
        return self.framework.y

    def chunkify(self,lst, n):
        return [lst[i::n] for i in range(n)]

    def get_variables_first_time(self):
        def foo(list_exps_dnf):
            variables = set()

            for exp in list_exps_dnf:
                variables = variables.union(set(list(exp.get_symbols())))

            return list(variables)

        k = 10
        chunks = self.chunkify(self.listExp_dnf, k)
        threads_list = list()
        results = set()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(k):
                t = executor.submit(foo, chunks[i])
                threads_list.append(t)

            for t in threads_list:
                results = results.union(set(t.result()))

        results = list(set(results)).copy()

        return results
    def generate_dict_variables_exps(self, ls_dnf):

        variables_appearances = {var: [] for var in self.variables}

        for exp_i in range(0, len(ls_dnf)):
            for var in ls_dnf[exp_i].get_symbols():
                temp_set=list(set(list(variables_appearances[var])).union(set([exp_i])))
                variables_appearances[var]=temp_set
        return variables_appearances
    def assign_new_answer(self,variable, value):
        bh = BooleanHelper()

        self.listExp_dnf, self.listExp_cnf, self.variables_expressions_mapper = bh.assign_value_in_formulas(
            self.variables_expressions_mapper, self.listExp_dnf, self.listExp_cnf, variable, str(value))

        self.X_pool, self.y_pool = self.update_pool()

    def update_pool(self):
        self.variables = self.get_variables()
        variables_indices = []

        counter = 0
        for concept in self.variables:
            transaction_num = int(str(concept).split('v')[-1])
            index_X = self.index_of(self.get_X(), transaction_num)
            variables_indices.append(index_X)
        X_pool, y_pool = (self.get_X())[variables_indices], (self.get_y())[variables_indices]
        self.pool_indices = dict(zip(list(X_pool[:, 0]), list(range(0, X_pool.shape[0]))))

        return X_pool, y_pool

    def get_variables(self):
        temp = self.variables_expressions_mapper.copy()
        self.variables_expressions_mapper = {k: v.copy() for k, v in temp.items() if len(v) > 0}
        del temp
        return list(self.variables_expressions_mapper.keys())

    def check_evaluation(self, with_counter=False):
        bh=BooleanHelper()
        return  bh.check_evaluation(self.listExp_dnf, with_counter=with_counter)
