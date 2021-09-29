import operator
from statistics import median_high

import numpy as np

from BooleanEvaluationModule import symbol


class ProbeSelectorModule:
    def chooseNextProbe(self):
        pass

    def argmax_of_weights(self, weights):
        max_value = max(weights.items(), key=operator.itemgetter(1))[1]
        listOfKeys = list()
        for key, value in weights.items():
            if value == max_value:
                listOfKeys.append(key)
        symbol_ls = [int(str(val).split('v')[-1]) for val in listOfKeys]
        variable = symbol("v" + str(int(median_high(symbol_ls))))
        return (variable)



class UtilityOnly(ProbeSelectorModule):
    def chooseNextProbe(self,scen,utilities,uncertainties):
        weights=scen.convert_pool_probabilities_to_variables_probabilities(utilities)
        var=self.argmax_of_weights(weights)
        query_idx=scen.index_of(scen.X_pool,int(str(var).split("v")[-1]))
        value = max(utilities)
        query_label = scen.y_pool[query_idx]
        concept = scen.convert_instance_to_variable(scen.X_pool[query_idx])

        return query_idx, scen.X_pool[query_idx], value, query_label,concept


class UncertaintyOnly(ProbeSelectorModule):
    def chooseNextProbe(self,scen,utilities,uncertainties):
        query_idx = np.argmax(uncertainties)
        value = max(uncertainties)
        query_label = scen.y_pool[query_idx]
        concept = scen.convert_instance_to_variable(scen.X_pool[query_idx])

        return query_idx, scen.X_pool[query_idx], value, query_label,concept

class SimpleMultiplicationWithIntentionalFading(ProbeSelectorModule):
    def __init__(self,variables):
        self.last_uncertainties = []
        self.variables=variables
        self.variables_fading_counter = {int(str(var).split("v")[-1]): 0 for var in self.variables}


    def chooseNextProbe(self,scen,utilities,uncertainties):
        ones = np.ones(scen.X_pool.shape[0])

        uncertainties = self.calculate_intentional_fading(uncertainties, scen.X_pool)

        weights = np.multiply(ones+uncertainties,utilities)

        query_idx = np.argmax(weights)
        value = max(weights)
        query_label = scen.y_pool[query_idx]
        concept = scen.convert_instance_to_variable(scen.X_pool[query_idx])

        return query_idx, scen.X_pool[query_idx], value,query_label,concept




    def calculate_intentional_fading(self, uncertainties, x_pool):


        epsilon=0.01
        t = 5

        variables_uncertainties_mapper=dict(zip(list(map(int, list(x_pool[:,0]))), list(uncertainties)))
        updated_uncertainties=variables_uncertainties_mapper.copy()
        self.last_uncertainties.append(variables_uncertainties_mapper)
        if len(self.last_uncertainties) > t:
            del self.last_uncertainties[0]

        if len(self.last_uncertainties) == t:
            for variable in variables_uncertainties_mapper.keys():
                intervals_trn=[]
                for i in range(0, len(self.last_uncertainties)):
                    interval_i=(self.last_uncertainties[i])[variable]
                    intervals_trn.append(interval_i)
                variance=np.array(intervals_trn).var()# Variance
                if variance<epsilon:
                    self.variables_fading_counter[variable]=self.variables_fading_counter[variable]+1
                    updated_uncertainties[variable]=variables_uncertainties_mapper[variable]*(epsilon**self.variables_fading_counter[variable])
        updated_uncertainties=np.around(np.array(list(updated_uncertainties.values())),decimals=3)
        self.stop_fading=np.all(updated_uncertainties == 0)
        return updated_uncertainties
