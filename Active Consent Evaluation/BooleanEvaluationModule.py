import math
from statistics import median, median_low, median_high
from fractions import Fraction
from decimal import Decimal
from collections import Counter
import boolean
import operator
import pandas as pd
import itertools
import  numpy as np

from BooleanHelpers import BooleanHelper

algebra = boolean.BooleanAlgebra()
TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
class BooleanEvaluationAlgorithm:
    def chooseNextConcept(self):
        pass
    def argmax_of_weights(self):
        pass
class BooleanEvaluationModule:
    def __init__(self, BE_algorithm:BooleanEvaluationAlgorithm):
        self.utility_algorithm = BE_algorithm
        
    def extract_utility(self,scen, probabilities):
        utilities=self.utility_algorithm.chooseNextConcept( scen.listExp_dnf,scen.listExp_cnf,scen.variables, probabilities,scen.variables_expressions_mapper)
        utilities=scen.convert_variables_weight_to_pool_weights(utilities)

        return utilities


class Q_Value(BooleanEvaluationAlgorithm):
    def calc_mi_li_all_expressions(self, listExp_dnf, listExp_cnf):
        mi_li = 0
        ls_mi_li = []
        for exp_i in range(len(listExp_dnf)):
            dnf_i = listExp_dnf[exp_i]
            cnf_i = listExp_cnf[exp_i]
            mi = self.calcMi(cnf_i)
            li = self.calcLi(dnf_i)
            mi_li += mi * li
            ls_mi_li.append(mi * li)
            # print("mi=" + str(mi))
        return mi_li, ls_mi_li
    def cnf(self,expList):
        return algebra.cnf(expList)


    def dnf(self,expList):
        return algebra.dnf(expList)


    def calcMi(self,exp):  # calc number of terms (conjunctions) 
        if (len(exp.get_symbols()) == 0):
           return 0
        return str(exp).count('&') + 1

    def calcLi(self,exp):  # calc number of clauses (disjunctions)
        if (len(exp.get_symbols()) == 0):
            return 0
        return str(exp).count('|') + 1

    def calcG(self,var,listExp_dnf, listExp_cnf,ls_mi_li,probabilities,variables_expressions_mapper):
        gi=0
        indices=variables_expressions_mapper[var].copy()
        for exp_i in indices:
            dnf_i = listExp_dnf[exp_i]
            cnf_i = listExp_cnf[exp_i]
            mi_li = ls_mi_li[exp_i]
            p_i = probabilities[var]
            dnfExp = self.condensedEXP_0(dnf_i, var)
            cnfExp = self.condensedEXP_0(cnf_i, var)

            temp_gi0 = self.calcLi(dnfExp)
            temp_gi1 = self.calcMi(cnfExp)
            gi0 = temp_gi0 * temp_gi1
            dnfExp = self.condensedEXP_1(dnf_i, var)
            cnfExp = self.condensedEXP_1(cnf_i, var)

            temp_gi0 = self.calcLi(dnfExp)
            temp_gi1 = self.calcMi(cnfExp)
            gi1 = temp_gi0 * temp_gi1


            gi +=(1-p_i)* (mi_li- (gi0)) + (p_i)* (mi_li- (gi1))





        return gi


    def condensedEXP_1(self, exp, concept):
        algebra = boolean.BooleanAlgebra()

        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()

        temp_dnf = exp

        temp_dnf = exp.subs({symbol(str(concept)): TRUE}, simplify=True)
        return temp_dnf
    def condensedCNF_0(self, cnf, concept):
        algebra = boolean.BooleanAlgebra()
        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        temp_cnf = cnf
        temp_cnf = cnf.subs({symbol(str(concept)): FALSE}, simplify=True)
        return temp_cnf

    def condensedEXP_0(self, exp, concept):
        algebra = boolean.BooleanAlgebra()

        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        temp_dnf = exp
        temp_dnf = exp.subs({symbol(str(concept)): FALSE}, simplify=True)
        return temp_dnf


    def condensedCNF_1(self,cnf, concept):
        temp_cnf = cnf
        temp_cnf = cnf.subs({symbol(str(concept)): TRUE}, simplify=True)

        return temp_cnf

    def calc_li_all_expressions(self, listExp_dnf):
        mi_li = 0
        ls_mi_li = []
        for exp_i in range(len(listExp_dnf)):
            dnf_i = listExp_dnf[exp_i]
            li = self.calcLi(dnf_i)
            mi_li +=li
            ls_mi_li.append( li)
        return mi_li, ls_mi_li

    def chooseNextConcept(self, listExp_dnf,listExp_cnf,variables, probabilities,variables_expressions_mapper={}):
        index = 1
        sum_mi_li,ls_mi_li=self.calc_mi_li_all_expressions(listExp_dnf,listExp_cnf)
        generalDic = {}
        for var in variables:
                utility = self.calcG(var,listExp_dnf,listExp_cnf,ls_mi_li,probabilities,variables_expressions_mapper)
                generalDic[var] = utility
        return generalDic




    def argmax_of_weights(self,weights):
        max_value = max(weights.items(), key=operator.itemgetter(1))[1]
        listOfKeys = list()
        for key, value in weights.items():
            if value == max_value:
                listOfKeys.append(key)

        symbol_ls = [int(str(val).split('v')[-1]) for val in listOfKeys]
        variable= symbol("v"+str(int(median_high(symbol_ls))))
        return(variable)





class General(BooleanEvaluationAlgorithm):

    def calculate_lcd(self, terms_Weights):
        terms_Weights=list(np.around(terms_Weights, decimals=3))

        ls_d=[Fraction(we).limit_denominator(1000).denominator for we in terms_Weights]
        result=np.lcm.reduce(ls_d)
        print(result)
        return result


    def __init__(self):
        self.roundRobin_0_or_1 =True
    def cnf(self,expList):
        return algebra.cnf(expList)


    def dnf(self,expList):
        return algebra.dnf(expList)


    def calcMi(self,exp):
        if (len(exp.get_symbols()) == 0):
           return 0
        return str(exp).count('&') + 1

    def calcLi(self,exp):
        if (len(exp.get_symbols()) == 0):
            return 0
        return str(exp).count('|') + 1

    def calcG(self,var, listExp_dnf, probabilities,variables_expressions_mapper,ls_li):
            gi = 0
            indices = variables_expressions_mapper[var].copy()
            for exp_i in indices:
                dnf_i = listExp_dnf[exp_i]

                li = ls_li[exp_i]

                p_i = probabilities[var]

                dnfExp = self.condensedEXP_0(dnf_i, var)

                temp_gi0 = self.calcLi(dnfExp)
                gi0 = temp_gi0

                dnfExp = self.condensedEXP_1(dnf_i, var)

                temp_gi0 = self.calcLi(dnfExp)
                gi1 = temp_gi0


                gi += (1 - p_i) * (li - (gi0)) + (p_i) * (li - (gi1))

            return gi

    def condensedEXP_1(self, exp, concept):
        algebra = boolean.BooleanAlgebra()

        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()

        temp_dnf = exp

        temp_dnf = exp.subs({symbol(str(concept)): TRUE}, simplify=True)
        return temp_dnf



    def condensedEXP_0(self, exp, concept):
        algebra = boolean.BooleanAlgebra()

        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        temp_dnf = exp
        temp_dnf = exp.subs({symbol(str(concept)): FALSE}, simplify=True)
        return temp_dnf



    def calc_li_all_expressions(self, listExp_dnf):
        mi_li = 0
        ls_mi_li = []
        for exp_i in range(len(listExp_dnf)):
            dnf_i = listExp_dnf[exp_i]
            li = self.calcLi(dnf_i)
            mi_li += li
            ls_mi_li.append(li)
            # print("mi=" + str(mi))
        return mi_li, ls_mi_li



    def chooseNextConcept(self, listExp_dnf,listExp_cnf,variables, probabilities,variables_expressions_mapper={}):
        index = 1
        generalDic = {}
        if self.roundRobin_0_or_1==True:
            self.roundRobin_0_or_1=False

            generalDic=self.algo0(  listExp_dnf,variables, probabilities,variables_expressions_mapper)
        else:
            self.roundRobin_0_or_1=True
            generalDic=self.algo1( listExp_dnf, probabilities)




        return generalDic

    def calc_W(self, args,
               probabilities):
        probas = list()

        mult_of_probabilities = 1
        if args.sort_order == 25:  # OR
            for lit in args.literals:
                probas.append(probabilities[lit])
                mult_of_probabilities *= (1 - probabilities[lit])
            mult_of_probabilities = 1 - mult_of_probabilities
        else:
            for lit in args.literals:
                probas.append(probabilities[lit])
                mult_of_probabilities *= probabilities[lit]
        mult_of_probabilities = mult_of_probabilities / len(args.literals)
        return mult_of_probabilities





    def algo0(self, listExp_dnf,variables, probabilities,variables_expressions_mapper):
        index = 1
        sum_mi_li, ls_li = self.calc_li_all_expressions(listExp_dnf)
        generalDic = {}
        for var in variables:
            utility = self.calcG(var, listExp_dnf,  probabilities, variables_expressions_mapper,ls_li)
            generalDic[var] = utility
        return generalDic


    def algo1(self, listExp_dnf, probabilities):
        algebra = boolean.BooleanAlgebra()
        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        generalDic = {}

        t = dict()
        probabilities_t = dict()
        num_of_terms = 0
        for exp in listExp_dnf:

            if exp != TRUE and exp != FALSE:

                if exp.sort_order == 10:
                    t.update({symbol('t{}'.format(num_of_terms)): exp})
                    t_proba = self.calc_W(t[symbol('t{}'.format(num_of_terms))],
                                                                     probabilities)
                    probabilities_t.update({symbol('t{}'.format(num_of_terms)): t_proba + 0.001})
                    num_of_terms += 1
                    continue
                if exp.sort_order == 5:
                    t.update({symbol('t{}'.format(num_of_terms)): exp})
                    probabilities_t.update({symbol('t{}'.format(num_of_terms)): probabilities[exp] + 0.001})
                    num_of_terms += 1
                    continue

                for i in range(0, len(exp.args)):
                    t.update({symbol('t{}'.format(num_of_terms)): exp.args[i]})
                    t_proba = self.calc_W(t[symbol('t{}'.format(num_of_terms))],
                                                                     probabilities)
                    probabilities_t.update({symbol('t{}'.format(num_of_terms)): t_proba + 0.001})
                    num_of_terms += 1
        lcd=self.calculate_lcd(list(probabilities_t.values()))
        for term in t.keys():
            for var in t[term].get_symbols():
                weight = lcd * round(probabilities_t[term],3)
                weight += 1 - probabilities[var]
                generalDic[var] = max(generalDic.get(var, 0), weight)

        return generalDic


    def argmax_of_weights(self,weights):
        max_value = max(weights.items(), key=operator.itemgetter(1))[1]
        listOfKeys = list()
        for key, value in weights.items():
            if value == max_value:
                listOfKeys.append(key)
        symbol_ls = [int(str(val).split('v')[-1]) for val in listOfKeys]

        variable= symbol("v"+str(int(median_high(symbol_ls))))

        return(variable)









class RO(BooleanEvaluationAlgorithm):

    def calculate_lcd(self,terms_Weights):
        terms_Weights=list(np.around(terms_Weights, decimals=3))
        ls_d=[Fraction(we).limit_denominator(1000).denominator for we in terms_Weights]
        result= np.lcm.reduce(ls_d)
        print(result)
        return result

    def calc_W(self, args, probabilities): 
        probas = list()

        mult_of_probabilities = 1
        if args.sort_order == 25:# OR
            print("found")
            for lit in args.literals:
                probas.append(probabilities[lit])
                mult_of_probabilities *= (1 - probabilities[lit])
            mult_of_probabilities = 1 - mult_of_probabilities
        else:
            for lit in args.literals: #AND
                probas.append(probabilities[lit])
                mult_of_probabilities *= probabilities[lit]
        mult_of_probabilities=mult_of_probabilities/len(args.literals)
        return mult_of_probabilities

    def argmax_of_weights(self, weights):
        max_value = max(weights.items(), key=operator.itemgetter(1))[1]
        listOfKeys = list()
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in weights.items():
            if value == max_value:
                listOfKeys.append(key)
        symbol_ls = [int(str(val).split('v')[-1]) for val in listOfKeys]
        variable = symbol("v" + str(int(median_high(symbol_ls))))
        return (variable)


    def chooseNextConcept(self, listExp_dnf,listExp_cnf,variables, probabilities,variables_expressions_mapper={}):

        algebra = boolean.BooleanAlgebra()
        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        generalDic = {}

        t = dict()
        probabilities_t = dict()
        num_of_terms = 0
        for exp in listExp_dnf:

            if exp!=TRUE and exp!=FALSE:

                if exp.sort_order == 10:  # when there is only one term in the experssion e.g x1&x2&x3...
                    t.update({symbol('t{}'.format(num_of_terms)): exp})
                    t_proba = self.calc_W(t[symbol('t{}'.format(num_of_terms))], probabilities)
                    probabilities_t.update({symbol('t{}'.format(num_of_terms)): t_proba + 0.001})
                    num_of_terms += 1
                    continue
                if exp.sort_order == 5:  # when there is only one symbol e.g x1
                    t.update({symbol('t{}'.format(num_of_terms)): exp})
                    probabilities_t.update({symbol('t{}'.format(num_of_terms)): probabilities[exp] + 0.001})
                    num_of_terms += 1
                    continue

                for i in range(0, len(exp.args)):
                    t.update({symbol('t{}'.format(num_of_terms)): exp.args[i]})
                    t_proba = self.calc_W(t[symbol('t{}'.format(num_of_terms))], probabilities)
                    probabilities_t.update({symbol('t{}'.format(num_of_terms)): t_proba + 0.001})
                    num_of_terms += 1

        lcd=self.calculate_lcd(list(probabilities_t.values()))

        for term in t.keys():
            for var in t[term].get_symbols():
                weight = lcd * round(probabilities_t[term],3)
                weight += 1 - probabilities[var]
                generalDic[var] = max(generalDic.get(var, 0),  weight)

        return generalDic




class BaselinesAlgorithms(BooleanEvaluationAlgorithm):
    pass
class Greedy(BaselinesAlgorithms):



    def chooseNextConcept(self, listExp_dnf,listExp_cnf,variables, probabilities,variables_expressions_mapper={}):
        generalDic = {}
        for exp in listExp_dnf:
                dic_exp=dict(Counter(exp.get_symbols()))

                for key in list(dic_exp.keys()):
                    generalDic[key] = generalDic.get(key, 0) + dic_exp[key]


        ls={key:val for key,val in generalDic.items() if val>1}

        return generalDic

    def argmax_of_weights(self,weights):
        max_value = max(weights.items(), key=operator.itemgetter(1))[1]
        listOfKeys = list()
        for key, value in weights.items():
            if value == max_value:
                listOfKeys.append(key)

        symbol_ls = [int(str(val).split('v')[-1]) for val in listOfKeys]
        variable= symbol("v"+str(int(median_high(symbol_ls))))
        return(variable)

class Random_selection(BaselinesAlgorithms):
    def chooseNextConcept(self, listExp_dnf, listExp_cnf, variables, probabilities,variables_expressions_mapper={}):
        variables_weights={k:0 for k in variables}
        chosen_variable=list(np.random.choice(list(variables), size=1, replace=False))[0]
        variables_weights[chosen_variable]=1
        return variables_weights
    def argmax_of_weights(self,weights):
        max_value = max(weights.items(), key=operator.itemgetter(1))[1]
        listOfKeys = list()
        for key, value in weights.items():
            if value == max_value:
                listOfKeys.append(key)

        symbol_ls = [int(str(val).split('v')[-1]) for val in listOfKeys]
        variable= symbol("v"+str(int(median_high(symbol_ls))))

        return(variable)
