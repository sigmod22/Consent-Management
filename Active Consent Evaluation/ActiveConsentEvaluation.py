import numpy as np

from BooleanEvaluationModule import BooleanEvaluationModule
from BooleanHelpers import BooleanHelper
from KnownProbesRepository import KnownProbesRepository
from LearnerModule import LearnerModule
from sklearn.ensemble import RandomForestClassifier
from boolean import boolean

from ProbeSelectorModule import SimpleMultiplicationWithIntentionalFading

class ActiveConsentEvaluation:
    def __init__(self,LearnerModule,BooleanEvaluationModule,ProbeSelectorModule):
        self.LearnerModule = LearnerModule
        self.BooleanEvaluationModule=BooleanEvaluationModule
        self.ProbeSelectorModule=ProbeSelectorModule
    def Evaluate_consent(self, repo,scen):


        idx = 0
        evaluated, truthValue = scen.check_evaluation()
        while evaluated == False:
            idx += 1
            pool_probabilities,variables_probabilities,uncertainties=self.LearnerModule.run(repo, scen)
            utilities=self.BooleanEvaluationModule.extract_utility(scen,variables_probabilities)
            query_idx, query_instance, value,query_label,concept =self.ProbeSelectorModule.chooseNextProbe(scen,utilities,uncertainties)
            
            value = scen.get_value_of_concept(concept)
            scen.assign_new_answer(concept,value)
            repo.add_new_answer(query_instance,query_label)
    
            evaluated, truthValue, counter_expressions = scen.check_evaluation(with_counter=True)
            print("The number of variables left to evaluation:{}".format(len(scen.variables)))

            print("Probe: {} , The Chosen concept: {}".format(str(idx), str(concept)))

            print(
                "The Number of expressions left to evaluation are: {}".format(
                    len(scen.listExp_dnf) - counter_expressions))
    
        return  idx, truthValue

