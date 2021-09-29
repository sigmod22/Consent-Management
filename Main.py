import concurrent

from sklearn.ensemble import RandomForestClassifier

from ActiveConsentEvaluation import ActiveConsentEvaluation
from BooleanEvaluationModule import *
from KnownProbesRepository import KnownProbesRepository
from LearnerModule import LearnerModule, LC, LAL, Learn_Once, No_Learning
from ProbeSelectorModule import SimpleMultiplicationWithIntentionalFading, UtilityOnly
from Scenario import Scenarios, H1B, H1B_S3, TPCH_Q8


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def get_variables(ls_dnf):
    def foo(list_exps_dnf):
        variables = set()

        for exp in list_exps_dnf:
            variables = variables.union(set(list(exp.get_symbols())))

        return list(variables)

    k = 10
    chunks = chunkify(ls_dnf, k)
    threads_list = list()
    results = set()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(k):
            t = executor.submit(foo, chunks[i])
            threads_list.append(t)

        for t in threads_list:
            results = results.union(set(t.result()))

    results = list(set(results)).copy()
    # print(len(results))

    return results
def randomize_known_probes(scen, init_number):
    variables_indices = []
    variables=get_variables(scen.get_ls_dnf())
    for concept in variables:
        transaction_num = int(str(concept).split('v')[-1])
        index_X = scen.index_of(scen.get_X(), transaction_num)
        variables_indices.append(index_X)
    initial_indices=np.random.choice(list(set(range(2,scen.get_X().shape[0])).difference(set(variables_indices))), size=init_number, replace=False)
    return initial_indices

def Online_variant(repo,scen):
    RO_Algorithm = RO()
    Q_Value_Algorithm = Q_Value()
    Heuristic_Allen = General()
    BooleanEvaluationModule_Online = BooleanEvaluationModule(BE_algorithm=Q_Value_Algorithm)
    ProbeSelectorModule = UtilityOnly()
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100),uncertainty_estimator=LC())
    architecture_1=ActiveConsentEvaluation(learnerModule,BooleanEvaluationModule_Online,ProbeSelectorModule)



    idx,truth_value=architecture_1.Evaluate_consent(repo,scen)
    print(idx)
    print(truth_value)


def variant_LC_plus_CtU(repo,scen):
    RO_Algorithm = RO()
    Q_Value_Algorithm = Q_Value()
    Heuristic_Allen = General()
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100), uncertainty_estimator=LC())
    BooleanEvaluationModule_Q_Value = BooleanEvaluationModule(BE_algorithm=Q_Value_Algorithm)
    ProbeSelectorModule = SimpleMultiplicationWithIntentionalFading(scen.get_variables())
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Q_Value, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)



def variant_LAL_plus_CtU(repo,scen):
    Q_Value_Algorithm = Q_Value()
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100), uncertainty_estimator=LAL())
    BooleanEvaluationModule_Q_Value = BooleanEvaluationModule(BE_algorithm=Q_Value_Algorithm)
    ProbeSelectorModule = SimpleMultiplicationWithIntentionalFading(scen.get_variables())
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Q_Value, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)



def Offline_Variant(repo, scen):
    RO_Algorithm = RO()
    Q_Value_Algorithm = Q_Value()
    Heuristic_Allen = General()
    learnerModule = Learn_Once(classifier=RandomForestClassifier(n_estimators=100))
    BooleanEvaluationModule_Q_Value = BooleanEvaluationModule(BE_algorithm=Q_Value_Algorithm)
    ProbeSelectorModule = UtilityOnly()
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Q_Value, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)



def Greedy_Variant(repo, scen):
    Greedy_algorithm=Greedy()
    learnerModule = No_Learning()
    BooleanEvaluationModule_Greedy = BooleanEvaluationModule(BE_algorithm=Greedy_algorithm)
    ProbeSelectorModule = UtilityOnly()
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Greedy, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)

def EP_Variant(repo, scen):
    Q_Value_Algorithm=Q_Value()
    learnerModule = No_Learning()
    BooleanEvaluationModule_Greedy = BooleanEvaluationModule(BE_algorithm=Q_Value_Algorithm)
    ProbeSelectorModule = UtilityOnly()
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Greedy, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)

if __name__ == '__main__':
    RO_Algorithm = RO()
    Q_Value_Algorithm = Q_Value()
    Heuristic_Allen = General()
    
    #scen=Scenarios(H1B_S3())
    scen=Scenarios(TPCH_Q8(tree_number=8))
    initial_idx = randomize_known_probes(scen, 80)
    scen.experiment_init(initial_idx)
    repo = KnownProbesRepository(X_train=(scen.get_X())[initial_idx].astype(int), y_train= (scen.get_y())[initial_idx].astype(int))

    #variant_LC_plus_CtU()
    #Online_variant(repo,scen)
    #Offline_Variant(repo,scen)
    #Greedy_Variant(repo,scen)
    variant_LAL_plus_CtU(repo,scen)
    #EP_Variant(repo,scen)



