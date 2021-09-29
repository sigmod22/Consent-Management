import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class LearnerModule:
    def __init__(self,classifier,uncertainty_estimator):
        self.clf=classifier
        self.uncertainty_estimator=uncertainty_estimator
    def run(self,repo,scen):
        self.clf.fit(X=repo.X_train[:, 1:], y=repo.y_train.ravel())
        pool_probabilities = self.clf.predict_proba(scen.X_pool[:, 1:])[:, 1]
        variables_probabilities = scen.convert_pool_probabilities_to_variables_probabilities(pool_probabilities)
        uncertainties=self.uncertainty_estimator.extract_uncertainty(repo,scen.X_pool,self.clf,pool_probabilities)
        return pool_probabilities,variables_probabilities,uncertainties

class Learn_Once(LearnerModule):
    def __init__(self, classifier,uncertainty_estimator=None):
        if uncertainty_estimator==None:
            uncertainty_estimator=LC()
        super().__init__(classifier,uncertainty_estimator=uncertainty_estimator)
        self.training_lock=False

    def run(self,repo,scen):
        if self.training_lock==False:
            self.clf.fit(X=repo.X_train[:, 1:], y=repo.y_train.ravel())
            self.pool_probabilities = self.clf.predict_proba(scen.X_pool[:, 1:])[:, 1]
            self.variables_probabilities = scen.convert_pool_probabilities_to_variables_probabilities(self.pool_probabilities)
            self.uncertainties = self.uncertainty_estimator.extract_uncertainty(repo, scen.X_pool, self.clf, self.pool_probabilities)
            self.training_lock=True
        return self.pool_probabilities, self.variables_probabilities, self.uncertainties

class No_Learning(LearnerModule):
    def __init__(self):
       
        super().__init__(classifier=None,uncertainty_estimator=None)

    def run(self,repo,scen):
        self.pool_probabilities = np.ones(scen.X_pool.shape[0]).dot(0.5)
        self.variables_probabilities = scen.convert_pool_probabilities_to_variables_probabilities(self.pool_probabilities)
        self.uncertainties = np.ones(scen.X_pool.shape[0])
        return self.pool_probabilities, self.variables_probabilities, self.uncertainties


class UncertaintyEstimator:
    def extract_uncertainty(self, repo, X_pool, learnerd_Model, probabilities):
        pass
class LC(UncertaintyEstimator):

    def extract_uncertainty(self, repo, X_pool, learnerd_Model, probabilities):
        probabilities_new = np.array(
            list(zip(np.array(1 - np.array(probabilities).T), np.array(np.array(probabilities).T))))
        uncertainty = 1 - np.max(probabilities_new, axis=1)
        return uncertainty


class LAL(UncertaintyEstimator):
    def __init__(self):
        self.LAL_model = self.load_learned_model()

    def load_learned_model(self):
        fn = 'LAL-iterativetree-simulatedunbalanced-big.npz'
        filename = fn
        regression_data = np.load(filename)
        regression_features = regression_data['arr_0']
        regression_labels = regression_data['arr_1']
        lalModel = RandomForestRegressor(n_estimators=1000, max_depth=40, max_features=6, oob_score=True,
                                         n_jobs=1)
        lalModel.fit(regression_features, np.ravel(regression_labels))
        return lalModel

    def extract_uncertainty(self, repo, X_pool, learnerd_Model, probabilities):
        model = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=8)
        model = model.fit(repo.X_train, repo.y_train)
        unknown_data = X_pool
        known_labels = repo.y_train
        n_lablled = repo.y_train.shape[0]
        n_dim = repo.X_train.shape[1]

        temp = np.array([tree.predict_proba(unknown_data)[:, 0] for tree in model.estimators_])
        # - average and standard deviation of the predicted scores
        f_1 = np.mean(temp, axis=0)
        f_2 = np.std(temp, axis=0)
        # - proportion of positive points
        f_3 = (sum(known_labels > 0) / n_lablled) * np.ones_like(f_1)
        # the score estimated on out of bag estimate
        f_4 = model.oob_score_ * np.ones_like(f_1)
        # - coeficient of variance of feature importance
        f_5 = np.std(model.feature_importances_ / n_dim) * np.ones_like(f_1)
        # - estimate variance of forest by looking at avergae of variance of some predictions
        f_6 = np.mean(f_2, axis=0) * np.ones_like(f_1)
        # - compute the average depth of the trees in the forest
        f_7 = np.mean(np.array([tree.tree_.max_depth for tree in model.estimators_])) * np.ones_like(f_1)
        # - number of already labelled datapoints
        f_8 = n_lablled * np.ones_like(f_1)

        # all the featrues put together for regressor
        LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        LALfeatures = np.transpose(LALfeatures)

        # predict the expercted reduction in the error by adding the point
        LALprediction = self.LAL_model.predict(LALfeatures)
        return LALprediction

  
    
