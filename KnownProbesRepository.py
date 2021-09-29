import numpy as np


class KnownProbesRepository():
    def __init__(self, X_train, y_train):
        self.X_train = X_train.astype(int)
        self.y_train = y_train.astype(int)
    def add_new_answer(self,query_instance,query_label):
        self.X_train = np.insert(self.X_train, 0, np.array(query_instance), axis=0)
        self.y_train = np.insert(self.y_train, 0, np.array(query_label), axis=0)