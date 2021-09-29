import os

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder

from BooleanEvaluationModule import algebra


class Setting:
    def get_Boolean_Provenance(self):
        pass

class TPCH_Q8(Setting):
    def read_dataset(self,tree_num):
        
        self.dataset = pd.read_csv("**** METADATA FILE- TPC-H Q8 ****", low_memory=False)
    def __init__(self, tree_number):
        self.read_dataset(tree_num=tree_number)
        self.X, self.y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]
        self.preprocess_data()
        self.dict_variables = dict(zip(list(self.X['Transaction id'].values), list(self.y)))

        self.X, self.y = self.X.to_numpy(), self.y.to_numpy()
        self.ls_dnf,self.ls_cnf= self.get_Boolean_Provenance()

    

    def preprocess_data(self):
        lb_make = LabelEncoder()
        self.X = self.X.drop(columns=['key'])
        self.variables_real_probabilities = dict(
            zip(list(self.X['Transaction id'].values), list(self.X['Probability']).copy()))
        self.X = self.X.drop(columns=['Probability'])
        self.X['c_nationkey'] = self.X['c_nationkey'].astype(str).str.replace(" ", "").astype(str)
        self.X['c_mktsegment'] = self.X['c_mktsegment'].astype(str).str.replace(" ", "").astype(str)
        if 'p_type' in self.X.columns:
            self.X['p_type'] = self.X['p_type'].astype(str).str.replace(" ", "").astype(str)
            self.X['p_mfgr'] = self.X['p_mfgr'].astype(str).str.replace(" ", "").astype(str)
            self.X['p_brand'] = self.X['p_brand'].astype(str).str.replace(" ", "").astype(str)
        self.X['o_orderstatus'] = self.X['o_orderstatus'].astype(str).str.replace(" ", "").astype(str)
        self.X['o_orderpriority'] = self.X['o_orderpriority'].astype(str).str.replace(" ", "").astype(str)
        if 's_nationkey' in self.X.columns:
            self.X['s_nationkey'] = self.X['s_nationkey'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_returnflag'] = self.X['l_returnflag'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_linestatus'] = self.X['l_linestatus'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_shipinstruct'] = self.X['l_shipinstruct'].astype(str).str.replace(" ", "").astype(str)
        self.X['l_shipmode'] = self.X['l_shipmode'].astype(str).str.replace(" ", "").astype(str)
        self.X['o_orderstatus'] = self.X['o_orderstatus'].astype(str).str.replace(" ", "").astype(str)
        self.X['c_nationkey'] = lb_make.fit_transform(self.X["c_nationkey"])
        self.X['c_mktsegment'] = lb_make.fit_transform(self.X["c_mktsegment"])
        if 'p_type' in self.X.columns:
            self.X['p_type'] = lb_make.fit_transform(self.X["p_type"])
            self.X['p_mfgr'] = lb_make.fit_transform(self.X["p_mfgr"])
            self.X['p_brand'] = lb_make.fit_transform(self.X["p_brand"])
        self.X['o_orderstatus'] = lb_make.fit_transform(self.X["o_orderstatus"])
        self.X['o_orderpriority'] = lb_make.fit_transform(self.X["o_orderpriority"])
        if 's_nationkey' in self.X.columns:
            self.X['s_nationkey'] = lb_make.fit_transform(self.X["s_nationkey"])
        self.X['l_returnflag'] = lb_make.fit_transform(self.X["l_returnflag"])
        self.X['l_linestatus'] = lb_make.fit_transform(self.X["l_linestatus"])
        self.X['l_shipinstruct'] = lb_make.fit_transform(self.X["l_shipinstruct"])
        self.X['l_shipmode'] = lb_make.fit_transform(self.X["l_shipmode"])
        self.X['o_orderstatus'] = lb_make.fit_transform(self.X["o_orderstatus"])
        self.X['type'] = lb_make.fit_transform(self.X["type"])
        self.dict_variables = dict(zip(list(self.X['Transaction id'].values), list(self.y)))
        transactions = pd.DataFrame(self.X['Transaction id']).astype(int)
    def get_Boolean_Provenance(self):
        path="TPCH_RESULTS"
        text_file_dnf = open("****BOOLEAN PROVENANCE FILE- DNF****", "r")
        lines_dnf = text_file_dnf.readlines()

        text_file_cnf = open("****BOOLEAN PROVENANCE FILE- CNF****", "r")

        lines_cnf = text_file_cnf.readlines()

        expressions_dnf = []
        expressions_cnf = []

        for line in lines_dnf:
            dnf_form = algebra.parse(line)
            expressions_dnf.append(dnf_form)

        for line in lines_cnf:
            cnf_form = algebra.parse(line)
            expressions_cnf.append(cnf_form)

        return expressions_dnf, expressions_cnf
    
class H1B(Setting):
    def __init__(self):

        self.dataset = self.read_dataset()
        self.X, self.y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]
        self.preprocess_data()
        self.dict_variables = dict(zip(list(self.X['Transaction id'].values), list(self.y)))
        self.X, self.y = self.X.to_numpy(), self.y.to_numpy()
        self.dataset = self.dataset.replace({"CERTIFIED": 1, "WITHDRAWN": 0})
    def read_dataset(self):
   
        self.dataset = pd.read_csv("**** METADATA FILE- H1B ****", low_memory=False)
        return dataset
    def preprocess_data(self):
        lb_make = LabelEncoder()
        self.X['VISA_CLASS'] = lb_make.fit_transform(self.X["VISA_CLASS"])
        self.X['EMPLOYER_NAME'] = lb_make.fit_transform(self.X["EMPLOYER_NAME"])
        self.X['EMPLOYER_STATE'] = lb_make.fit_transform(self.X["EMPLOYER_STATE"])
        self.X['EMPLOYER_COUNTRY'] = lb_make.fit_transform(self.X["EMPLOYER_COUNTRY"])
        self.X['SOC_NAME'] = lb_make.fit_transform(self.X["SOC_NAME"])
        self.X['PW_UNIT_OF_PAY'] = lb_make.fit_transform(self.X["PW_UNIT_OF_PAY"])
        self.X['PW_SOURCE'] = lb_make.fit_transform(self.X["PW_SOURCE"])
        self.X['PW_SOURCE_OTHER'] = lb_make.fit_transform(self.X["PW_SOURCE_OTHER"])
        self.X['WAGE_UNIT_OF_PAY'] = lb_make.fit_transform(self.X["WAGE_UNIT_OF_PAY"])
        self.X = DataFrameImputer().fit_transform(self.X)
        #
        #imp = Imputer(strategy="most_frequent")
        # imp.fit_transform(self.X)
        self.X['H1B_DEPENDENT'] = lb_make.fit_transform(self.X["H1B_DEPENDENT"])
        self.X['WORKSITE_STATE'] = lb_make.fit_transform(self.X["WORKSITE_STATE"])
        self.X['WILLFUL_VIOLATOR'] = lb_make.fit_transform(self.X["WILLFUL_VIOLATOR"])
        self.X['FULL_TIME_POSITION'] = lb_make.fit_transform(self.X["FULL_TIME_POSITION"])
        self.y = self.y.replace({"CERTIFIED": 1, "WITHDRAWN": 0})

    def read_Boolean_expressions_H1B(self, path, query_number, num_of_companies):
        expressions_dnf = []
        expressions_cnf = []
        regular = []
        for query in query_number:
            dnf_file = open("****BOOLEAN PROVENANCE FILE-DNF****", "r")

            lines_dnf = dnf_file.readlines()
            for line in lines_dnf:
                line = line.replace("p", "v")
                expressions_dnf.append(algebra.parse(line))
            cnf_file = open("****BOOLEAN PROVENANCE FILE- CNF****", "r")
            lines_cnf = cnf_file.readlines()
            for line in lines_cnf:
                line = line.replace("p", "v")
                expressions_cnf.append(algebra.parse(line))
        return expressions_dnf,expressions_cnf
    def get_Boolean_Provenance(self):
        pass
    

class H1B_S3(H1B):
    def __init__(self):
        super().__init__()
        self.get_Boolean_Provenance()

    def get_Boolean_Provenance(self):
        num_of_companies = 3
        num_of_uc = 1652
        queries_numbers = list(range(1, num_of_uc + 1))
        dnf_s, cnf_s = self.read_Boolean_expressions_H1B("H1B", query_number=queries_numbers,
                                                         num_of_companies=num_of_companies)
        self.ls_dnf = dnf_s
        self.ls_cnf = cnf_s
        return self.ls_dnf.copy(), self.ls_cnf.copy()


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value

        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]

                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
