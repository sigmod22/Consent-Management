import itertools

from boolean import boolean


class BooleanHelper:
    def convert_list_to_and_term(self, mylist):
        algebra = boolean.BooleanAlgebra()

        mylist = str(mylist).replace('[', "")
        mylist = (mylist).replace(']', "")
        mylist = (mylist).replace("'", "")
        mylist = (mylist).replace(',', ' and ')
        return algebra.parse(mylist)

    def check_evaluation(self, listExp_dnf, with_counter=False):
        counter=0
        for exp in listExp_dnf:
            if (len(exp.get_symbols()) == 0):
                counter+=1

        if with_counter==True:
            return (counter == len(listExp_dnf) ), listExp_dnf, counter
        else:
            return (counter == len(listExp_dnf) ), listExp_dnf

    def convert_list_to_or_term(self, mylist):
        algebra = boolean.BooleanAlgebra()

        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        if len(mylist) == 0:
            return FALSE
        mylist = str(mylist).replace('[', "")
        mylist = (mylist).replace(']', "")
        mylist = (mylist).replace("'", "")
        mylist = (mylist).replace(',', ' or ')
        return algebra.parse(mylist)

    def get_permutations(self, ls: list, k: int):
        algebra = boolean.BooleanAlgebra()
        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        permutations = list(itertools.combinations(ls, k))
        if len(permutations)>1:
            form = self.convert_list_to_and_term(list(permutations[0]))
        else:
            form=self.convert_list_to_and_term(permutations)
        for i in range(1, len(permutations)):
            form = OR(form, self.convert_list_to_and_term(list(permutations[i])))
        return form
    def assign_value_in_formulas(self,variables_expressions_mapper, listExp_dnf,listExp_cnf, concept, value):
        algebra = boolean.BooleanAlgebra()
        TRUE, FALSE, NOT, AND, OR, symbol = algebra.definition()
        expressions_indices=variables_expressions_mapper[concept].copy()
     
        for exp_num in expressions_indices:
            temp_dnf = listExp_dnf[exp_num]
            temp_cnf = listExp_cnf[exp_num]
            if concept in list(temp_dnf.get_symbols()):
                old_variables=temp_dnf.get_symbols().copy()
                listExp_dnf[exp_num] = temp_dnf.subs({concept: algebra.parse((value))}, simplify=True)
                listExp_cnf[exp_num] = temp_cnf.subs({concept: algebra.parse((value))}, simplify=True)

                if len(listExp_dnf[exp_num].get_symbols())==0:#The expression was evaluated
                    temp_old_variables=list(set(old_variables.copy()))
                    for var in temp_old_variables:
                        if str(var) !=str(concept):


                            variables_expressions_mapper[var].remove(exp_num)

        del variables_expressions_mapper[concept]

        return listExp_dnf, listExp_cnf,variables_expressions_mapper

