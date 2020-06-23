import numpy as np


class Variable:
    def __init__(self, name, min_val, max_val, res):
        self.sets = {}
        self.max_val = max_val
        self.min_val = min_val
        self.res = res
        self.name = name

    def add_triangular(self, name, low, mid, high):
        new_set = FuzzySet.create_triangular(name, self.min_val, self.max_val, self.res, low, mid, high)
        self.sets[name] = new_set
        return new_set

    def add_trapezoidal(self, name, a, b, c, d):
        new_set = FuzzySet.create_trapezoidal(name, self.min_val, self.max_val, self.res, a, b, c, d)
        self.sets[name] = new_set
        return new_set

    def get_set(self, name):
        return self.sets[name]


class OutputVariable(Variable):
    def __init__(self, name, min_val, max_val, res):
        super().__init__(name, min_val, max_val, res)
        self.output_distribution = FuzzySet(name, min_val, max_val, res)
        
    def clear_output_distribution(self):
        self.output_distribution.clear_set()

    def add_rule_contribution(self, rule):
        self.output_distribution = self.output_distribution.union(rule)

    def get_crisp_output(self):
        return self.output_distribution.defuzificare()


class InputVariable(Variable):

    def __init__(self, name, min_val, max_val, res):
        super().__init__(name, min_val, max_val, res)

    def fuzzify(self, value):
        for set_name, f_set in self.sets.items():
            f_set.last_dom_value = f_set[value]


class FuzzyRule:
    def __init__(self):
        self.input = []
        self.output = []

    def add_input_clause(self, var, f_set):
        self.input.append(FuzzyClause(var, f_set))

    def add_output_clause(self, var, f_set):
        self.output.append(FuzzyClause(var, f_set))

    def evaluate(self):
        rule_strength = 1
        for ante_clause in self.input:
            rule_strength = min(ante_clause.evaluate_input(), rule_strength)
        for output_clause in self.output:
            output_clause.evaluate_output(rule_strength)


class FuzzyClause:
    def __init__(self, variable, f_set, degree=1):
        self.variable = variable
        self.set = f_set

    def variable_name(self):
        return self.variable.name

    def set_name(self):
        return self.set.name

    def evaluate_input(self):
        return self.set.last_dom_value

    def evaluate_output(self, dom):
        self.variable.add_rule_contribution(self.set.alpha_cut(dom))


class FuzzySet:
    def __init__(self, name, domain_min, domain_max, res):
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.res = res
        # returns an interval from min to max with divided in res parts
        self.domain = np.linspace(domain_min, domain_max, res+1, True)
        # returns a new array with 0 of length shape
        self.dom = np.zeros(self.domain.shape)
        self.name = name
        self.precision = 8
        self.last_dom_value = 0

    def __getitem__(self, x_val):
        return self.dom[np.abs(self.domain-x_val).argmin()]

    def __setitem__(self, x, dom):
        self.dom[np.abs(self.domain-x).argmin()] = round(dom, self.precision)

    def empty(self):
        return np.all(self._dom == 0)

    def create_trapezoidal(name, domain_min, domain_max, res, a, b, c, d):
        newSet = FuzzySet(name, domain_min, domain_max, res)
        a = newSet.adjust(a)
        b = newSet.adjust(b)
        c = newSet.adjust(c)
        d = newSet.adjust(d)
        if b == a:
            newSet.dom = np.round(np.maximum(np.minimum((d-newSet.domain)/(d-c),1), 0), newSet.precision)
        elif d == c:
            newSet.dom = np.round(np.maximum(np.minimum((newSet.domain-a)/(b-a),1), 0), newSet.precision)
        else:
            newSet.dom = np.round(np.maximum(np.minimum((newSet.domain-a)/(b-a), (d-newSet.domain)/(d-c), 1), 0), newSet.precision)
        return newSet

    def create_triangular(name, domain_min, domain_max, res, a, b, c):
        newSet = FuzzySet(name, domain_min, domain_max, res)
        a = newSet.adjust(a)
        b = newSet.adjust(b)
        c = newSet.adjust(c)
        if b == a:
            newSet.dom = np.round(np.maximum((c-newSet.domain)/(c-b), 0), newSet.precision)
        elif b == c:
            newSet.dom = np.round(np.maximum((newSet.domain-a)/(b-a), 0), newSet.precision)
        else:
            newSet.dom = np.round(np.maximum(np.minimum((newSet.domain-a)/(b-a), (c-newSet.domain)/(c-b)), 0), newSet.precision)
        return newSet

    def adjust(self, x):
        return self.domain[np.abs(self.domain-x).argmin()]

    def clear_set(self):
        self.dom.fill(0)

    def alpha_cut(self, val):
        result = FuzzySet(self.name, self.domain_min, self.domain_max, self.res)
        result.dom = np.minimum(self.dom, val)
        return result

    def union(self, f_set):
        result = FuzzySet(self.name, self.domain_min, self.domain_max, self.res)
        result.dom = np.maximum(self.dom, f_set.dom)
        return result


    def defuzificare(self):
        num = np.sum(np.multiply(self.dom, self.domain))
        den = np.sum(self.dom)
        return num/den
