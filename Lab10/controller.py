from model import FuzzyRule


class Controller():
    def __init__(self, repository):
        self.repository = repository
        self.input_variables = {}
        self.output_variables = {}
        self.rules = []

    def add_input_variable(self, variable):
        self.input_variables[variable.name] = variable

    def add_output_variable(self, variable):
        self.output_variables[variable.name] = variable

    def get_input_variable(self, name):
        return self.input_variables[name]

    def get_output_variable(self, name):
        return self.output_variables[name]

    def clear_output_distributions(self):
        for x in self.output_variables:
            self.output_variables[x].clear_output_distribution()

    def add_rule(self, input_clauses, output_clauses):
        new_rule = FuzzyRule()

        for var_name, set_name in input_clauses.items():
            var = self.get_input_variable(var_name)
            f_set = var.get_set(set_name)
            new_rule.add_input_clause(var, f_set)

        for var_name, set_name in output_clauses.items():
            var = self.get_output_variable(var_name)
            f_set = var.get_set(set_name)
            new_rule.add_output_clause(var, f_set)

        self.rules.append(new_rule)

    def evaluate_output(self, input_values):
        self.clear_output_distributions()
        for input_name, input_value in input_values.items():
            self.input_variables[input_name].fuzzify(input_value)
        for rule in self.rules:
            rule.evaluate()
        output = {}
        for output_var_name, output_var in self.output_variables.items():
            output[output_var_name] = output_var.get_crisp_output()
        return output

    def get_input_from_file(self):
        return self.repository.get_input_data()

    def store_output(self, result):
        self.repository.store_data(result)
