from controller import Controller


class View:
    def __init__(self, controller):
        self.controller = controller

    def run(self):
        inputs = self.controller.get_input_from_file()
        results = []
        for x in inputs:
            res = self.controller.evaluate_output(x)
            results.append(res)
            print("Input value is ", x, " the output is ",res)
        self.controller.store_output(results)
            
