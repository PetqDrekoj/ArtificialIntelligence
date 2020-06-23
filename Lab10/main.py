from model import OutputVariable, InputVariable
from view import View
from controller import Controller
from repository import Repository


repository = Repository("input.in", "output.in")
controller = Controller(repository)
view = View(controller)

texture = InputVariable('Texture', 0, 1, 10)
texture.add_trapezoidal('Very soft', 0, 0, 0.2, 0.4)
texture.add_triangular('Soft', 0.2, 0.4, 0.8)
texture.add_triangular('Normal', 0.3, 0.7, 0.9)
texture.add_trapezoidal('Resistant', 0.7, 0.9, 1, 1)

capacity = InputVariable('Capacity', 0, 5, 5)
capacity.add_trapezoidal('Small', 0, 0, 1, 2)
capacity.add_triangular('Medium', 1, 2.5, 4)
capacity.add_trapezoidal('High', 3, 4, 5, 5)

cycle = OutputVariable('Cycle', 0, 1, 10)
cycle.add_trapezoidal('Delicate', 0, 0, 0.2, 0.4)
cycle.add_triangular('Easy', 0.2, 0.5, 0.8)
cycle.add_triangular('Normal', 0.3, 0.6, 0.9)
cycle.add_trapezoidal('Intense', 0.7, 0.9, 1, 1)

controller.add_input_variable(texture)
controller.add_input_variable(capacity)
controller.add_output_variable(cycle)

controller.add_rule(
        {'Texture': 'Very soft', 'Capacity': 'Small'},
        {'Cycle': 'Delicate'})
controller.add_rule(
        {'Texture': 'Very soft', 'Capacity': 'Medium'},
        {'Cycle': 'Easy'})

controller.add_rule(
        {'Texture': 'Very soft', 'Capacity': 'High'},
        {'Cycle': 'Normal'})
controller.add_rule(
        {'Texture': 'Soft', 'Capacity': 'Small'},
        {'Cycle': 'Easy'})

controller.add_rule(
        {'Texture': 'Soft', 'Capacity': 'Medium'},
        {'Cycle': 'Normal'})

controller.add_rule(
        {'Texture': 'Soft', 'Capacity': 'High'},
        {'Cycle': 'Normal'})

controller.add_rule(
        {'Texture': 'Normal', 'Capacity': 'Small'},
        {'Cycle': 'Easy'})

controller.add_rule(
        {'Texture': 'Normal', 'Capacity': 'Medium'},
        {'Cycle': 'Normal'})

controller.add_rule(
        {'Texture': 'Normal', 'Capacity': 'High'},
        {'Cycle': 'Intense'})

controller.add_rule(
        {'Texture': 'Resistant', 'Capacity': 'Small'},
        {'Cycle': 'Easy'})

controller.add_rule(
        {'Texture': 'Resistant', 'Capacity': 'Medium'},
        {'Cycle': 'Normal'})

controller.add_rule(
        {'Texture': 'Resistant', 'Capacity': 'High'},
        {'Cycle': 'Intense'})

view.run()
