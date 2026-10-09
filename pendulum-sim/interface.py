import math
import sim_metrics

G = 9.81
STEP_SIZE = 0.01

PARAMS = [
    # name, type, min, max
    ("Initial angle (rad)", float, -math.pi / 2, math.pi / 2),
    ("Initial velocity", float, -1.0, 1.0),
    ("L", float, 0.5, 1.5),
    ("Step count", int, 1, 150),
]

def ask(name, cast, lo, hi):
    while True:
        try:
            value = cast(input(f"{name} [{lo:.2f} to {hi:.2f}]: "))
            if lo <= value <= hi:
                return value
            print(f"Must be between {lo:.2f} and {hi:.2f}.")
        except ValueError:
            print(f"Invalid input, expected a {cast.__name__}.")

def run(wb_params):
    menu = {
        "1": ("Graph a simulation", sim_metrics.graph),
        "2": ("Calculate RMSE for a simulation", sim_metrics.rmse),
        "3": ("Plot datapoints against accuracy for a simulation", sim_metrics.data_vs_accuracy),
    }

    for key, (desc, _) in menu.items():
        print(f"{key}. {desc}")

    choice = input("Enter your selection: ").strip()
    while choice not in menu:
        choice = input("Invalid selection, try again: ").strip()

    angle, velocity, length, steps = [ask(*p) for p in PARAMS]
    params = [angle, velocity, G, length, STEP_SIZE, steps]
    menu[choice][1](params, wb_params)