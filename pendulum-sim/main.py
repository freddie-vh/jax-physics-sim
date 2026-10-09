import simulator
import network
import training
import tests

from jax import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():   
    print(training.generate(10))
    tested_params = [-0.45, 0.1, 9.81, 1.0, 0.01, 100]
    #training_params = [[-0.5, 0.0, 9.81, 1.0, 0.01, 100], [-0.4, 0.1, 9.81, 1.0, 0.01, 100], [-0.5, 0.2, 9.81, 1.0, 0.01, 100], [-0.3, 0.5, 9.81, 1.0, 0.01, 100], [-0.4, 0.2, 9.81, 1.0, 0.01, 100]]
    training_params = training.generate(50)
    network_params = (2, 16, 2)
    wb_params = training.train(training_params=training_params, network_params=network_params, steps=100)

    tests.graph(tested_params=tested_params, wb_params=wb_params)
    tests.rmse_test(tested_params=tested_params, wb_params=wb_params)
    tests.energy_conservation_test(tested_params=tested_params)
    tests.symmetry_test(tested_params=tested_params)
    return 0

main()