import training
import tests

import matplotlib
matplotlib.use('TkAgg')

def main():   
    tested_params = [-0.45, 0.1, 9.81, 1.0, 0.01, 100]
    training_params = training.generate(50)
    network_params = (2, 16, 2)
    wb_params = training.train(training_params=training_params, network_params=network_params, steps=100)

    tests.graph(tested_params=tested_params, wb_params=wb_params)
    tests.rmse_test(tested_params=tested_params, wb_params=wb_params)
    tests.energy_conservation_test(tested_params=tested_params)
    tests.symmetry_test(tested_params=tested_params)
    return 0

main()