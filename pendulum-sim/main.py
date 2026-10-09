import training
import test_sim
import sim_metrics

import matplotlib
matplotlib.use('TkAgg')

def main():   
    tested_params = [-0.45, 0.1, 9.81, 1.0, 0.01, 100]
    training_params = training.generate(50)
    network_params = (2, 16, 2)
    wb_params = training.train(training_params=training_params, network_params=network_params, steps=100)

    sim_metrics.graph(tested_params=tested_params, wb_params=wb_params)
    sim_metrics.rmse(tested_params=tested_params, wb_params=wb_params)
    test_sim.test_symmetry(tested_params=tested_params)
    return 0

main()