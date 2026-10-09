import simulator
import training

import jax.numpy as jnp
import matplotlib.pyplot as plt

def graph(tested_params, wb_params): # Plots angle against angular velocity
    sim_data = simulator.simulate(*tested_params, False)
    true_data = simulator.simulate(*tested_params, True)
    result = simulator.simulate_with_nn(*tested_params, wb_params)
    plt.scatter(true_data[:, 0], true_data[:, 1], label="True data")
    plt.scatter(sim_data[:, 0], sim_data[:, 1], label="Sim only")
    plt.scatter(result[:, 0], result[:, 1], label="Sim with neural net")
    plt.xlabel("Angle")
    plt.ylabel("Angular velocity")
    plt.legend()
    plt.show()

def rmse(tested_params, wb_params):
    sim_data = simulator.simulate(*tested_params, False)
    true_data = simulator.simulate(*tested_params, True)
    result = simulator.simulate_with_nn(*tested_params, wb_params)
    rmse = jnp.sqrt(jnp.mean((result - true_data)**2))
    sim_rmse = jnp.sqrt(jnp.mean((sim_data - true_data)**2))
    print("RMSE with network:", rmse)
    print("RMSE without network:", sim_rmse)
    print("Improvement:", (sim_rmse / rmse) * 100,"%")
    return rmse

def data_vs_accuracy(tested_params, wb_params=None):
    data_points_nums = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    rmse_values = []
    network_params = (2, 16, 2)
    for i in data_points_nums:
        training_params = training.generate(i)
        wb_params = training.train(training_params=training_params, network_params=network_params, steps=100)
        rmse_values.append(rmse(tested_params=tested_params, wb_params=wb_params))
    plt.scatter(data_points_nums, rmse_values)
    plt.xlabel("Number of training datapoints")
    plt.ylabel("Root mean squared error")
    plt.show()
