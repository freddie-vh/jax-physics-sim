import simulator

import jax.numpy as jnp
import matplotlib.pyplot as plt

def graph(tested_params, wb_params):
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