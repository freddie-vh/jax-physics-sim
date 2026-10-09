import training
import simulator
import network

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

def rmse_test(tested_params, wb_params):
    sim_data = simulator.simulate(*tested_params, False)
    true_data = simulator.simulate(*tested_params, True)
    result = simulator.simulate_with_nn(*tested_params, wb_params)
    rmse = jnp.sqrt(jnp.mean((result - true_data)**2))
    sim_rmse = jnp.sqrt(jnp.mean((sim_data - true_data)**2))
    print("RMSE with network:", rmse)
    print("RMSE without network:", sim_rmse)
    print("Improvement:", (sim_rmse / rmse) * 100,"%")
    return rmse

def energy_conservation_test(tested_params):
    sim_data = simulator.simulate(*tested_params, False)
    timestep = tested_params[4]
    theta = sim_data[:, 0]
    omega = sim_data[:, 1]
    modified_energy = 0.5 * ((omega[0] ** 2) * (theta ** 2) + (omega ** 2) - (omega[0] ** 2) * timestep * theta * omega) #modified energy equation, cross term to remove energy oscillation from integrator
    modified_energy_change = modified_energy[-1] - modified_energy[0]
    mean_modified_energy = jnp.mean(modified_energy)
    std = jnp.std(modified_energy)
    print("Mean modified energy:", mean_modified_energy)
    print("Modified energy change:", modified_energy_change)
    print("Standard deviation of modified energy:", std)

def symmetry_test(tested_params):
    theta0, omega0, *rest = tested_params
    sim_data = simulator.simulate(*tested_params, False)
    sim_data_mirrored = simulator.simulate(-theta0, -omega0, *rest, False)
    dif = sim_data + sim_data_mirrored
    print("Mean difference between symmetrical simulations:", jnp.mean(dif))