import simulator
import network

from jax import random
import jax.numpy as jnp

def train(training_params, network_params, steps):
    key = random.PRNGKey(2)
    params = network.init_params(network_params, key)
    sim_data = jnp.array([])
    difference = jnp.array([])
    sim_data = jnp.stack([simulator.simulate(*training_cond, steps, False) for training_cond in training_params]).reshape(-1, 2)
    true_data = jnp.stack([simulator.simulate(*training_cond, steps, True) for training_cond in training_params]).reshape(-1, 2)
    difference = true_data - sim_data

    params = network.optimise(sim_data, difference, params)
    return params

def generate(data_points_num, n=100): # Generates random training data using a normal distribution, shuffles the data and returns required number of datapoints
    key = random.PRNGKey(1)
    key1, key2, key3 = random.split(key, 3)
    theta0 = jnp.linspace(-0.5, 0.5, n)
    omega0 = 0.2 * random.normal(key1, n)
    g = jnp.full(n, 9.81)
    L = 1 + 0.1 * random.normal(key2, n)
    dt = jnp.full(n, 0.01)
    arrays = [theta0, omega0, g, L, dt]
    data_point_matrix = jnp.stack(arrays,axis=1)
    shuffled_data_points = random.permutation(key3, data_point_matrix)
    return shuffled_data_points[:data_points_num]