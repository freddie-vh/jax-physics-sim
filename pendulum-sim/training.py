import simulator
import network

from jax import random
import jax.numpy as jnp

def train(training_params, network_params, steps):
    key = random.PRNGKey(2)
    params = network.init_params(network_params, key)

    xs, targets = [], []
    for training_cond in training_params:
        traj_data = simulator.simulate(*training_cond, steps, True)
        theta, omega, g, L, dt = training_cond
        theta0 = traj_data[:-1, 0]
        omega0 = traj_data[:-1, 1]
        angular_acceleration = -(g/L) * jnp.sin(theta0)
        new_omega = omega0 + dt * angular_acceleration
        new_theta = theta0 + dt * new_omega
        new_state = jnp.stack([new_theta, new_omega], axis=1)
        xs.append(new_state)
        targets.append((traj_data[1:] - new_state) / dt)
    return network.optimise(jnp.concatenate(xs), jnp.concatenate(targets), params)

def generate(data_points_num, n=100): # Generates random training data using a uniform distribution, shuffles the data and returns required number of datapoints
    key = random.PRNGKey(1)
    key1, key2, key3 = random.split(key, 3)
    theta0 = jnp.linspace(-0.5, 0.5, n)
    omega0 = random.uniform(key1, n, minval=-0.5, maxval=0.5)
    g = jnp.full(n, 9.81)
    L = random.uniform(key2, n, minval=0.5, maxval=1.5)
    dt = jnp.full(n, 0.01)
    arrays = [theta0, omega0, g, L, dt]
    data_point_matrix = jnp.stack(arrays,axis=1)
    shuffled_data_points = random.permutation(key3, data_point_matrix)
    return shuffled_data_points[:data_points_num]