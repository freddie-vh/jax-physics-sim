import simulator

import pytest
import jax.numpy as jnp

@pytest.fixture(scope="module")
def test_energy_conservation(tested_params):
    sim_data = simulator.simulate(*tested_params, False)
    timestep = tested_params[4]
    theta = sim_data[:, 0]
    omega = sim_data[:, 1]
    modified_energy = 0.5 * ((omega[0] ** 2) * (theta ** 2) + (omega ** 2) - (omega[0] ** 2) * timestep * theta * omega) #modified energy equation, cross term to remove energy oscillation from integrator
    deviation = jnp.max(jnp.abs(modified_energy - modified_energy[0])) / jnp.abs(modified_energy[0])
    assert deviation < 2e-2, f"max relative deviation {deviation}"

def test_symmetry(tested_params):
    theta0, omega0, *rest = tested_params
    sim_data = simulator.simulate(*tested_params, False)
    sim_data_mirrored = simulator.simulate(-theta0, -omega0, *rest, False)
    assert jnp.allclose(sim_data, -sim_data_mirrored, atol=1e-6)