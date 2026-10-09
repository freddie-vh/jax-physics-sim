import network

import jax.numpy as jnp
import jax.lax as lax
import jax

@jax.jit(static_argnames=("steps", "unknown"))
def simulate(theta0, omega0, g, L, dt, steps, unknown):
    def forward_step(carry, _):
        theta, omega = carry
        angular_acceleration = -(g/L) * jnp.sin(theta)
        new_omega = omega + dt * angular_acceleration # Velocity updated first to stop energy from drifting over time
        if unknown == True:
            new_omega -= 0.01 * omega
        new_theta = theta + dt * new_omega
        new_state = jnp.array([new_theta, new_omega])
        return new_state, new_state
    final, result = lax.scan(forward_step, init=jnp.array([theta0, omega0]), xs=None, length=steps)
    initial = jnp.array([theta0, omega0])
    trajectories = jnp.concatenate([initial[None, :], result], axis=0)
    return trajectories

@jax.jit(static_argnames="steps")
def simulate_with_nn(theta0, omega0, g, L, dt, steps, params):
    def forward_step(carry, _):
        theta, omega = carry
        angular_acceleration = -(g/L) * jnp.sin(theta)
        new_omega = omega + dt * angular_acceleration 
        new_theta = theta + dt * new_omega
        new_state = jnp.array([new_theta, new_omega])
        difference = network.forward(params, new_state)
        new_state = new_state + dt * difference # Difference learned as a rate, so must be multiplied by dt to get difference for that timestep
        return new_state, new_state
    final, result = lax.scan(forward_step, init=jnp.array([theta0, omega0]), xs=None, length=steps)
    initial = jnp.array([theta0, omega0])
    return jnp.concatenate([initial[None, :], result], axis=0)