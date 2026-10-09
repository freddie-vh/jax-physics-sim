import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax import value_and_grad
import jax

def init_params(layer_size, key): # Initialises network with random weights
    params = []
    for n_in, n_out in zip(layer_size[:-1], layer_size[1:]):
        key, w_key = random.split(key)
        w = random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append((w,b))
    return params

def forward(params, x): # Forward pass, calculates preactivation and passes it through a ReLU. Does this for each layer of the network and returns the result
    *hidden, last = params
    for w, b in hidden:
        x = jax.nn.relu(x @ w + b)
    w, b = last
    return x @ w + b

def loss(params, x, target): # Calculates mean squared error of a forward pass
    y = forward(params, x)
    return jnp.mean((y - target)**2)

@jax.jit
def optimise(x, target, params): # Trains the network on the provided data
    s = jax.tree.map(jnp.zeros_like, params) # Creates a pytree with the same structure as params, used to store a running average of past squared gradients for each weight/bias
    velocity = s # Used to store running average of gradients for each weight/bias
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    loss_and_grad = value_and_grad(loss) # Makes a function that evaluates loss and it's gradient
    def training_step(carry, _):
        params, s, velocity = carry
        loss_value, grads = loss_and_grad(params, x, target)
        s = jax.tree.map(lambda s_i, g: beta2 * s_i + (1-beta2) * g**2, s, grads) # Updates s, previous running average weighted more heavily than new squared gradient
        velocity = jax.tree.map(lambda v_i, g: beta1 * v_i + (1-beta1) * g, velocity, grads) # Updates velocity, similarly to s but with different beta
        params = jax.tree.map(lambda p, v, s_i: p - lr * v / (jnp.sqrt(s_i) + 1e-8), params, velocity, s) # Updates parameters, velocity used for direction, s to control step size (smaller for larger gradients)
        return (params, s, velocity), loss_value
    init_carry = (params, s, velocity)
    (params, s, velocity), losses =  lax.scan(training_step, init_carry, xs=None, length=500)
    return params