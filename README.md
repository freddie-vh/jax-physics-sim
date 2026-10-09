# Pendulum Simulation

Differentiable pendulum simulation made using JAX. Uses a semi-implicit Euler integrator to model known physics, with a neural network trained to correct for an unknown component that the integrator doesn't account for.

## Setup

```bash
pip install -r requirements.txt
```
## Usage

```bash
python main.py
```

Pick an option the menu and then enter parameters when prompted:
- Graph a simulation (angle vs angular velocity)
- Calculate RMSE with and without the network
- Plot RMSE against number of training datapoints

Parameter ranges: initial angle within ±π/2 rad, initial velocity within ±0.5, length 0.5 to 1.5. g is fixed at 9.81 and step size at 0.01.

## Files

- `main.py`: trains the network and starts the CLI
- `interface.py`: menu and input handling
- `sim_metrics.py`: graphing and error metrics
- `test_sim.py`: testing for simulator without the neural net
- `simulator.py`, `training.py`: simulation and network training
- `network.py`: optimiser and methods for the network
