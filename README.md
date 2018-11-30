## Setup
1. Install Python 3.6 (anaconda)

`tensorflow` does not support 3.7

2. Install `qutip`

Refer to `qutip`'s documentation [here](http://qutip.org/docs/latest/installation.html).

In short, install anaconda (and create and activate a venv)
and run `conda install numpy scipy cython nose matplotlib
`

Then run `conda config --append channels conda-forge`
followed by `conda install qutip`

3. Install `tensorflow` (or `tensorflow-gpu`)

Run `conda install -c conda-forge tensorflow`, then `conda update --all`.
Alternatively, install through whatever method works according to the [docs](https://www.tensorflow.org/install/).
While the GPU-enabled version `tensorflow-gpu` runs faster, its installation is much more involved.

4. Install OpenAI gym envs

`pip install gym`

5. Install `dataclasses`

While part of Python 3.7+, it has been backported to 3.6 and can be installed with `pip install dataclasses`.


## Running

### Quantum simulation

`quantum_evolution/simulations/protocol_evaluator.py` has been set up to test the found protocols within the paper <sup id="a1">[1](#paper)</sup>.

Running it should plot the protocols as a graph of field over time, as well as print the output fidelities to console.
It should then display a Bloch figure animation of the protocol in action.


### Reinforcement learning

The examples in `reinforcement_learning/runners/examples/` are a good place to start. 
When run, the root directory needs to be added to `PYTHONPATH`. 
The easiest way to achieve this is to simply set up the project in PyCharm, then run the file within the project.


## Features

### Quantum simulation

`EnvSimulation` provides a standardised interface for solving a given hamiltonian with a certain field, making the assumption that the second term is time-dependent.

`BlochAnimator` creates an animation for a given set of `solve()` `Results`. 
It has additional parameter to plot static states, and methods to show or save the generated animation.

`BlochFigure` enables Bloch figure plotting by calling `update()` with states. 
This enables usage with `envs`' `render` methods (which call the `BlochFigure.update(states)` method with new states for that step).

`QEnv2` replicates an OpenAI gym env, with `render` and `step` and `reset` methods.
It returns `(h_x, t)` as state. Reward is 0 for all steps except the last, where it is `fidelity ** 2`. 

`QEnv3` is like `QEnv2`, but with `fidelity` as an additional value in state. State is thus `(h_x, t, fidelity)`.

`QEnv2SingleSolve` solves for final state with all actions for a given protocol only once. 
This means that it does not support `render` calls as the intermediate states are not calculated before all actions are taken.

`PresolvedQEnv2` is like `QEnv2SingleSolve`, except it also stores the calculated fidelities (for a given protocol) to a `dict` to avoid calling `solve` for repeated protocols.
This is not suitable for protocols with a large number of steps, but saves calculation on shorter protocols.

### Reinforcement learning

A high-level `run()` function is available, taking in `TRAINER`, `TRAINER_OPTIONS`, `MODEL`, `ENV` parameters (among others).
Most of these parameters have presets available (such as `TrainerPreset`, `ModelPreset`, and `EnvPreset`), which can be passed in place of the raw required information.

See the `reinforcement_learning/runners/examples/` for examples of usage. Most `[...]_trainer.py` files also contain runnable examples in their `if name == '__main__'` sections. 


## References

<b id="paper">1</b> Bukov M, Day AGR, Sels D, Weinberg P, Polkovnikov A, Mehta P. Reinforcement Learning in Different Phases of Quantum Control. _Physical Review X_. [Online] 2018;8(3). Available from: doi:10.1103/physrevx.8.031086  [↩](#a1)