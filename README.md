# Exact Verification of First-Order Methods via Mixed-Integer Linear Programming

This respository is by [Vinit Ranjan](https://vinitranjan1.github.io/), [Stefano Gualandi](https://mate.unipv.it/gualandi/), [Andrea Lodi](https://tech.cornell.edu/people/andrea-lodi/) and [Bartolomeo Stellato](https://stellato.io/) and contains the Python source code to reproduce experiments in our paper [Exact Verification of First-Order Methods via Mixed-Integer Linear Programming](https://arxiv.org/abs/2412.11330).

# Abstract

We present exact mixed-integer programming linear formulations for verifying the performance of first-order methods for parametric quadratic optimization. We formulate the verification problem as a mixed-integer linear program where the objective is to maximize the infinity norm of the fixed-point residual after a given number of iterations. Our approach captures a wide range of gradient, projection, proximal iterations through affine or piecewise affine constraints. We derive tight polyhedral convex hull formulations of the constraints representing the algorithm iterations. To improve the scalability, we develop a custom bound tightening technique combining interval propagation, operator theory, and optimization-based bound tightening. Numerical examples, including linear and quadratic programs from network optimization and sparse coding using Lasso, show that our method provides several orders of magnitude reductions in the worst-case fixed-point residuals, closely matching the true worst-case performance.

## Packages

The main required packages are
```
cvxpy >= 1.2.0
gurobipy
PEPit
hydra
```

### Running experiments
Experiments for the paper should be run from the `experiments/` folder with the command:

```
python run_experiment.py <experiment> local
```
where ```<experiment>``` is one of `LP`, `ISTA`, or `FISTA`.

All configuration details are contained in the `experiments/configs/` folder and the respective `.yaml` files can be modified for different experimental setups.

### Results
The `hydra` package will save the results in the respective subfolder under `experiments/<experiment>/outputs` folder with a timestamped directory.
The results include the worst-case residuals, the optimality gaps for the mixed integer solution, and the solve times.
