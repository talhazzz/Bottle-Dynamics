# Bottle-Dynamics

A custom, physics-based reinforcement learning environment for bottle-flip control.

## Overview

Bottle-Dynamics explores how reinforcement learning agents can learn to execute bottle flips and achieve stable, upright landings through trial and error.

Using a 2D rigid-body physics simulation integrated with Gymnasium, the environment allows agents to learn launch strategies by controlling the forces and rotational impulses applied to a bottle.

The project investigates how different water-fill levels affect bottle dynamics and flipping performance, with experiments at **20%, 50%, and 70% water fill**.

Trained agents will be evaluated across repeated trials, with model checkpoints saved at regular training intervals to identify the most reliable policies.

A visualization interface will allow users to observe bottle trajectories, evaluate checkpoints, and compare reinforcement learning algorithms.

## Objectives

- Develop a custom 2D physics simulation for bottle-flip dynamics.
- Implement a Gymnasium environment with a continuous action space.
- Train reinforcement learning agents to achieve stable, upright landings.
- Investigate the effects of 20%, 50%, and 70% water-fill levels.
- Save and evaluate model checkpoints at configurable training intervals.
- Compare algorithms using landing success rates, accuracy, and stability.
- Build an interactive interface for visualizing flips and comparing results.

## Technology Stack

- Python — Core implementation
- Gymnasium — Reinforcement learning environment
- Pymunk — 2D rigid-body physics
- Pygame — Visualization
- Stable-Baselines3 — Reinforcement learning algorithms
- TensorBoard — Training metrics and experiment monitoring

## Evaluation

Models will be evaluated across repeated episodes using metrics such as landing success rate, landing-angle error, residual angular velocity, and landing stability.

Checkpoint comparisons will help identify reliable policies rather than automatically selecting the final training model.

## Project Status

Under development.
