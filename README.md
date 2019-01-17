# BezierTrajectory

The purpose of this repository is to build a library that can be used to generate optimal trajectories for multiple vehicles using Bezier curves.

This project has three main parts: Bezier curve library, optimal generation, and displaying the results.

## Bezier Curve Library

Bezier curves are parametric curves which use Bernstein polynomials as a basis. They have many useful properties which can be found at this website (https://pomax.github.io/bezierinfo/) and this paper (Farouki, The Bernstein polynomial basis: a centennial retrospective). The Bezier curve library is used to leverage the power of these curves by performing actions such as taking the derivative, increasing the order, and finding the norm squared of curves.

## Optimal Generation

By applying a desired cost function and constraints, the optimal trajectories for vehicles can be computed. The optimizer being used is the SciPy function minimize. This function is very similar to MATLAB's fmincon. The specific algorithm being used is "SLSQP".

## Displaying Results

Once the optimal trajectories have been generated, the results are displayed to the user. This is important when debugging trajectories and further improving the code. When used on a robot, it is advised that the engineer supress any plots being produced.
