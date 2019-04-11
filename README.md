# BezierTrajectory

The purpose of this repository is to build a library that can be used to generate optimal trajectories, states and control inputs for autonomous vehicles using Bernstein polynomials.

This project has three main parts: Bezier curve library, Optimal control solver, and results plotting.

## Bezier Curve Library

Bezier curves are parametric curves which use Bernstein polynomials as a polynomial basis. They possess many useful properties which can exploited for motion planning [1]. The Bezier Curve Library implements such properties together with computationally efficient algorithms to compute the max/min of a Bezier curve, the distance between Bezier curves, etc.

## Optimal Control solver

By applying a desired cost function and constraints, the optimal trajectories for vehicles can be computed. The optimizer being used is the SciPy function minimize. The specific algorithm being used is "SLSQP".

## Displaying Results

Once the optimal trajectories have been generated, the results are displayed to the user. This is important when debugging trajectories and further improving the code. When used on a robot, it is advised that the engineer supress any plots being produced.

## Important Notes

Due to a bug in previous version of the Scipy package, it is important that you use version 1.2.0 or higher. In earlier versions, the minimizing routine would return successfully even if the constraints to the problem were not met.

## Acknowledgement

This research is funded by ONR.


[1] Farouki, Rida T. "The Bernstein polynomial basis: A centennial retrospective." Computer Aided Geometric Design 29.6 (2012): 379-419.
