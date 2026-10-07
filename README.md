# Autonomous Aerodynamic Multidisciplinary Design Optimisation (MDO) Framework

## Overview
An automated Multidisciplinary Design Optimisation (MDO) tool developed for the LUMotorsport Formula Student team. This framework optimises vehicle development by autonomously updating CAD models, executing CFD simulations, building statistical surrogate models and generating highly accurate aero-maps for the vehicle performance team.

## Architecture

### I. CAD and CFD Model Adaptation
* **Parametric Modelling:** If the user uploads parametric CAD, the tool uses a Java macro to read and store the values of select parameters and can adjust them in line with the DoE throughout the run.
* **CFD Implementation:** These new models are automatically imported into a fully automated CFD sim, tagged and renamed. Any predefined parameters in the sim may also be adjusted for design optimisation or aero-map generation.

### II. Design Space Exploration (DoE)
* **Latin Hypercube Sampling (LHS):** Utilised to generate the initial space-filling training matrix. This ensures uniform coverage of the multidimensional design space (e.g., tracking front vs rear ride-height envelopes) while minimising computational cost.

### III. Surrogate Modelling & Statistical Mathematics
* **Gaussian Process Regression (Kriging):** Acts as the core statistical surrogate model. It constructs a continuous, probabilistic response surface interpolating the CFD data points to map aerodynamic performance metrics (Total Downforce, $C_D A$, Aero Balance etc.).
* **Stochastic Estimation:** Predicts non-linear aerodynamic sensitivities across the ride height map, calculating both a predicted mean and a quantified standard deviation (variance) for every point in the unsimulated design space.

### IV. Adaptive Infill Strategy (Active Learning)
* **Maximum Uncertainty Hunting:** An autonomous acquisition function dictates the next CFD run. The overarching Python algorithm queries the Gaussian Process to locate the exact design coordinates where the predictive variance (statistical uncertainty) is highest.
* **Targeted Resolution:** By actively hunting these statistical blind spots, the script launches targeted simulations to resolve highly non-linear aerodynamic phenomena—such as front wing ground-effect pinch or diffuser stall—using the minimum number of CFD runs.

### V. Autonomous CFD Execution Engine
* **Python-to-Java Orchestration:** A master Python sweep script handles subprocess execution of headless Simcenter STAR-CCM+ sessions.

## Software
* **Development:** Python (SciPy, Scikit-Learn), Java
* **CFD Solver & Automation:** Simcenter STAR-CCM+ (Headless Execution, Java API)
* **Mathematics:** LHS, Gaussian Process Regression (Kriging), Active Learning Acquisition Functions
