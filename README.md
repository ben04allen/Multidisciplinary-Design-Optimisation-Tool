# Autonomous Aerodynamic Multidisciplinary Design Optimisation (MDO) Framework

## 📌 Overview
An automated, end-to-end Multidisciplinary Design Optimisation (MDO) pipeline developed for the LUMotorsport Formula Student team. This framework accelerates vehicle development by autonomously executing CFD simulations, building statistical surrogate models, and generating highly accurate aero-maps for the vehicle performance team.

## ⚙️ Pipeline Architecture

### I. Design Space Exploration (DoE)
* **Latin Hypercube Sampling (LHS):** Utilised to generate the initial space-filling training matrix. This ensures orthogonal, highly uniform coverage of the multidimensional design space (e.g., tracking front vs rear ride-height envelopes) while minimising the computational overhead of traditional full-factorial grids.

### II. Surrogate Modelling & Statistical Mathematics
* **Gaussian Process Regression (Kriging):** Acts as the core statistical surrogate model. It constructs a continuous, probabilistic response surface interpolating the sparse CFD data points to map key aerodynamic performance metrics (Total Downforce, $C_D A$, and Aero Balance).
* **Stochastic Estimation:** Predicts non-linear aerodynamic sensitivities across the ride height map, calculating both a predicted mean and a quantified standard deviation (variance) for every point in the unsimulated design space.

### III. Adaptive Infill Strategy (Active Learning)
* **Maximum Uncertainty Hunting:** An autonomous acquisition function dictates the next CFD run. The overarching Python algorithm queries the Gaussian Process to locate the exact design coordinates where the predictive variance (statistical uncertainty) is highest.
* **Targeted Resolution:** By actively hunting these statistical blind spots, the script autonomously launches targeted simulations to resolve highly non-linear aerodynamic phenomena—such as front wing ground-effect pinch or diffuser stall—using the absolute minimum number of compute-heavy RANS solver evaluations.

### IV. Autonomous CFD Execution Engine
* **Python-to-Java Orchestration:** A master Python sweep script handles subprocess execution of headless Simcenter STAR-CCM+ sessions.

## 🛠️ Tech Stack
* **Orchestration:** Python (SciPy, Scikit-Learn)
* **CFD Solver & Automation:** Simcenter STAR-CCM+ (Headless Execution, Java API)
* **Mathematics:** LHS, Gaussian Process Regression (Kriging), Active Learning Acquisition Functions
