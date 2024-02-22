# anomaly_detection_gas_sensor_arrays

Anomaly detection gas sensor arrays is a project to showcase the utility of the support vector data description (SVDD) in an unsupervised setting by using the python package, scikit-learn, in the julia programming language to make an anomaly detector for a gas sensor array. Via simulation, we demonstrate with (1) a hypothetical sensor array composed of two gravimetric sensors utilizing zeolitic imidazolate frameworks and (2) a synthetic (gas composition, sensor array response) data set pertaining to gas sensing in a fruit ripening room, where (a) CO₂, C₂H₄, and H₂O vary in concentration and (b) anomalies are defined with respect to CO₂ and C₂H₄ concentrations.
We also showcase a much simpler alternative anomaly detection algorithm, the elliptic envelope. This algorithm assumes the underlying distribution of observations are Gaussian, therby using the subset of data that provides the minimum determinant to draw a tight ellipse around our data.

# Anomaly detection using the support vector data description (SVDD) and the elliptic envelope

## Using Pluto.jl
* run `identify_henry_coeffs.jl`
    * this Pluto notebook exports the Henry coefficients jld file necessary for generating synthetic data.
* run `SVDD_anomalyDet_fruitRipening.jl`
    * this Pluto notebook uses data in the example folder to generate and evaluate the performance of the support vector data description (SVDD) anomaly detector in a variety of background interferent (H₂O) variance and sensor error settings.
* run `Eenvelope_anomalyDet_fruitRipening.jl`
    * this Pluto notebook uses data in the example folder to generate and evaluate an alternative anomaly detector, the elliptic envelope.

 ### src folder
 This folder contains three files, AnomalyDetection.jl, AnomalyDetectionPlots.jl and SyntheticDataGen.jl. 
 * AnomalyDetection.jl
   This file contains functions necessary to train, deploy, optimize and evaluate the SVDD and elliptic envelope algorithms.
 * AnomalyDetectionPlots.jl
   This file contains any plotting functions related to sensor readouts, deployment, optimization and evaluation of the SVDD and elliptic envelope algorithms.
 * SyntheticDataGen.jl
   This file contains any functions associated with gas compositions of our model prior to realization of the sensor response.

 ### data folder
 * This folder contains equilibrium adsorption isotherm csv files used to extract the Henry coefficients for zeolitic imidizolate frameworks ZIF-8 and ZIF-71.
   
 ### example folder
 * This folder contains JLD files used for quick visualizations of simulated response vectors and their associated statistics already computed.


## Usage example

```julia
using CSV, DataFrames, ColorSchemes, Distributions, PlutoUI, ScikitLearn, Colors, Random, JLD, JLD2, LinearAlgebra, PyCall, Makie.GeometryBasics, CairoMakie

AnomalyDetection = include("src/AnomalyDetection.jl")
AnomalyDetectionPlots = include("src/AnomalyDetectionPlots.jl")
SyntheticDataGen = include("src/SyntheticDataGen.jl")

# size of data in the synthetic data set.
num_normal_train_points  = 100
num_anomaly_train_points = 0
num_normal_test_points   = 100
num_anomaly_test_points  = 5

# set the variance of the distribution of water vapor concentration [RH].
σ_H₂O = 0.005

# set the error of the sensor.
σ_m   = 0.00005

# generate the synthetic data set.
data_set = AnomalyDetection.setup_dataset(num_normal_train_points,
                                          num_anomaly_train_points,
                                          num_normal_test_points,
                                          num_anomaly_test_points,
                                          σ_H₂O, 
                                          σ_m)


# optimize hyperparameters.
(ν_opt, γ_opt), _ = AnomalyDetection.bayes_validation(Data.X_train_scaled, n_iter=50, plot_data_flag=false)

# train the anomaly detector.
svm = AnomalyDetection.train_anomaly_detector(data_set.X_train_scaled, ν_opt, γ_opt)

# test performance.
f1_score = AnomalyDetection.performance_metric(data_set.y_test, svm.predict(data_set.X_test_scaled))
```

