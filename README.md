# Anomaly detection using the support vector data description (SVDD)

* run `identify_henry_coeffs.jl`
* run `SVDD_anomalyDet_fruitRipening.jl`

# anomaly_detection_gas_sensor_arrays

Anomaly detection gas sensor arrays is a project to showcase the utility of the support vector data description (SVDD) in an unsupervised setting by using the python package, scikit-learn, in the julia programming language to make an anomaly detector for a gas sensor array. Via simulation, we demonstrate with (1) a hypothetical sensor array composed of two gravimetric sensors utilizing zeolitic imidazolate frameworks and (2) a synthetic (gas composition, sensor array response) data set pertaining to gas sensing in a fruit ripening room, where (a) CO₂, C₂H₄, and H₂O vary in concentration and (b) anomalies are defined with respect to CO₂ and C₂H₄ concentrations.


## Usage

```julia
using CSV, DataFrames, ColorSchemes, Distributions, PlutoUI, ScikitLearn, Colors, Random, JLD, JLD2, LinearAlgebra, PyCall, Makie.GeometryBasics, CairoMakie

AnomalyDetection = include("src/AnomalyDetection.jl")
AnomalyDetectionPlots = include("src/AnomalyDetectionPlots.jl")
SyntheticDataGen = include("src/SyntheticDataGen.jl")

# create a synthetic data set
num_normal_train_points  = 100
num_anomaly_train_points = 0
num_normal_test_points   = 100
num_anomaly_test_points  = 5

σ_H₂O = 0.005
σ_m   = 0.00005

data_set = AnomalyDetection.setup_dataset(num_normal_train_points,
                                          num_anomaly_train_points,
                                          num_normal_test_points,
                                          num_anomaly_test_points,
                                          σ_H₂O, 
                                          σ_m)


# optimize hyperparameters
(ν_opt, γ_opt), _ = AnomalyDetection.bayes_validation(Data.X_train_scaled, n_iter=50, plot_data_flag=false)

# train anomaly detector
svm = AnomalyDetection.train_anomaly_detector(data_set.X_train_scaled, ν_opt, γ_opt)

# test performance
f1_score = AnomalyDetection.performance_metric(data_set.y_test, svm.predict(data_set.X_test_scaled))

print(f1_score)
```

