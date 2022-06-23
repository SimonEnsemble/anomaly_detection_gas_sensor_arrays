module AnomalyDetection

using ScikitLearn, DataFrames, CairoMakie, ColorSchemes, LinearAlgebra, Statistics, Random, PyCall
SyntheticDataGen = include("SyntheticDataGen.jl")
skopt = pyimport("skopt")

@sk_import svm : OneClassSVM
@sk_import preprocessing : StandardScaler
@sk_import metrics : confusion_matrix
@sk_import metrics : f1_score

gases = ["C₂H₄", "CO₂", "H₂O"]
mofs = ["ZIF-71", "ZIF-8"]

anomaly_labels = ["CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"]

label_to_int = Dict(zip(anomaly_labels, [-1 for i = 1:length(anomaly_labels)]))
label_to_int["normal"]    = 1
label_to_int["anomalous"] = -1

mutable struct DataSet
	data_train::DataFrame
	
	X_train::Matrix{Float64}
	X_train_scaled::Matrix{Float64}
	y_train::Vector{Int64}

	data_test::DataFrame

	X_test::Matrix{Float64}
	X_test_scaled::Matrix{Float64}
	y_test::Vector{Int64}

	scaler
end

"""
generates a data set struct containing feature matrixes, target vectors, dataframes and scaler for 
injective or non-injective systems.
"""
function setup_dataset(num_normal_train_points,
				 	   num_anomaly_train_points,
					   num_normal_test_points,
					   num_anomaly_test_points,
	 				   σ_H₂O, 
					   σ_m;
					   system="non-injective")
	@assert system=="non-injective" || system=="injective"
	
	# generate synthetic training data
	if system=="non-injective"
		data_train = SyntheticDataGen.gen_data(num_normal_train_points, 
										num_anomaly_train_points, 
										σ_H₂O, 
										σ_m)
	elseif system=="injective"
		data_train = SyntheticDataGen.gen_data(num_normal_train_points, 
										num_anomaly_train_points, 
										σ_H₂O, 
										σ_m)

		data_train[:, "p H₂O [bar]"] .= 0
	else
		throw("training data injective vs non-injective system control flow error")
	end


	X_train, y_train = AnomalyDetection.data_to_Xy(data_train)
	scaler      	 = StandardScaler().fit(X_train)
	X_train_scaled   = scaler.transform(X_train)

	# generate synthetic test data
	if system=="non-injective"
		data_test = SyntheticDataGen.gen_data(num_normal_test_points, 
										num_anomaly_test_points, 
										σ_H₂O, 
										σ_m)
	elseif system=="injective"
		data_test = SyntheticDataGen.gen_data(num_normal_test_points, 
										num_anomaly_test_points, 
										σ_H₂O, 
										σ_m)

		data_test[:, "p H₂O [bar]"] .= 0
	else
		throw("training data injective vs non-injective system control flow error")
	end

	X_test, y_test = AnomalyDetection.data_to_Xy(data_test)
	X_test_scaled  = scaler.transform(X_test)

	# set up struct
	data_set = DataSet(data_train, 
					   X_train, 
					   X_train_scaled, 
					   y_train, 
					   data_test,
					   X_test, 
					   X_test_scaled,
					   y_test,
					   scaler)
	return data_set
end

"""
returns feature matrix and target vector from the synthetically generated data dataframe.
"""
function data_to_Xy(data::DataFrame)
	# X: (n_samples, n_features)
	X = Matrix(data[:, "m " .* mofs .* " [g/g]"])
	# y: anomolous or normal
	y = [label_to_int[label] for label in data[:, "label"]]
	return X, y
end



"""
****************************************************
optimize ν, γ via density measure.
****************************************************
"""

"""
returns the optimized ν and γ given k neighbors
"""
function opt_ν_γ_by_density_measure_method(X::Array{Float64}, K::Int)
	density_measures = compute_density_measures(X, K)

	elbow_id, density_measure_at_elbow = find_elbow(density_measures)

	nb_data = size(X)[1]
	ν_opt = (nb_data - elbow_id) / nb_data
	γ_opt = 1 / density_measure_at_elbow
	return ν_opt, γ_opt
end

"""
visualizes plot of ordered density measures and knee value
"""
function viz_density_measures(X::Matrix{Float64}, K::Int)
	density_measures = compute_density_measures(X, K)
	sorted_density_measures = sort(density_measures)

	elbow_id, density_measure_at_elbow = find_elbow(density_measures)

    fig = Figure()
    ax = Axis(fig[1, 1], ylabel="density measure", xlabel="(sorted) index")

	lines!(1:length(density_measures), sorted_density_measures)

	lines!([0, elbow_id, elbow_id],
		   [density_measure_at_elbow, density_measure_at_elbow, 0],
		   linestyle = :dash,
		   color = :red)

	text!("$(round(density_measure_at_elbow, digits=3))",
	      position = (0, density_measure_at_elbow))

    fig
end

"""
calculates the density measures for data set given K neighbors
"""
function compute_density_measures(X::Matrix{Float64}, K::Int)
    nb_data = size(X)[1]

    distance_matrix = [norm(X[i, :] - X[j, :]) for i = 1:nb_data, j = 1:nb_data]
    density_measures = zeros(nb_data)

    @assert K < nb_data

    for i = 1:nb_data
        density_measures[i] = mean(sort(distance_matrix[i, :])[2:K+1])
    end

    return density_measures
end

"""
find point of maximal curvature according to finite difference.
"""
function find_elbow(density_measures::Vector{Float64})
	# sorted density measure
	f = sort(density_measures)
	
	curvature = zeros(length(f))
	Δx = 1
	for i = 2:length(f) - 1
		f′  = (f[i + 1] - f[i - 1]) / 2
		f′′ = f[i + 1] - 2 * f[i] + f[i - 1]

		curvature[i] = f′′ / sqrt(1 + f′ ^ 2)
	end
	return argmax(curvature), f[argmax(curvature)]
end



"""
****************************************************
optimize ν, γ via synthetic anomaly hypersphere
****************************************************
"""

"""
returns optimal ν and γ using bayesian optimization (BayesSearchCV) method from skopt.
"""
function bayes_validation(X_train_scaled::Matrix{Float64}; 
						  n_iter::Int=10,
						  num_outliers::Int=500,
						  λ::Float64=0.5,
						  ν_space::Tuple{Float64, Float64}=(3/size(X_train_scaled, 1), 0.1),
						  γ_space::Tuple{Float64, Float64}=(1.0e-3, 1.0))

	R_sphere = maximum([norm(x) for x in eachrow(X_train_scaled)])
	X_sphere = AnomalyDetection.generate_uniform_vectors_in_hypersphere(num_outliers, R_sphere)

	#nested function used by BayesSearchCV as a scoring method, the return is maximized
	function bayes_objective_function(svm, X, _)
		# term 1: estimate of error on normal data as fraction support vectors
		fraction_svs = svm.n_support_[1] / size(X, 1)

		# term 2: estimating the volume inside the decision boundary
		y_sphere_pred   = svm.predict(X_sphere)
		num_inside 		= sum(y_sphere_pred .== 1)
		fraction_inside = num_inside / length(y_sphere_pred)

		#debuging code
		#=
		println("num sv's = $(svm.n_support_[1])")
		println("gamma = $(svm.gamma)")
		println("nu = $(svm.nu)")
		println("fraction sv's = $(fraction_svs)")
		println("fraction synthetic data inside = $(fraction_inside)")
		println("Size of X used in optimizer = $(size(X, 1))")
		=#

		#the objective function is maximized so,
		#in order to minimize the error function, make error negative.
		return -(λ * fraction_svs + (1 - λ) * fraction_inside)
	end

	params = Dict("nu" => ν_space, "gamma" => γ_space)

	opt = skopt.BayesSearchCV(
		OneClassSVM(), 
		params,
		n_iter=n_iter,
		scoring=bayes_objective_function,
		cv = [(collect(0:size(X_train_scaled, 1)-1), collect(0:size(X_train_scaled, 1)-1))]
		)

	#create a new y as the target vector
	#this isn't used but required by the BayesSearchCV algorithm.
	y′ = [NaN for i=1:size(X_train_scaled, 1)]

	#fit the optimizer using X and y'
	opt.fit(X_train_scaled, y′)

	#return opt.cv_results_
	return (opt.best_params_["nu"], opt.best_params_["gamma"]), X_sphere
end

"""
returns optimal ν and γ using exhaustive grid search and hypersphere of synthetic anomalies
λ weights the hyperparameters to favor false negatives or support vectors, default 0.5
"""
function determine_ν_opt_γ_opt_hypersphere_grid_search(X_train_scaled::Matrix{Float64};
	num_outliers::Int=500, λ::Float64=0.5, ν_range=0.1:0.1:0.3, γ_range=0.1:0.1:0.5)
	# generate data in hypersphere
	R_sphere = maximum([norm(x) for x in eachrow(X_train_scaled)])
	X_sphere = generate_uniform_vectors_in_hypersphere(num_outliers, R_sphere)

	opt_ν_γ = (0.0, 0.0)
	Λ_opt = Inf
	max_num_inside = 0

	for (i, ν) in enumerate(ν_range)
		for (j, γ) in enumerate(γ_range)
			if ν > 1.0
				ν = 1.0
			elseif ν <= 0.0
				ν = 10^(-5)
			end
			svm = AnomalyDetection.train_anomaly_detector(X_train_scaled, ν, γ)
			y_sphere   = svm.predict(X_sphere)
			num_inside = sum(y_sphere .== 1)
			if num_inside > max_num_inside
				max_num_inside = num_inside
			end
		end
	end

	for (i, ν) in enumerate(ν_range)
		for (j, γ) in enumerate(γ_range)
			if ν > 1.0
				ν = 1.0
			elseif ν <= 0.0
				ν = 10^(-5)
			end
			Λ = objective_function_scaled(X_train_scaled, X_sphere, ν, γ,max_num_inside, λ=λ)
			if Λ < Λ_opt
				Λ_opt = deepcopy(Λ)
				opt_ν_γ = (ν, γ)
			end	
		end
	end

	if opt_ν_γ[1] == ν_range[1]
		@warn "grid search optimized at lower ν boundary."
	elseif opt_ν_γ[1] == ν_range[end]
		@warn "grid search optimized at upper ν boundary."
	end
	if opt_ν_γ[2] == γ_range[1] 
		@warn "grid search optimized at lower γ boundary."
	elseif opt_ν_γ[2] == γ_range[end]
		@warn "grid search optimized at upper γ boundary."
	end

	return opt_ν_γ, X_sphere
end

"""
visualizes a heatmap of optimization values for ν and γ in the exhaustive grid search.
"""
function viz_νγ_opt_heatmap(σ_H₂O::Float64, 
	σ_m::Float64; 
	n_runs::Int=100, 
	λ=0.5,
	ν_range=0.01:0.03:0.30,
	γ_range=0.01:0.03:0.5)

	num_normal_test_points = num_normal_train_points = 100
	num_anomaly_train_points = 0
	num_anomaly_test_points = 5
	data = AnomalyDetection.setup_dataset(num_normal_train_points,
					num_anomaly_train_points,
					num_normal_test_points,
					num_anomaly_test_points,
					σ_H₂O, 
					σ_m)

	νγ_opt_grid = zeros(length(ν_range), length(γ_range))

	for i=1:n_runs
		(ν_opt, γ_opt), _ = AnomalyDetection.determine_ν_opt_γ_opt_hypersphere_grid_search(
		data.X_train_scaled,
		ν_range=ν_range, 
		γ_range=γ_range, 
		λ=λ)

		ν_index = findall(x->x==ν_opt,ν_range)[1]
		γ_index = findall(x->x==γ_opt,γ_range)[1]

		νγ_opt_grid[ν_index, γ_index] += 1
	end

	fig = Figure(resolution = (500, 600))

	ax = Axis(fig[1, 1],
			  xticks=(1:length(ν_range), ["$(AnomalyDetection.truncate(ν_range[i], 4))" for i=1:length(ν_range)]),
			  yticks=(1:length(γ_range), ["$(AnomalyDetection.truncate(γ_range[i], 4))" for i=1:length(γ_range)]),
			  xticklabelrotation=45.0,
			  ylabel="γ",
			  xlabel="ν",
			  title="σ_H₂O=$(σ_H₂O) σ_m=$(σ_m)"
	)

	hm = heatmap!(1:length(ν_range), 1:length(γ_range), νγ_opt_grid,
	colormap=ColorSchemes.dense, colorrange=(0.0, maximum(νγ_opt_grid)))
	Colorbar(fig[1, 2], hm, label="count")
	save("νγ_opt_heatmap_σ_H₂O=$(σ_H₂O)_σ_m=$(σ_m).pdf", fig)

	return fig
end

"""
visualizes the distribution of outliers in a hypersphere around our data
"""
function viz_synthetic_anomaly_hypersphere(X_sphere::Matrix{Float64}, X_scaled::Matrix{Float64})
	fig = Figure()
	ax = Axis(fig[1,1], aspect=DataAspect(), xlabel="m₁ scaled", ylabel="m₂ scaled")

	scatter!(X_sphere[:, 1], X_sphere[:, 2], markersize = 10, color=:red, marker=:x, label="synthetic data")
	scatter!(X_scaled[:, 1], X_scaled[:, 2], markersize = 5, color = :darkgreen, label="normal")
	xlims!(minimum(X_sphere[:, 1]) - 0.5, maximum(X_scaled[:, 1]) + 3)
	axislegend(position=:rb)
	return fig
end

"""
error function to be minimized based on λ
"""
function objective_function(X_train_scaled::Matrix{Float64}, 
						    X_sphere::Matrix{Float64}, 
							ν::Float64, 
							γ::Float64; 
							λ::Float64=0.5)
	svm = AnomalyDetection.train_anomaly_detector(X_train_scaled, ν, γ)

	# term 1: estimate of error on normal data as fraction support vectors
	fraction_svs = svm.n_support_[1] / size(X_train_scaled)[1]

	# term 2: estimating the volume inside the decision boundary
	y_sphere   		= svm.predict(X_sphere)
	num_inside 		= sum(y_sphere .== 1)
	fraction_inside = num_inside / length(y_sphere)

	return λ * fraction_svs + (1 - λ) * fraction_inside
end

"""
error function to be minimized based on λ, however instead of dividing by
the total number of synthetic data, divide by the maximum number of synthetic
data included in a test SVM during the grid search.
"""
function objective_function_scaled(X_train_scaled::Matrix{Float64}, 
								   X_sphere::Matrix{Float64}, 
								   ν::Float64, 
								   γ::Float64, 
								   max_inside_contour::Int; 
								   λ::Float64=0.5)
	svm = AnomalyDetection.train_anomaly_detector(X_train_scaled, ν, γ)

	# term 1: estimate of error on normal data as fraction support vectors
	fraction_svs = svm.n_support_[1] / size(X_train_scaled)[1]

	# term 2: estimating the volume inside the decision boundary
	y_sphere   	= svm.predict(X_sphere)
	num_inside 	= sum(y_sphere .== 1)

	#instead of dividing by the total in the sphere, divide by the maximum inside a trained svm
	fraction_inside = num_inside / max_inside_contour

	return λ * fraction_svs + (1 - λ) * fraction_inside
end

"""
generates a distribution of outliers in a hypersphere around our data
"""
function generate_uniform_vectors_in_hypersphere(num_outliers::Int64, R_sphere::Float64)
	X = zeros(num_outliers, 2)
	for i = 1:num_outliers
		X[i, :] = gen_uniform_vector_in_hypersphere()
	end
	return R_sphere * X
end

"""
generate a vector uniformly distributed in a hypersphere 
via generating a uniformly distributed vector in a cube and rejecting the corners.
"""
function gen_uniform_vector_in_hypersphere()
   x = randn(2)
	
   if norm(x) > 1
	   return gen_uniform_vector_in_hypersphere()
   else
	   return x
   end
end

"""
****************************************************
deployment of the One class support vector machine
****************************************************
"""


"""
trains a one class support vector machine
"""
function train_anomaly_detector(X_scaled::Matrix, ν::Float64, γ::Float64)
	oc_svm = OneClassSVM(kernel="rbf", nu=ν, gamma=γ)
	return oc_svm.fit(X_scaled)
end

"""
returns the f1 score, note the anomaly detector returns -1 for anomaly
and +1 for normal however the f1 score requires the opposite
"""
function performance_metric(y_true, y_pred)
    # anomalies (-1) are considered "positives" so need to switch sign.
    return f1_score(-y_true, -y_pred)
end

"""
visualizes the confusion matrix by anomaly type
"""
#TODO modularize into multiple functions compute cm, so that an axis can be input to be used in other places
function viz_cm(svm, data_test::DataFrame, scaler)
	all_labels = SyntheticDataGen.viable_labels
	n_labels = length(all_labels)

	cm = generate_cm(svm, data_test, scaler, all_labels)

	fig = Figure()
	ax = Axis(fig[1, 1],
		  xticks=([1, 2], ["anomalous", "normal"]),
		  yticks=([i for i=1:n_labels], all_labels),
		  xticklabelrotation=45.0,
		  ylabel="truth",
		  xlabel="prediction"
    )

	@assert SyntheticDataGen.viable_labels[1] == "normal"
	# anomalies
	heatmap!(1:2, 2:n_labels, cm[:, 2:end],
			      colormap=ColorSchemes.amp, colorrange=(0, maximum(cm[:, 2:end])))
	# normal data
	heatmap!(1:2, 1:1, reshape(cm[:, 1], (2, 1)),
			      colormap=ColorSchemes.algae, colorrange=(0, maximum(cm[:, 1])))
    for i = 1:2
        for j = 1:length(all_labels)
            text!("$(cm[i, j])",
                  position=(i, j), align=(:center, :center), 
                  color=cm[i, j] > sum(cm[:, j]) / 2 ? :white : :black)
        end
    end
    # Colorbar(fig[1, 2], hm, label="# data points")
    fig
end

function generate_cm(svm, data_test::DataFrame, scaler, labels)
	n_labels = length(labels)

	# confusion matrix. each row pertains to a label.
	# col 1 = -1 predicted anomaly, col 2 = 1 predicted normal.
	cm = zeros(Int, 2, n_labels)

	for (l, label) in enumerate(labels)
		# get all test data with this label
		data_test_l = filter(row -> row["label"] == label, data_test)
		# get feature matrix
		X_test_l, y_test_l = AnomalyDetection.data_to_Xy(data_test_l)
		# scale
		X_test_l_scaled = scaler.transform(X_test_l)
		# make predictions for this subset of test data
		y_pred_l = svm.predict(X_test_l_scaled)

		# how many are predicted as anomaly?
		cm[1, l] = sum(y_pred_l .== -1)
		# how many predicted as normal?
		cm[2, l] = sum(y_pred_l .== 1)
	end
	@assert sum(cm) == nrow(data_test)

	return cm
end

"""
visualizes a one class SVM decision contour given a particular nu, gamma and resolution.
"""
function viz_decision_boundary(svm, scaler, data_test::DataFrame; res::Int=700)
	X_test, _ = data_to_Xy(data_test)

	xlims = (0.98 * minimum(X_test[:, 1]), 1.02 * maximum(X_test[:, 1]))
	ylims = (0.98 * minimum(X_test[:, 2]), 1.02 * maximum(X_test[:, 2]))

	fig = Figure(resolution=(res, res))
	
	ax = Axis(fig[1, 1], 
			   xlabel = "m, " * mofs[1] * " [g/g]",
			   ylabel = "m, " * mofs[2] * " [g/g]",
			   aspect = DataAspect())

	viz_decision_boundary!(ax, svm, scaler, data_test, xlims, ylims)

	fig
end

function viz_decision_boundary!(ax, 
								svm, 
								scaler,
								data_test::DataFrame, 
								xlims, 
								ylims; 
								incl_legend::Bool=true)

	# generate the grid
	x₁s, x₂s, predictions = generate_response_grid(svm, scaler, xlims, ylims)

	viz_responses!(ax, data_test, incl_legend=incl_legend)
	contour!(x₁s, 
			 x₂s,
             predictions, 
		     levels=[0.0], 
			 color=:black,
		     label="decision boundary"
	) 
end

"""
generates a grid of anomaly scores based on a trained svm and given resolution
"""
function generate_response_grid(svm, scaler, xlims, ylims, res=100)
	# lay grid over feature space
	x₁s = range(xlims[1], xlims[2], length=res)
	x₂s = range(ylims[1], ylims[2], length=res)
	
	# get svm prediciton at each point
	predictions = zeros(res, res)
	for i = 1:res
		for j = 1:res
			x = [x₁s[i] x₂s[j]]
			x_scaled = scaler.transform(x)
			predictions[i, j] = svm.predict(x_scaled)[1]
		end
	end
	
	return x₁s, x₂s, predictions
end

"""
visualizes sensor response data by label
"""
function viz_responses!(ax, data::DataFrame; incl_legend::Bool=true)
	for data_l in groupby(data, "label")
		label = data_l[1, "label"]
		scatter!(ax, data_l[:, "m $(mofs[1]) [g/g]"], data_l[:, "m $(mofs[2]) [g/g]"],
				strokewidth=1,
				markersize=15,
				marker=label == "normal" ? :circle : :x,
				color=(:white, 0.0),
				strokecolor=SyntheticDataGen.label_to_color[label],
				label=label)
		
    end

	if incl_legend
    	axislegend(ax, position=:rb)
	end
end

"""
truncates floating digits for plots
"""
function truncate(n::Float64, digits::Int)
	n = n*(10^digits)
	n = trunc(Int, n)
	convert(AbstractFloat, n)
	return n/(10^digits)
end

"""
vizualizes effects of water variance and sensor error using validation:
method 1: hypersphere
method 2: knee
"""
function viz_sensorδ_waterσ_grid(σ_H₂Os::Vector{Float64}, 
								 σ_ms::Vector{Float64},
								 num_normal_train::Int64,
								 num_anomaly_train::Int64,
								 num_normal_test::Int64,
								 num_anomaly_test::Int64; 
								 validation_method::String="hypersphere")
	@assert validation_method=="hypersphere" || validation_method=="knee"

	#establish axes and figs for 9x9 grid
	fig  = Figure(resolution = (2400, 1200))
	axes = [Axis(fig[i, j]) for i in 1:3, j in 1:3]
	figs = [fig[i, j] for i in 1:3, j in 1:3]

	#top sensor error labels σ_m
	for (label, layout) in zip(["σₘ=$(σ_ms[1])","σₘ=$(σ_ms[2])", "σₘ=$(σ_ms[3])"], figs[1, 1:3])
    Label(layout[1, 1, Top()], 
		  label,
          textsize = 40,
          padding = (0, -385, 25, 0),
		  halign = :center)
	end

	#left water variance labels σ_H₂O
	for (label, layout) in zip(["σH₂O=$(σ_H₂Os[1])","σH₂O=$(σ_H₂Os[2])", "σH₂O=$(σ_H₂Os[3])"], figs[1:3, 1])
	Label(layout[1, 1, Left()], 
	  	  label,
		  textsize = 40,
	  	  padding = (0, 25, 0, 0),
	  	  valign = :center,
	  	  rotation = pi/2)
	end

		#create test data and find max/min for plots
		data_set_test = Dict()

		zif71_lims_high_σ = [Inf, 0]
		zif8_lims_high_σ  = [Inf, 0]

		zif71_lims_low_σ = [Inf, 0]
		zif8_lims_low_σ  = [Inf, 0]

		for (i, σ_H₂O) in enumerate(σ_H₂Os)
			for (j, σ_m) in enumerate(σ_ms)
				data_set_test[[i, j]] = SyntheticDataGen.gen_data(num_normal_test, num_anomaly_test, σ_H₂O, σ_m)
					#low variance
					if (i < 3) && (j < 3)
						zif71_lims_low_σ = [minimum([minimum(data_set_test[[i, j]][:, "m ZIF-71 [g/g]"]), 
													 zif71_lims_low_σ[1]]),
											maximum([maximum(data_set_test[[i, j]][:, "m ZIF-71 [g/g]"]), 
												     zif71_lims_low_σ[2]])]
						zif8_lims_low_σ  = [minimum([minimum(data_set_test[[i, j]][:, "m ZIF-8 [g/g]"]), 
													 zif8_lims_low_σ[1]]),
											maximum([maximum(data_set_test[[i, j]][:, "m ZIF-8 [g/g]"]), 
													 zif8_lims_low_σ[2]])]
					#high variance
					else
						zif71_lims_high_σ = [minimum([minimum(data_set_test[[i, j]][:, "m ZIF-71 [g/g]"]), 
													  zif71_lims_high_σ[1]]),
											 maximum([maximum(data_set_test[[i, j]][:, "m ZIF-71 [g/g]"]), 
													  zif71_lims_high_σ[2]])]
						zif8_lims_high_σ  = [minimum([minimum(data_set_test[[i, j]][:, "m ZIF-8 [g/g]"]), 
												      zif8_lims_high_σ[1]]),
											 maximum([maximum(data_set_test[[i, j]][:, "m ZIF-8 [g/g]"]), 
													  zif8_lims_high_σ[2]])]
					end
			end
		end

		zif71_lims_low_σ  = [0.98 * zif71_lims_low_σ[1], 1.02 * zif71_lims_low_σ[2]]
		zif8_lims_low_σ   = [0.98 * zif8_lims_low_σ[1], 1.02 * zif8_lims_low_σ[2]]
		zif71_lims_high_σ = [0.98 * zif71_lims_high_σ[1], 1.02 * zif71_lims_high_σ[2]]
		zif8_lims_high_σ  = [0.98 * zif8_lims_high_σ[1], 1.02 * zif8_lims_high_σ[2]]



	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)

			#generate test and training data, feature vectors, target vectors and standard scaler
			data = setup_dataset(num_normal_train, num_anomaly_train, num_normal_test, num_anomaly_test, σ_H₂O, σ_m)

			#optimize hyperparameters and determine f1score
			if validation_method == "hypersphere"
				(ν_opt, γ_opt), _ = bayes_validation(data.X_train_scaled)
			elseif validation_method == "knee"
				K            = trunc(Int, num_normal_train*0.05)
				ν_opt, γ_opt = opt_ν_γ_by_density_measure_method(data.X_train_scaled, K)
			end

			svm      = AnomalyDetection.train_anomaly_detector(data.X_train_scaled, ν_opt, γ_opt)
			y_pred 	 = svm.predict(data.X_test_scaled)
			f1_score = AnomalyDetection.performance_metric(data.y_test, y_pred)
			f1_score = truncate(f1_score, 2)

			#draw a background box colored according to f1score
			fig[i, j]        = Box(fig, color = (ColorSchemes.RdYlGn_4[f1_score], 0.7))
			axes[i, j].title = "f1 score = $(f1_score)"
			hidedecorations!(axes[i, j])

			#scatter and contour plot position LEFT
			pos = fig[i, j][1, 1] 
			
			#set x and y limits on contour plot
			#low variance data i&&j <= 2
			if i <=2 && j <= 2
				zif71_lims = zif71_lims_low_σ
				zif8_lims  = zif8_lims_low_σ
			#high variance data i||j > 2
			else
				zif71_lims = zif71_lims_high_σ
				zif8_lims  = zif8_lims_high_σ
			end
			
			ax = Axis(fig[i, j][1, 1], 
					  xlabel    = "m, " * mofs[1] * " [g/g]",
					  ylabel    = "m, " * mofs[2] * " [g/g]",
					  aspect    = 1,
					  xticks    = LinearTicks(3),
					  yticks    = LinearTicks(3),
					  alignmode = Outside(10))

	  
			viz_decision_boundary!(ax, 
								   svm, 
								   data.scaler, 
								   data_set_test[[i, j]], 
								   zif71_lims, 
								   zif8_lims, 
								   incl_legend=false)

			#confusion matrix position RIGHT
			pos = fig[i, j][1, 2] 

			all_labels = SyntheticDataGen.viable_labels
			n_labels   = length(all_labels)

			cm = generate_cm(svm, data_set_test[[i, j]], data.scaler, all_labels)

			ax = Axis(fig[i, j][1, 2],
			  	 	  xticks=([1, 2], ["anomalous", "normal"]),
			  		  yticks=([i for i=1:n_labels], all_labels),
			  		  xticklabelrotation=45.0,
			  		  ylabel="truth",
			  		  xlabel="prediction",
			  		  alignmode = Outside())

			@assert SyntheticDataGen.viable_labels[1] == "normal"

			# anomalies
			heatmap!(1:2, 
					 2:n_labels, 
					 cm[:, 2:end], 
					 colormap=ColorSchemes.amp, 
					 colorrange=(0, maximum(cm[:, 2:end])))
			
			# normal data
			heatmap!(1:2, 
					 1:1, 
					 reshape(cm[:, 1], (2, 1)), 
					 colormap=ColorSchemes.algae, 
					 colorrange=(0, maximum(cm[:, 1])))
			
			for i = 1:2
				for j = 1:length(all_labels)
					text!("$(cm[i, j])",
						  position=(i, j), 
						  align=(:center, :center), 
						  color=cm[i, j] > sum(cm[:, j]) / 2 ? :white : :black)
				end
			end
		end
	end

	if validation_method == "hypersphere"
		save("sensor_error_&_H2O_variance_plot_hypersphere.pdf", fig)
	elseif validation_method == "knee"
		save("sensor_error_&_H2O_variance_plot_knee.pdf", fig)
	end
	fig
end

"""
vizualizes a res x res plot of f1 scores as a heatmap of the two validation methods:
method 1: hypersphere
method 2: knee
"""
function viz_f1_score_heatmap(σ_H₂O_max::Float64, 
							  σ_m_max::Float64; 
							  res::Int=10, 
							  validation_method="knee",
							  hyperparameter_method="bayesian", 
							  n_avg::Int=10, 
							  λ=0.5)
	@assert validation_method=="hypersphere" || validation_method=="knee"
	@assert hyperparameter_method=="bayesian" || hyperparameter_method=="grid"
	
	#σ_H₂O_max = 0.1
	#σ_m_max = 0.001

	σ_H₂Os = 0:σ_H₂O_max/res:σ_H₂O_max
	σ_ms = 0:σ_m_max/res:σ_m_max

	num_normal_test_points = num_normal_train_points = 100
	num_anomaly_train_points = 0
	num_anomaly_test_points = 5

	f1_score_grid = zeros(res+1, res+1)
	

	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)
			f1_avg = 0.0
			
			for k = 1:n_avg
				data = AnomalyDetection.setup_dataset(num_normal_train_points,
										  num_anomaly_train_points,
										  num_normal_test_points,
										  num_anomaly_test_points,
								 		  σ_H₂O, 
										  σ_m)
	
				#optimize hyperparameters and determine f1score
				if validation_method == "hypersphere"
					if hyperparameter_method == "bayesian"
						(ν_opt, γ_opt), _ = bayes_validation(data.X_train_scaled)
					elseif hyperparameter_method == "grid"
						(ν_opt, γ_opt), _ = determine_ν_opt_γ_opt_hypersphere_grid_search(data.X_train_scaled)
					end
				elseif validation_method == "knee"
					K            = trunc(Int, num_normal_train_points*0.05)
					ν_opt, γ_opt = AnomalyDetection.opt_ν_γ_by_density_measure_method(data.X_train_scaled, K)
				end
	
				svm      = AnomalyDetection.train_anomaly_detector(data.X_train_scaled, ν_opt, γ_opt)
				y_pred 	 = svm.predict(data.X_test_scaled)
				f1_score = AnomalyDetection.performance_metric(data.y_test, y_pred)
	
				f1_avg += f1_score
			end
			
			f1_score_grid[i, j] = f1_avg/n_avg

		end
	end

	fig = Figure()
	
	ax = Axis(fig[1, 1],
		  xticks=(1:res+1, ["$(AnomalyDetection.truncate(i, 3))" for i=0:σ_H₂O_max/res:σ_H₂O_max]),
		  yticks=(1:res+1, ["$(AnomalyDetection.truncate(i, 5))" for i=0:σ_m_max/res:σ_m_max]),
		  xticklabelrotation=45.0,
		  ylabel="σ_m [g/g]",
		  xlabel="σ_H₂O [relative humidity]"
    )

	hm = heatmap!(1:res+1, 1:res+1, f1_score_grid,
			      colormap=ColorSchemes.RdYlGn_4, colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], hm, label="f1 score")

	if validation_method == "hypersphere"
		save("f1_score_plot_hypersphere.pdf", fig)
	elseif validation_method == "knee"
		save("f1_score_plot_knee.pdf", fig)
	end

	fig
end



"""
vizualizes the f1 scores of the range of potential λ values by averaging the number of runs
and performing a rolling average of every 3 points for smoothing then identifies the λ value that
corresponds to the highest f1 score.
"""
function lambda_plot(num_normal_train_points::Int,
					num_anomaly_train_points::Int,
					num_normal_test_points::Int,
					num_anomaly_test_points::Int;
					σ_H₂O::Float64=0.005, 
					σ_m::Float64=0.00005, 
					res::Int=50, 
					runs::Int=20)
	
	avg_f1_scores = zeros(res)
	λ_min 		  = 0
	λ_max 		  = 1.0
	λs 			  = [λ_min + ((λ_max-λ_min) * i-1) / (res) for i=1:res]
	
	for j=1:runs
		data_set = AnomalyDetection.setup_dataset(num_normal_train_points,
								num_anomaly_train_points,
								num_normal_test_points,
								num_anomaly_test_points,
								σ_H₂O, 
								σ_m)

		for (i, λ) in enumerate(λs)
			(ν_opt, γ_opt), _ = bayes_validation(data_set.X_train_scaled, λ=λ)
			svm = AnomalyDetection.train_anomaly_detector(data_set.X_train_scaled, ν_opt, γ_opt)
			f1_score = AnomalyDetection.performance_metric(data_set.y_test, svm.predict(data_set.X_test_scaled))
			avg_f1_scores[i] += f1_score
		end
	end

	#calculate rolling average of the average f1 scores and ID λ opt
	avg_f1_scores = avg_f1_scores./runs
	rolling_avg_f1 = [mean([avg_f1_scores[j] for j=(i-1):(i+1)]) for i=2:res-1]
	λ_opt = λs[argmax(rolling_avg_f1)+1]
	λ_opt = AnomalyDetection.truncate(λ_opt, 2)

	
	#Plot
	fig = Figure()
	ax = Axis(fig[1, 1], ylabel="f1 score", xlabel="λ", xticks=λ_min:0.1:(λ_max+0.1), title="σ_H₂O=$σ_H₂O, σ_m=$σ_m") 
	lines!([λs[i] for i=2:res-1], rolling_avg_f1, label="avg f1 score")
	lines!([λ_opt, λ_opt], [minimum(rolling_avg_f1), maximum(rolling_avg_f1)], linestyle=:dash, label="λ opt=$λ_opt")
	axislegend(ax, position=:rb)
	save("λ_opt_plot_σ_H₂O=$(σ_H₂O)_σ_m=$(σ_m).pdf", fig)
	fig
end

end
