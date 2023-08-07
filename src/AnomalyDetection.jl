module AnomalyDetection

using ScikitLearn, DataFrames, CairoMakie, ColorSchemes, LinearAlgebra, Statistics, Random, PyCall, JLD2
SyntheticDataGen = include("SyntheticDataGen.jl")
skopt = pyimport("skopt")
#
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
					   system="non-injective",
					   seed=abs(rand(Int)))
	@assert system=="non-injective" || system=="injective"
	Random.seed!(seed)
	
	# generate synthetic training data
	data_train = SyntheticDataGen.gen_data(num_normal_train_points, 
										num_anomaly_train_points, 
										σ_H₂O, 
										σ_m)
	if system=="injective"
		data_train[:, "p H₂O [bar]"] .= 0
	end


	X_train, y_train = AnomalyDetection.data_to_Xy(data_train)
	scaler      	 = StandardScaler().fit(X_train)
	X_train_scaled   = scaler.transform(X_train)

	# generate synthetic test data
	data_test = SyntheticDataGen.gen_data(num_normal_test_points, 
										num_anomaly_test_points, 
										σ_H₂O, 
										σ_m)
	if system=="injective"
		data_test[:, "p H₂O [bar]"] .= 0
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
						  n_iter::Int=35,
						  num_outliers::Int=5000,
						  λ::Float64=0.5,
						  ν_space::Tuple{Float64, Float64}=(1/size(X_train_scaled, 1), 0.3),
						  γ_space::Tuple{Float64, Float64}=(1.0e-3, 1.0),
						  plot_data_flag::Bool=false)

	R_sphere = maximum([norm(x) for x in eachrow(X_train_scaled)])
	X_sphere = AnomalyDetection.generate_uniform_vectors_in_hypersphere(num_outliers, R_sphere)
	plot_data::Vector{Tuple{Float64, Float64, Float64}} = []

	#nested function used by BayesSearchCV as a scoring method, the return is maximized
	function bayes_objective_function(svm, X, _)
		# term 1: estimate of error on normal data as fraction support vectors
		fraction_svs = svm.n_support_[1] / size(X, 1)

		# term 2: estimating the volume inside the decision boundary
		y_sphere_pred   = svm.predict(X_sphere)
		num_inside 		= sum(y_sphere_pred .== 1)
		fraction_inside = num_inside / length(y_sphere_pred)

		#error function
		Λ = λ * fraction_svs + (1 - λ) * fraction_inside

		#push data needed for plot
		push!(plot_data, (svm.nu, svm.gamma, Λ))

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
		return -Λ
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

	#return opt.cv_results_, include plotting data if flagged
	if plot_data_flag
		return (opt.best_params_["nu"], opt.best_params_["gamma"]), X_sphere, plot_data
	end

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
function generate_uniform_vectors_in_hypersphere(num_outliers::Int64, 
												R_sphere::Float64)
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
****************************************************
learning curve
****************************************************
"""

"""
returns a matrix of f1 scores for a given vector of training set sizes.
num_iter is the number rows in the matrix.
"""
function learning_curve(num_normal_train_points::Vector{Int64};
						num_normal_test::Int64=100,
						num_anomaly_test::Int64=5,
						gen_data_flag::Bool=true,
						num_iter::Int64=5,
						σ_H₂O::Float64=0.01,
						σ_m::Float64=1.0e-5)

	data_storage = zeros(num_iter, length(num_normal_train_points))
	num_anomaly_train = 0 #this should always stay 0

	for (i, num_normal_train) in enumerate(num_normal_train_points)
		for j=1:num_iter
			#GENERATE DATASET 
			Data = AnomalyDetection.setup_dataset(num_normal_train, 
											num_anomaly_train, 
											num_normal_test, 
											num_anomaly_test, 
											σ_H₂O, 
											σ_m)

			#STEP 1 - VALIDATE
			(ν_opt, γ_opt), _ = AnomalyDetection.bayes_validation(Data.X_train_scaled, n_iter=50, plot_data_flag=false)

			#STEP 2 - TRAIN ANOMALY DETECTOR
			svm = AnomalyDetection.train_anomaly_detector(Data.X_train_scaled, ν_opt, γ_opt)

			#STEP 3 - DETERMINE F1-SCORE
			data_storage[j, i] = AnomalyDetection.performance_metric(Data.y_test,svm.predict(Data.X_test_scaled))
		end
	end
	return data_storage
end

"""
runs the learning curve function iteratively and saves matrices in JLD files.
JLD files are stored in the jld parent folder.
"""
function simulate(num_normal_train::Vector{Int64}; 
				run_start::Int64=1, 
				run_end::Int64=20, 
				iter_per_run::Int64=5,
				σ_H₂O::Float64=0.01,
				σ_m::Float64=1.0e-5)
	for i=run_start:run_end
		simulation = learning_curve(num_normal_train, num_iter=iter_per_run, σ_H₂O=σ_H₂O, σ_m=σ_m)
		JLD2.jldsave("jld/learning_curve_sim_$(i)"*".jld2", i=simulation)  
	end
end

"""
gathers jld files from learning curve simulation and catenates them into a single matrix.
"""
function catenate_data(;folder::String="jld/", rows_per_file::Int=5)

	# gather all the files
	files = [file for file in readdir(folder) if isfile(abspath(joinpath(folder, file)))]

	# make sure directory isn't empty
	@assert length(files) > 0 "no files found"

	num_data = rows_per_file * length(files)
	columns = size(JLD2.load_object(joinpath(folder, files[1])), 2)

	# make empty matrix to hold data
	data = zeros(num_data, columns)

	for (i, file) in enumerate(files)
		file_data = JLD2.load_object(joinpath(folder, file))
		for j=1:rows_per_file
			row = (i-1) * rows_per_file + j
			data[row, :] = file_data[j, :]
		end
	end

	return data
end

"""
gathers a vector of (mean, standard error) for each training set size from the learning curve simulation matrix.
"""
function score_stats(data::Matrix{Float64})

	data_storage = zeros(size(data, 2))
	data_storage = convert(Array{Any}, data_storage)

	for (i, col) in enumerate(eachcol(data))
		#calc mean f1
		f1_scores_mean = mean(col)
	
		#calc standard error
		f1_scores_se = std(col) / sqrt(size(data, 1))
	
		#store f1 for particular sized set of training data
		data_storage[i] = (f1_scores_mean, f1_scores_se)
	end

	return data_storage
end

end
