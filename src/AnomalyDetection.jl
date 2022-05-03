module AnomalyDetection

using ScikitLearn, DataFrames, CairoMakie, ColorSchemes, LinearAlgebra, Statistics
SyntheticDataGen = include("SyntheticDataGen.jl")

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
returns optimal ν and γ using grid search and hypersphere of synthetic anomalies
λ weights the hyperparameters to favor false negatives or support vectors, default 0.5
"""
function determine_ν_opt_γ_opt_hypersphere(X_train_scaled::Matrix{Float64};
	num_outliers::Int=1000, λ::Float64=0.5)
	# generate data in hypersphere
	R_sphere = maximum([norm(x) for x in eachrow(X_train_scaled)])
	X_sphere = generate_uniform_vectors_in_hypersphere(num_outliers, R_sphere)

	# grid search for optimal ν, γ
	ν_range = 0.01:0.01:0.26
	γ_range = 0.05:0.01:0.7

	opt_ν_γ = (0.0, 0.0)
	Λ_opt = Inf

	for (i, ν) in enumerate(ν_range)
		for (j, γ) in enumerate(γ_range)
			Λ = objective_function(X_train_scaled, X_sphere, ν, γ, λ)
			if Λ < Λ_opt
				Λ_opt = deepcopy(Λ)
				opt_ν_γ = (ν, γ)
			end	
		end
	end

	if opt_ν_γ[1] == ν_range[1] || opt_ν_γ[1] == ν_range[end]
		@warn "grid search optimized at boundary... change ν range."
	end
	if opt_ν_γ[2] == γ_range[1] || opt_ν_γ[2] == γ_range[end]
		@warn "grid search optimized at boundary... change γ range."
	end
	return opt_ν_γ
end

"""
visualizes the distribution of outliers in a hypersphere around our data
"""
function viz_synthetic_anomaly_hypersphere(X_sphere::Matrix{Float64}, X_scaled::Matrix{Float64})
	fig = Figure()
	ax = Axis(fig[1,1], aspect=DataAspect(), xlabel="m₁ scaled", ylabel="m₂ scaled")

	scatter!(X_sphere[:, 1], X_sphere[:, 2], markersize = 10, color=:red, marker=:x, label="synthetic data")
	scatter!(X_scaled[:, 1], X_scaled[:, 2],
	markersize = 5, color = :darkgreen, label="normal")
	xlims!(minimum(X_scaled[:, 1]) - 1, maximum(X_scaled[:, 1]) + 3)
	axislegend(position=:rb)
	return fig
end

"""
error function to be minimized based on λ
"""
function objective_function(X_train_scaled::Matrix{Float64}, X_sphere::Matrix{Float64}, ν::Float64, γ::Float64, λ::Float64=0.5)
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
visualizes the confursion matrix by anomaly type
"""
function viz_cm(svm, data_test::DataFrame, scaler)
	all_labels = SyntheticDataGen.viable_labels
	n_labels = length(all_labels)

	# confusion matrix. each row pertains to a label.
	# col 1 = -1 predicted anomaly, col 2 = 1 predicted normal.
	cm = zeros(Int, 2, n_labels)

	for (l, label) in enumerate(all_labels)
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
	heatmap!(1:2, 2:6, cm[:, 2:end],
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

"""
visualizes a one class SVM decision contour given a particular nu, gamma and resolution.
"""
function viz_decision_boundary(svm, scaler, data_test::DataFrame, res::Int=700, incl_low_humidity::Bool=false)
	if incl_low_humidity
    	X_test, _ = data_to_Xy(data_test)
	else
		data_test = deepcopy(filter(row -> row[:label] != "low humidity", data_test))
		X_test, _ = data_to_Xy(data_test)
	end

	xlims = (0.98 * minimum(X_test[:, 1]), 1.02 * maximum(X_test[:, 1]))
	ylims = (0.98 * minimum(X_test[:, 2]), 1.02 * maximum(X_test[:, 2]))

	# generate the grid
	x₁s, x₂s, predictions = generate_response_grid(svm, scaler, xlims, ylims)
	fig = Figure(resolution=(res, res))
	
	
	ax = Axis(fig[1, 1], 
			   xlabel = "m, " * mofs[1] * " [g/g]",
			   ylabel = "m, " * mofs[2] * " [g/g]",
			   aspect = DataAspect())

	viz_responses!(ax, data_test)
	contour!(x₁s, 
			 x₂s,
             predictions, 
		     levels=[0.0], 
			 color=:black,
		     label="decision boundary"
	) 
	
	fig
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
function viz_responses!(ax, data::DataFrame)
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

    axislegend(ax, position=:rb)
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
method 1: uniform hypersphere
method 2: knee
"""
function viz_sensorδ_waterσ_grid(σ_H₂O::Vector{Float64}, σ_m::Vector{Float64}, method::Int=1)
	@assert method==1 || method==2

	fig 					= Figure(resolution = (2400, 1200))
	ideal_fig 				= fig[1, 1]
	med_sensor_error_fig 	= fig[1, 2]
	high_sensor_error_fig 	= fig[1, 3]
	med_water_variance_fig 	= fig[2, 1]
	high_water_variance_fig = fig[3, 1]

	#set contour plot boundaries for high variance data
	temp 	   		  = SyntheticDataGen.gen_data(200, 0, σ_H₂O[3], σ_m[3])
	zif71_lims_high_σ = (0.98 * minimum(temp[:, "m ZIF-71 [g/g]"]), 
				  		 1.02 * maximum(temp[:, "m ZIF-71 [g/g]"]))
	zif8_lims_high_σ  = (0.98 * minimum(temp[:, "m ZIF-8 [g/g]"]), 
				  		 1.02 * maximum(temp[:, "m ZIF-8 [g/g]"]))

	#set contour plot boundaries for low variance data
	temp2 	   		 = SyntheticDataGen.gen_data(200, 0, σ_H₂O[2], σ_m[2])
	zif71_lims_low_σ = (0.98 * minimum(temp2[:, "m ZIF-71 [g/g]"]), 
				  		1.02 * maximum(temp2[:, "m ZIF-71 [g/g]"]))
	zif8_lims_low_σ  = (0.98 * minimum(temp2[:, "m ZIF-8 [g/g]"]), 
				  		1.02 * maximum(temp2[:, "m ZIF-8 [g/g]"]))
	

	#top sensor error labels σ_m
	for (label, layout) in zip(["σₘ=$(σ_m[1])","σₘ=$(σ_m[2])", "σₘ=$(σ_m[3])"], [ideal_fig, med_sensor_error_fig, high_sensor_error_fig])
    Label(layout[1, 1, Top()], 
		  label,
          textsize = 40,
          padding = (0, -385, 25, 0),
		  halign = :center)
	end

	#left water variance labels σ_H₂O
	for (label, layout) in zip(["σH₂O=$(σ_H₂O[1])","σH₂O=$(σ_H₂O[2])", "σH₂O=$(σ_H₂O[3])"], [ideal_fig, med_water_variance_fig, high_water_variance_fig])
	Label(layout[1, 1, Left()], 
	  	  label,
		  textsize = 40,
	  	  padding = (0, 25, 0, 0),
	  	  valign = :center,
	  	  rotation = pi/2)
	end

	#establish axes for 9x9 grid
	axes = [Axis(fig[i, j]) for i in 1:3, j in 1:3]

	for i = 1:3
		for j = 1:3

			#generate test and training data
			num_normal = 100
			num_anomaly = 5

			dataTest  		 = SyntheticDataGen.gen_data(num_normal, num_anomaly, σ_H₂O[i], σ_m[j])
			dataTrain 		 = SyntheticDataGen.gen_data(num_normal, 0, σ_H₂O[i], σ_m[j])
			XTrain, yTrain   = AnomalyDetection.data_to_Xy(dataTrain)
			XTest, yTest     = AnomalyDetection.data_to_Xy(dataTest)
			scaler_temp		 = StandardScaler().fit(XTrain)
			XTrainScaled 	 = scaler_temp.transform(XTrain)
			XTestScaled 	 = scaler_temp.transform(XTest)

			#optimize hyperparameters and determine f1score
			if method == 1
				νOpt, γOpt = determine_ν_opt_γ_opt_hypersphere(XTrainScaled)
			elseif method == 2
				K = trunc(Int, num_normal*0.05)
				νOpt, γOpt = opt_ν_γ_by_density_measure_method(XTrainScaled, K)
			end
			temp_svm   = AnomalyDetection.train_anomaly_detector(XTrainScaled, 
																				 νOpt, 
																				 γOpt)
			yPred 	   = temp_svm.predict(XTestScaled)
			f1score    = AnomalyDetection.performance_metric(yTest, yPred)
			f1score    = truncate(f1score, 2)

			#draw a background box colored according to f1score
			fig[i, j]  = Box(fig, color = (ColorSchemes.RdYlGn_4[f1score], 0.7))
			axes[i, j].title = "f1 score = $(f1score)"
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
					  xlabel = "m, " * mofs[1] * " [g/g]",
					  ylabel = "m, " * mofs[2] * " [g/g]",
					  aspect = 1,
					  xticks = LinearTicks(3),
					  yticks = LinearTicks(3),
					  alignmode = Outside(10))

			for data_l in groupby(dataTest, "label")
				label = data_l[1, "label"]
				
				if label != "low humidity"
					scatter!(ax, 
							 data_l[:, "m $(mofs[1]) [g/g]"], 
							 data_l[:, "m $(mofs[2]) [g/g]"],
							 strokewidth=1,
							 markersize=15,
							 marker=label == "normal" ? :circle : :x,
							 color=(:white, 0.0),
							 strokecolor=SyntheticDataGen.label_to_color[label],
							 label=label)
				end
			end
			
			x₁s, x₂s, predictions = AnomalyDetection.generate_response_grid(temp_svm, scaler_temp, zif71_lims, zif8_lims)
		
			contour!(ax, x₁s, 
					 x₂s,
					 predictions, 
					 levels=[0.0], 
					 color=:black,
					 label="decision boundary") 

			#confusion matrix position RIGHT
			pos = fig[i, j][1, 2] 

			all_labels = SyntheticDataGen.viable_labels
			n_labels   = length(all_labels)

			# confusion matrix. each row pertains to a label.
			# col 1 = -1 predicted anomaly, col 2 = 1 predicted normal.
			cm = zeros(Int, 2, n_labels)
			
			for (l, label) in enumerate(all_labels)
				# get all test data with this label
				data_test_l = filter(row -> row["label"] == label, dataTest)
				# get feature matrix
				X_test_l, y_test_l = AnomalyDetection.data_to_Xy(data_test_l)
				# scale
				X_test_l_scaled = scaler_temp.transform(X_test_l)
				# make predictions for this subset of test data
				y_pred_l = temp_svm.predict(X_test_l_scaled)
				# how many are predicted as anomaly?
				cm[1, l] = sum(y_pred_l .== -1)
				# how many predicted as normal?
				cm[2, l] = sum(y_pred_l .== 1)
			end

			@assert sum(cm) == nrow(dataTest)

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
					 2:6, 
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

	save("sensor_error_&_H2O_variance_plot.pdf", fig)
	fig
end

end
