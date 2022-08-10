module AnomalyDetectionPlots

using ScikitLearn, DataFrames, CairoMakie, ColorSchemes, LinearAlgebra, Statistics, Random, PyCall, LaTeXStrings
SyntheticDataGen = include("SyntheticDataGen.jl")
AnomalyDetection = include("AnomalyDetection.jl")
skopt = pyimport("skopt")

@sk_import svm : OneClassSVM
@sk_import preprocessing : StandardScaler
@sk_import metrics : confusion_matrix
@sk_import metrics : f1_score

gases = ["C₂H₄", "CO₂", "H₂O"]
mofs = ["ZIF-71", "ZIF-8"]

anomaly_labels = ["CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"]
reduced_labels = Dict("normal" => "normal", 
					  "CO₂ buildup" => "CO₂ ↑", 
					  "C₂H₄ buildup" => "C₂H₄ ↑", 
					  "C₂H₄ off" => "C₂H₄ ↓↓", 
					  "CO₂ & C₂H₄ buildup" => "CO₂ & C₂H₄ ↑", 
					  "low humidity" => "H₂O ↓",
					  "anomalous" => "anomaly")

label_to_int = Dict(zip(anomaly_labels, [-1 for i = 1:length(anomaly_labels)]))
label_to_int["normal"]    = 1
label_to_int["anomalous"] = -1

"""
****************************************************
optimize ν, γ via density measure.
****************************************************
"""

"""
visualizes plot of ordered density measures and knee value
"""
function viz_density_measures(X::Matrix{Float64}, K::Int)
	density_measures = AnomalyDetection.compute_density_measures(X, K)
	sorted_density_measures = sort(density_measures)

	elbow_id, density_measure_at_elbow = AnomalyDetection.find_elbow(density_measures)

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
****************************************************
optimize ν, γ via synthetic anomaly hypersphere
****************************************************
"""

"""
visualizes the progression of bayesian optimization of
hyperparameters ν and γ colored and sized according to the error function.
"""
function viz_bayes_values(plot_data::Vector{Tuple{Float64, Float64, Float64}})
    fig = Figure()
    ax = Axis(fig[1, 1], ylabel="γ", xlabel="ν")

	#unpack data
	num_data = length(plot_data)
	νs = [plot_data[i][1] for i=1:num_data]
	γs = [plot_data[i][2] for i=1:num_data]
	Λs = [plot_data[i][3] for i=1:num_data]
	Λs_norm = [1-(Λs[i]-minimum(Λs))/(maximum(Λs)-minimum(Λs)) for i=1:num_data]
	marker_size = 12
	colors = [ColorSchemes.thermal[Λs_norm[i]] for i=1:num_data]

	#plot
	sl = scatterlines!(νs, γs, color=(:grey, 0.3), markersize=marker_size, markercolor=colors)
	Colorbar(fig[1, 2], 
			limits = (maximum(Λs), minimum(Λs)), 
			colormap= reverse(ColorSchemes.thermal), 
			label="error function")

    return fig
end

"""
visualizes the progression of bayesian optimization of
hyperparameters ν and γ colored and sized according to the error function
point by point.
"""
function viz_bayes_values_by_point(plot_data::Vector{Tuple{Float64, Float64, Float64}}, points::Int)

	num_data = length(plot_data)

	xmin = minimum([plot_data[i][1] for i=1:num_data])
	xmax = maximum([plot_data[i][1] for i=1:num_data])
	ymin = minimum([plot_data[i][2] for i=1:num_data])
	ymax = maximum([plot_data[i][2] for i=1:num_data])

	
    fig = Figure()
    ax  = Axis(fig[1, 1], ylabel="γ", xlabel="ν", limits = (0.5xmin, 1.25*xmax, -50*ymin, 1.25*ymax))

	#unpack data
	νs = [plot_data[i][1] for i=1:points]
	γs = [plot_data[i][2] for i=1:points]
	Λs = [plot_data[i][3] for i=1:num_data]
	Λs_norm = [1-(Λs[i]-minimum(Λs))/(maximum(Λs)-minimum(Λs)) for i=1:num_data]

	colors = [ColorSchemes.thermal[Λs_norm[i]] for i=1:num_data]

	#plot
	sl = scatterlines!(νs, γs, color=(:grey, 0.3), markersize=12, markercolor=[colors[i] for i=1:points])
	Colorbar(fig[1, 2], limits = (maximum(Λs), minimum(Λs)), colormap= reverse(ColorSchemes.thermal), label="error function")

	if points == length(Λs)
		ideal_index = argmin(Λs)
		ν_opt = νs[ideal_index]
		γ_opt = γs[ideal_index]
		scatter!([ν_opt], [γ_opt], marker=:x, markersize=25, color=:red)
		text!("($(AnomalyDetectionPlots.truncate(ν_opt, 2)), $(AnomalyDetectionPlots.truncate(γ_opt, 2)))",position = (ν_opt, 1.1*γ_opt), align=(:left, :baseline))
	end

    return fig
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
			  xticks=(1:length(ν_range), ["$(truncate(ν_range[i], 4))" for i=1:length(ν_range)]),
			  yticks=(1:length(γ_range), ["$(truncate(γ_range[i], 4))" for i=1:length(γ_range)]),
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
****************************************************
deployment of the One class support vector machine
****************************************************
"""

"""
visualizes the confusion matrix by anomaly type
"""
function viz_cm(svm, data_test::DataFrame, scaler)
	all_labels = SyntheticDataGen.viable_labels
	n_labels = length(all_labels)

	cm = generate_cm(svm, data_test, scaler, all_labels)

	fig = Figure()
	ax = Axis(fig[1, 1],
		  xticks=([1, 2], ["anomaly", "normal"]),
		  yticks=([i for i=1:n_labels], [reduced_labels[all_labels[i]] for i=1:n_labels]),
		  xticklabelrotation=25.5,
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
    fig
end

"""
generates the confusion matrix to be plotted
"""
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
function viz_decision_boundary(svm, scaler, data_test::DataFrame; res::Int=700, incl_legend::Bool=true, incl_contour::Bool=true)
	X_test, _ = AnomalyDetection.data_to_Xy(data_test)

	xlims = (0.98 * minimum(X_test[:, 1]), 1.02 * maximum(X_test[:, 1]))
	ylims = (0.98 * minimum(X_test[:, 2]), 1.02 * maximum(X_test[:, 2]))

	fig = Figure(resolution=(res, res))
	
	ax = Axis(fig[1, 1], 
			   xlabel = "m, " * mofs[1] * " [g/g]",
			   ylabel = "m, " * mofs[2] * " [g/g]",
			   aspect = DataAspect())
			   
	viz_decision_boundary!(ax, svm, scaler, data_test, xlims, ylims,incl_legend=incl_legend, incl_contour=incl_contour)

	fig
end

function viz_decision_boundary!(ax, 
								svm, 
								scaler,
								data_test::DataFrame, 
								xlims, 
								ylims; 
								incl_legend::Bool=true,
								incl_contour::Bool=true)

	# generate the grid
	x₁s, x₂s, predictions = generate_response_grid(svm, scaler, xlims, ylims)

	viz_responses!(ax, data_test, incl_legend=incl_legend)
	if incl_contour
		contour!(x₁s, 
				x₂s,
				predictions, 
				levels=[0.0], 
				color=:black,
				label="decision boundary"
		) 
	end
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

num_runs: number of iterations to run each plot then sort by f1 score,
the median plot is selected for display
"""
function viz_sensorδ_waterσ_grid(σ_H₂Os::Vector{Float64}, 
								σ_ms::Vector{Float64},
								num_normal_train::Int64,
								num_anomaly_train::Int64,
								num_normal_test::Int64,
								num_anomaly_test::Int64; 
								validation_method::String="hypersphere",
								num_runs::Int=100)

	@assert validation_method=="hypersphere" || validation_method=="knee"

#establish axes and figs for 9x9 grid
	fig  = Figure(resolution = (2400, 1200))
	axes = [Axis(fig[i, j]) for i in 1:3, j in 1:3]
	figs = [fig[i, j] for i in 1:3, j in 1:3]

#top sensor error labels σ_m
	for (label, layout) in zip(["σ, m =$(σ_ms[1])","σ, m =$(σ_ms[2])", "σ, m  = $(σ_ms[3])"], figs[1, 1:3])
		Label(layout[1, 1, Top()], 
			 label,
			 textsize = 40,
			 padding = (0, -385, 25, 0),
			 halign = :center)
	end

#left water variance labels σ_H₂O
	for (label, layout) in zip(["σ, H₂O =$(σ_H₂Os[1])","σ, H₂O =$(σ_H₂Os[2])", "σ, H₂O =$(σ_H₂Os[3])"], figs[1:3, 1])
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

	@assert num_runs%2 == 0
	plot_data_storage = zeros(length(σ_H₂Os), length(σ_H₂Os), num_runs)
	plot_data_storage = convert(Array{Any, 3}, plot_data_storage)

	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)
			for k = 1:num_runs

				plot_data_storage[i, j, k] = Dict{String, Any}("data_test" => SyntheticDataGen.gen_data(num_normal_test, num_anomaly_test, σ_H₂O, σ_m))

				#generate test and training data, feature vectors, target vectors and standard scaler
				plot_data_storage[i, j, k]["data"] = AnomalyDetection.setup_dataset(num_normal_train, 
																					num_anomaly_train, 
																					num_normal_test, 
																					num_anomaly_test, 
																					σ_H₂O, 
																					σ_m)

				#optimize hyperparameters and determine f1score
				if validation_method == "hypersphere"
					(ν_opt, γ_opt), X_sphere, bayes_plot_data = AnomalyDetection.bayes_validation(plot_data_storage[i, j, k]["data"].X_train_scaled, n_iter=40, plot_data_flag=true)
					plot_data_storage[i, j, k]["X_sphere"] = X_sphere
					plot_data_storage[i, j, k]["bayes_plot_data"] = bayes_plot_data
				elseif validation_method == "knee"
					K            = trunc(Int, num_normal_train*0.05)
					ν_opt, γ_opt = AnomalyDetection.opt_ν_γ_by_density_measure_method(plot_data_storage[i, j, k]["data"].X_train_scaled, K)
				end

				plot_data_storage[i, j, k]["svm"]      = AnomalyDetection.train_anomaly_detector(plot_data_storage[i, j, k]["data"].X_train_scaled, ν_opt, γ_opt)
				plot_data_storage[i, j, k]["f1_score"] = truncate(
					AnomalyDetection.performance_metric(plot_data_storage[i, j, k]["data"].y_test, 
														plot_data_storage[i, j, k]["svm"].predict(plot_data_storage[i, j, k]["data"].X_test_scaled)), 2)
			end

			#sort the plot data storage by f1score and identify median data
			
			plot_data_storage[i, j, :] = plot_data_storage[i, j, sortperm([plot_data_storage[i, j, m]["f1_score"] for m=1:num_runs])]
		end
	end
	#sortslices(plot_data_storage,dims=3, by = x -> x["f1_score"])

#This piece identifies the boundaries for the median plot
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)

			#low variance
			if (i < 3) && (j < 3)
			zif71_lims_low_σ = [minimum([minimum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-71 [g/g]"]), zif71_lims_low_σ[1]]), 
					maximum([maximum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-71 [g/g]"]), zif71_lims_low_σ[2]])]

			zif8_lims_low_σ  = [minimum([minimum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-8 [g/g]"]), zif8_lims_low_σ[1]]), 
					maximum([maximum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-8 [g/g]"]), zif8_lims_low_σ[2]])]

			#high variance
			else
			zif71_lims_high_σ = [minimum([minimum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-71 [g/g]"]), zif71_lims_high_σ[1]]), 
					maximum([maximum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-71 [g/g]"]), zif71_lims_high_σ[2]])]

			zif8_lims_high_σ  = [minimum([minimum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-8 [g/g]"]), zif8_lims_high_σ[1]]), 
					maximum([maximum(plot_data_storage[i, j, trunc(Int, num_runs/2)]["data"].data_test[:, "m ZIF-8 [g/g]"]), zif8_lims_high_σ[2]])]
			end
		end
	end

#Set boundaries to be 2% outside min and max data
	zif71_lims_low_σ  = [0.999 * zif71_lims_low_σ[1], 1.001 * zif71_lims_low_σ[2]]
	zif8_lims_low_σ   = [0.999 * zif8_lims_low_σ[1], 1.001 * zif8_lims_low_σ[2]]

	zif71_lims_high_σ = [0.98 * zif71_lims_high_σ[1], 1.02 * zif71_lims_high_σ[2]]
	zif8_lims_high_σ  = [0.98 * zif8_lims_high_σ[1], 1.02 * zif8_lims_high_σ[2]]

#Store the new boundaries to be used in the plots later
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)
			if (i < 3) && (j < 3)
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif71_lims"] = zif71_lims_low_σ
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif8_lims"]  = zif8_lims_low_σ
			else
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif71_lims"] = zif71_lims_high_σ
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif8_lims"]  = zif8_lims_high_σ
			end
		end
	end

#Plot the median data, contour, confusion matrix for each water variance and sensor error value
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)

#identify median data set as the third dim index num_runs/2
			median_data = plot_data_storage[i, j, trunc(Int, num_runs/2)]

#draw a background box colored according to f1score
			fig[i, j]        = Box(fig, color = (ColorSchemes.RdYlGn_4[median_data["f1_score"]], 0.7))
			axes[i, j].title = "F1 score = $(median_data["f1_score"])"
			hidedecorations!(axes[i, j])

#scatter and contour plot position LEFT
			pos = fig[i, j][1, 1] 
			ax  = Axis(fig[i, j][1, 1], 
					  xlabel    = "m, " * mofs[1] * " [g/g]",
					  ylabel    = "m, " * mofs[2] * " [g/g]",
					  aspect    = 1,
					  xticks    = LinearTicks(3),
					  yticks    = LinearTicks(3),
					  alignmode = Outside(10))


			viz_decision_boundary!(ax, 
								  median_data["svm"], 
								  median_data["data"].scaler, 
								  median_data["data"].data_test, 
								  median_data["zif71_lims"], 
								  median_data["zif8_lims"], 
								  incl_legend=false)

#confusion matrix position RIGHT
			pos 	   = fig[i, j][1, 2] 
			all_labels = SyntheticDataGen.viable_labels
			n_labels   = length(all_labels)

			cm = generate_cm(median_data["svm"], 
							median_data["data"].data_test, 
							median_data["data"].scaler, 
							all_labels)

			ax = Axis(fig[i, j][1, 2],
					 xticks=([1, 2], ["anomaly", "normal"]),
					 yticks=([i for i=1:n_labels], [reduced_labels[all_labels[i]] for i=1:n_labels]),
					 xticklabelrotation=25.5,
					 ylabel="truth",
					 xlabel="prediction",
					 alignmode = Outside())

			@assert SyntheticDataGen.viable_labels[1] == "normal"

#Plot cm colors anomalies
			heatmap!(1:2, 
					2:n_labels, 
					cm[:, 2:end], 
					colormap=ColorSchemes.amp, 
					colorrange=(0, maximum(cm[:, 2:end])))

#Plot cm colors normal data
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

#Save finished 3x3 plot
	if validation_method == "hypersphere"
		save("sensor_error_&_H2O_variance_plot_hypersphere.pdf", fig)
	elseif validation_method == "knee"
		save("sensor_error_&_H2O_variance_plot_knee.pdf", fig)
	end

	return plot_data_storage[2, 2, trunc(Int, num_runs/2)], fig
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

	#σ_H₂Os = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050]
	#σ_ms = [0, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00010, 0.00020, 3*10^-4, 0.00040, 0.00050]

	σ_H₂Os = 0:σ_H₂O_max/res:σ_H₂O_max
	σ_ms = 0:σ_m_max/res:σ_m_max

	num_normal_test_points = num_normal_train_points = 100
	num_anomaly_train_points = 0
	num_anomaly_test_points = 5

	f1_score_grid = zeros(length(σ_H₂Os), length(σ_ms))
	

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
						(ν_opt, γ_opt), _ = AnomalyDetection.bayes_validation(data.X_train_scaled, n_iter=40)
					elseif hyperparameter_method == "grid"
						(ν_opt, γ_opt), _ = AnomalyDetection.determine_ν_opt_γ_opt_hypersphere_grid_search(data.X_train_scaled)
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
			
			f1_score_grid[i, res-j+2] = f1_avg/n_avg

			@warn "grid space ($(i), $(j)) finished"

		end
	end

	fig = Figure()
	
	ax = Axis(fig[1, 1],
		  xticks=(1:length(σ_H₂Os), ["$(truncate(σ_H₂Os[i], 4))" for i=1:length(σ_H₂Os)]),
		  yticks=(1:length(σ_ms), ["$(truncate(σ_ms[length(σ_ms)-i+1], 6))" for i=1:length(σ_ms)]),
		  xticklabelrotation=45.0,
		  ylabel="σ, m [g/g]",
		  xlabel="σ, H₂O [relative humidity]"
    )

	hm = heatmap!(1:length(σ_H₂Os), 1:length(σ_ms), f1_score_grid,
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
			(ν_opt, γ_opt), _ = AnomalyDetection.bayes_validation(data_set.X_train_scaled, λ=λ)
			svm = AnomalyDetection.train_anomaly_detector(data_set.X_train_scaled, ν_opt, γ_opt)
			f1_score = AnomalyDetection.performance_metric(data_set.y_test, svm.predict(data_set.X_test_scaled))
			avg_f1_scores[i] += f1_score
		end
	end

	#calculate rolling average of the average f1 scores and ID λ opt
	avg_f1_scores = avg_f1_scores./runs
	rolling_avg_f1 = [mean([avg_f1_scores[j] for j=(i-1):(i+1)]) for i=2:res-1]
	λ_opt = λs[argmax(rolling_avg_f1)+1]
	λ_opt = truncate(λ_opt, 2)

	
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
