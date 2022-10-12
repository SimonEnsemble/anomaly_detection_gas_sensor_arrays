module AnomalyDetectionPlots

using ScikitLearn, DataFrames, CairoMakie, ColorSchemes, LinearAlgebra, Statistics, Random, PyCall, LaTeXStrings, JLD2, Makie.GeometryBasics, Revise, PyCallJLD
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
					  "C₂H₄ off" => "C₂H₄ off", 
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

	#xmin = minimum([plot_data[i][1] for i=1:num_data])
	xmax = maximum([plot_data[i][1] for i=1:num_data])
	#ymin = minimum([plot_data[i][2] for i=1:num_data])
	ymax = maximum([plot_data[i][2] for i=1:num_data])

	
    fig = Figure()
    ax  = Axis(fig[1, 1], ylabel="γ", xlabel="ν", limits = (0, 1.25*xmax, -0.05, 1.25*ymax))

	#unpack data
	νs = [plot_data[i][1] for i=1:points]
	γs = [plot_data[i][2] for i=1:points]
	Λs = [plot_data[i][3] for i=1:num_data]
	Λs_norm = [1-(Λs[i]-minimum(Λs))/(maximum(Λs)-minimum(Λs)) for i=1:num_data]

	colors = [ColorSchemes.thermal[Λs_norm[i]] for i=1:num_data]

	#plot
	sl = scatterlines!(νs, γs, color=(:grey, 0.3), markersize=12, markercolor=[colors[i] for i=1:points])
	Colorbar(fig[1, 2], limits = (maximum(Λs), minimum(Λs)), colormap= reverse(ColorSchemes.thermal), label="error function")

	#indicate starting point
	ν_init = νs[1]
	γ_init = γs[1]
	scatter!([ν_init], [γ_init], marker=:x, markersize=25, color=:green)

	#indicate optimal nu and gamma
	if points == length(Λs)
		ideal_index = argmin(Λs)
		ν_opt = νs[ideal_index]
		γ_opt = γs[ideal_index]
		scatter!([ν_opt], [γ_opt], marker=:x, markersize=25, color=:red)
		text!("($(AnomalyDetectionPlots.truncate(ν_opt, 2)), $(AnomalyDetectionPlots.truncate(γ_opt, 2)))",position = (ν_opt, 1.1*γ_opt), align=(:left, :baseline))
		save("bayes_plot.pdf", fig)
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
visualizes effect on SVM by increasing/decreasing ν and γ
"""
function viz_ν_γ_effects(data::Dict{String, Any},
						ν_opt::Float64,
						γ_opt::Float64)
	scale_factor = 4
	zif71_lims   = (0.99 * minimum(data["data"].data_train[:, "m ZIF-71 [g/g]"]),
				    1.01 * maximum(data["data"].data_train[:, "m ZIF-71 [g/g]"]))
	zif8_lims    = (0.99 * minimum(data["data"].data_train[:, "m ZIF-8 [g/g]"]),
				    1.01 * maximum(data["data"].data_train[:, "m ZIF-8 [g/g]"]))
	γs 			 = [γ_opt/scale_factor, γ_opt, scale_factor*γ_opt]
	νs 			 = [ν_opt/scale_factor, ν_opt, scale_factor*ν_opt]
	x_ticks 	 = [AnomalyDetectionPlots.truncate(zif71_lims[1], 4), AnomalyDetectionPlots.truncate(zif71_lims[2], 4)]
	y_ticks 	 = [AnomalyDetectionPlots.truncate(zif8_lims[1], 4), AnomalyDetectionPlots.truncate(zif8_lims[2], 4)]

	#establish axes and figs for 3x3 grid
	fig  = Figure(resolution = (1400, 1400))
	figs = [fig[i, j] for i in 1:length(νs), j in 1:length(γs)]
	axes = [Axis(fig[i, j], 
				 xlabel = "m, " * AnomalyDetectionPlots.mofs[1] * " [g/g]",
				 ylabel = "m, " * AnomalyDetectionPlots.mofs[2] * " [g/g]",
				 xlabelsize = 25,
				 ylabelsize = 25,
				 xticks = x_ticks,
				 yticks = y_ticks,
				 xlabelpadding = -15,
				 ylabelpadding = -40,
				 aspect = DataAspect()) 
			for i in 1:length(νs), j in 1:length(γs)]

	#top labels
	for (label, layout) in zip(["νₒₚₜ / $(scale_factor)", "νₒₚₜ", "$(scale_factor) × νₒₚₜ"], figs[1, 1:length(νs)])
		Label(layout[1, 1, Top()], 
		  	  label,
			  textsize = 40,
			  padding = (0, 0, 25, 0),
			  halign = :center)
	end

	#left labels
	for (label, layout) in zip(["$(scale_factor) × γₒₚₜ","γₒₚₜ", "γₒₚₜ / $(scale_factor)"], figs[1:length(γs), 1])
		Label(layout[1, 1, Left()], 
			  label,
			  textsize = 40,
			  padding = (0, 50, 0, 0),
			  valign = :center,
			  rotation = pi/2)
	end

	#build plots for grid spaces
	for (i, γ) in enumerate(γs)
		for (j, ν) in enumerate(νs)
			xlims!(axes[i, j], x_ticks)
			ylims!(axes[i, j], y_ticks)

			svm    = AnomalyDetection.train_anomaly_detector(data["data"].X_train_scaled, ν, γ)
			scaler = StandardScaler().fit(data["data"].X_train)
			AnomalyDetectionPlots.viz_decision_boundary!(axes[i, j], 
														svm, 
														scaler, 
														data["data"].data_train, 
														zif71_lims, 
														zif8_lims, 
														incl_legend=false)
			AnomalyDetectionPlots.viz_limit_box!(axes[i, j], x_ticks, y_ticks, 4, dashed=false)
		end
	end

	#Save finished 3x3 plot
	save("ν_γ_plot.pdf", fig)

	return fig
end

"""
****************************************************
gas composition space and sensor calibration visualizations
****************************************************
"""

function viz_C2H4_CO2_H2O_density_compositions(training_data::DataFrame, test_data::DataFrame)

	# this is the figure that will hold the layout (single column of composition plots)
	fig = Figure(resolution=(1200, 1600),figure_padding = 100)

	# create training (left) and testing (right) data axis'
	train_ax = Axis(fig[1, 1], title="Training Data", titlegap=80, titlesize=50)
	test_ax = Axis(fig[1, 2], title="Test Data", titlegap=80, titlesize=50)
	hidedecorations!(train_ax)
	hidedecorations!(test_ax)
	data = [training_data, test_data]
	gas = AnomalyDetection.gases

	# create a vector of axis'
	axs = [[Axis(fig[1, j][i, 1], ylabel="# compositions", title=gas[i], ) for i in 1:length(gas)] for j=1:2]

	# create axis labels
	for i=1:length(gas)
		for j=1:2
			if gas[i] == "H₂O"
				axs[j][i].xlabel = "p, H₂O [relative humidity]"
			else
				axs[j][i].xlabel = "p, " * gas[i] * " [ppm]"
			end
		end
	end

	# plot data histograms
	for i=1:length(gas)
		for j=1:2
			for data_g in groupby(data[j], :label)
			label = data_g[1, "label"]
												
				if gas[i] == "H₂O"
					hist!(axs[j][i], 
						  data_g[:, "p H₂O [bar]"] / SyntheticDataGen.p_H₂O_vapor,
						  label=label,
						  color=(SyntheticDataGen.label_to_color[label], 0.5),
						  alignmode = Outside(10))
				else
					hist!(axs[j][i],
						  data_g[:, "p " * gas[i] * " [bar]"] * 1e6,
						  label=label,	
						  color=(SyntheticDataGen.label_to_color[label], 0.5),
						  alignmode = Outside(10))
				end
			end
		end
	end

	# manually setting the CO2 test data axis ticks
	axs[2][2].xticks = ([5e3, 1e4, 1.5e4, 2e4], ["5000", "10000", "15000", "20000"])
	rowgap!(fig.layout, Relative(0.05))

	# set plot legends
	for i=1:length(gas)
		linkyaxes!(axs[1][i], axs[2][i])
		for j=1:2
			if gas[i] == "H₂O"
				axislegend(axs[j][i], position=:lt)
			else
				axislegend(axs[j][i], position=:rt)
			end
		end
	end
	
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

	viz_cm!(ax, svm, data_test, scaler, gen_cm_flag=false, cm=cm)

    fig
end

function viz_cm!(ax, 
				 svm, 
				 data_test::DataFrame, 
				 scaler; 
				 gen_cm_flag=true, 
				 cm=zeros(2, length(SyntheticDataGen.viable_labels)))

	all_labels = SyntheticDataGen.viable_labels
	n_labels = length(all_labels)

	if gen_cm_flag
		cm = AnomalyDetectionPlots.generate_cm(svm, data_test, scaler, all_labels)
	end

	@assert SyntheticDataGen.viable_labels[1] == "normal"
	# anomalies
	heatmap!(1:1, 2:n_labels, reshape(cm[1, 2:end], (1, 4)),
			      colormap=ColorSchemes.algae, colorrange=(0, 5))
	heatmap!(2:2, 2:n_labels, reshape(cm[2, 2:end], (1, 4)),
			      colormap=ColorSchemes.amp, colorrange=(0, 5))
	# normal data
	heatmap!(1:1, 1:1, [cm[1, 1]],
			      colormap=ColorSchemes.amp, colorrange=(0, 100))
	heatmap!(2:2, 1:1, [cm[2, 1]],
			      colormap=ColorSchemes.algae, colorrange=(0, 100))
    for i = 1:2
        for j = 1:length(all_labels)
            text!("$(cm[i, j])",
                  position=(i, j), align=(:center, :center), 
                  color=cm[i, j] > sum(cm[:, j]) / 2 ? :white : :black)
        end
    end
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
function viz_decision_boundary(svm, 
							   scaler, 
							   data_test::DataFrame; 
							   res::Int=700, 
							   incl_legend::Bool=true, 
							   incl_contour::Bool=true,
							   default_lims=true,
							   xlims::Tuple{Float64, Float64}= (0.01, 0.02),
							   ylims::Tuple{Float64, Float64}= (0.01, 0.02))
	X_test, _ = AnomalyDetection.data_to_Xy(data_test)

	if default_lims
		xlims = (0.98 * minimum(X_test[:, 1]), 1.02 * maximum(X_test[:, 1]))
		ylims = (0.98 * minimum(X_test[:, 2]), 1.02 * maximum(X_test[:, 2]))
	end


	fig = Figure(resolution=(res, res))
	
	ax = Axis(fig[1, 1], 
			   xlabel = "m, " * mofs[1] * " [g/g]",
			   ylabel = "m, " * mofs[2] * " [g/g]",
			   aspect = DataAspect())
			   
	viz_decision_boundary!(ax, svm, scaler, data_test, xlims, ylims,incl_legend=incl_legend, incl_contour=incl_contour)

	if !default_lims
		xlims!(ax, xlims)
		ylims!(ax, ylims)
	end

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
		contour!(ax,
				x₁s, 
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
generates a box to visualize varying limits in data
"""
function viz_limit_box!(ax, 
						xlims::Vector{Float64}, 
						ylims::Vector{Float64}, 
						w::Int;
						dashed::Bool=true)

	line_xs = [[xlims[1], xlims[1]],
			   [xlims[1], xlims[2]],
			   [xlims[2], xlims[2]],
			   [xlims[2], xlims[1]]]
	line_ys = [[ylims[1], ylims[2]],
			   [ylims[2], ylims[2]],
			   [ylims[2], ylims[1]],
			   [ylims[1], ylims[1]]]	
	for i=1:4		
		if dashed
			line_style = :dash  
		else
			line_style = :solid 
		end		
		lines!(ax, line_xs[i], line_ys[i], linestyle=line_style, linewidth=w, color=(:grey, 0.8))
	end

end

"""
vizualizes effects of water variance and sensor error using validation:
method 1: hypersphere
method 2: knee

num_runs: number of iterations to run each plot then sort by f1 score,
the median plot is selected for display

bound tuning format (x1, x2, y1, y2)
"""
function viz_sensorδ_waterσ_grid(σ_H₂Os::Vector{Float64}, 
								σ_ms::Vector{Float64},
								num_normal_train::Int64,
								num_anomaly_train::Int64,
								num_normal_test::Int64,
								num_anomaly_test::Int64; 
								validation_method::String="hypersphere",
								num_runs::Int=100,
								gen_data_flag=true,
								tune_bounds_flag=true,
								bound_tuning_low_variance=(-0.01, 0.01, -0.01, 0.01),
								bound_tuning_high_variance=(-0.01, 0.01, -0.01, 0.01))

	@assert validation_method=="hypersphere" || validation_method=="knee"


#establish axes and figs for 9x9 grid
	fig  = Figure(resolution = (2400, 1400))
	axes = [Axis(fig[i, j]) for i in 1:3, j in 1:3]
	figs = [fig[i, j] for i in 1:3, j in 1:3]

#top sensor error labels σ_m
	for (label, layout) in zip(["σ, m =$(σ_ms[1]) [g/g]","σ, m =$(σ_ms[2]) [g/g]", "σ, m  = $(σ_ms[3]) [g/g]"], figs[1, 1:3])
		Label(layout[1, 1, Top()], 
			 label,
			 textsize = 40,
			 padding = (0, -385, 25, 0),
			 halign = :center)
	end

#left water variance labels σ_H₂O
	for (label, layout) in zip(["σ, H₂O =$(σ_H₂Os[1]) [RH]","σ, H₂O =$(σ_H₂Os[2]) [RH]", "σ, H₂O =$(σ_H₂Os[3]) [RH]"], figs[1:3, 1])
		Label(layout[1, 1, Left()], 
			 label,
			 textsize = 40,
			 padding = (0, 25, 0, 0),
			 valign = :center,
			 rotation = pi/2)
	end

if gen_data_flag
#find max/min for plots
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

				#generate test and training data, feature vectors, target vectors and standard scaler
				plot_data_storage[i, j, k] = Dict{String, Any}("data" => AnomalyDetection.setup_dataset(num_normal_train, 
																					num_anomaly_train, 
																					num_normal_test, 
																					num_anomaly_test, 
																					σ_H₂O, 
																					σ_m))

				#optimize hyperparameters and determine f1score
				if validation_method == "hypersphere"
					(ν_opt, γ_opt), X_sphere, bayes_plot_data = AnomalyDetection.bayes_validation(plot_data_storage[i, j, k]["data"].X_train_scaled, n_iter=50, plot_data_flag=true)
					plot_data_storage[i, j, k]["X_sphere"] = X_sphere
					plot_data_storage[i, j, k]["bayes_plot_data"] = bayes_plot_data
					plot_data_storage[i, j, k]["ν_opt, γ_opt"] = (ν_opt, γ_opt)
				elseif validation_method == "knee"
					K            = trunc(Int, num_normal_train*0.05)
					ν_opt, γ_opt = AnomalyDetection.opt_ν_γ_by_density_measure_method(plot_data_storage[i, j, k]["data"].X_train_scaled, K)
					plot_data_storage[i, j, k]["ν_opt, γ_opt"] = (ν_opt, γ_opt)
				end

				plot_data_storage[i, j, k]["svm"]      = AnomalyDetection.train_anomaly_detector(plot_data_storage[i, j, k]["data"].X_train_scaled, ν_opt, γ_opt)
				plot_data_storage[i, j, k]["f1_score"] = truncate(
					AnomalyDetection.performance_metric(plot_data_storage[i, j, k]["data"].y_test, 
														plot_data_storage[i, j, k]["svm"].predict(plot_data_storage[i, j, k]["data"].X_test_scaled)), 2)
			end
			@warn "grid space ($(i), $(j)) finished"
			#sort the plot data storage by f1score and identify median data
			
			plot_data_storage[i, j, :] = plot_data_storage[i, j, sortperm([plot_data_storage[i, j, m]["f1_score"] for m=1:num_runs])]
		end
	end
	#sortslices(plot_data_storage,dims=3, by = x -> x["f1_score"])

#This piece identifies the boundaries for the median plot
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)

			#low variance
			if (i > 1)
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

#Set boundaries to be slightly outside min and max data
	zif71_lims_low_σ  = [1.02 * zif71_lims_low_σ[1], 1.05 * zif71_lims_low_σ[2]]
	zif8_lims_low_σ   = [1.02 * zif8_lims_low_σ[1], 1.05 * zif8_lims_low_σ[2]]

	zif71_lims_high_σ = [0.98 * zif71_lims_high_σ[1], 1.05 * zif71_lims_high_σ[2]]
	zif8_lims_high_σ  = [0.98 * zif8_lims_high_σ[1], 1.05 * zif8_lims_high_σ[2]]

#Store the new boundaries to be used in the plots later
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)
			if (i > 1)
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif71_lims"] = zif71_lims_low_σ
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif8_lims"]  = zif8_lims_low_σ
			else
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif71_lims"] = zif71_lims_high_σ
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif8_lims"]  = zif8_lims_high_σ
			end
		end
	end
else
	#plot_data_storage = JLD.load("sensor_error_&_H2O_variance_plot.jld", "plot_data_storage")
	@load "sensor_error_&_H2O_variance_plot.jld" plot_data_storage

	#due to an issue with jld2 storing the SVM py object as null, I have to retrain the SVM
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)
			mid_num = trunc(Int, num_runs/2)

			ν_opt = plot_data_storage[i, j, mid_num]["ν_opt, γ_opt"][1]
			γ_opt = plot_data_storage[i, j, mid_num]["ν_opt, γ_opt"][2]
			plot_data_storage[i, j, mid_num]["svm"] = AnomalyDetection.train_anomaly_detector(plot_data_storage[i, j, mid_num]["data"].X_train_scaled, ν_opt, γ_opt)
			plot_data_storage[i, j, mid_num]["data"].scaler = StandardScaler().fit(plot_data_storage[i, j, mid_num]["data"].X_train)
		end
	end
end

if tune_bounds_flag
	zif71_lims_low_σ = plot_data_storage[2, 1, trunc(Int, num_runs/2)]["zif71_lims"]
	zif8_lims_low_σ = plot_data_storage[2, 1, trunc(Int, num_runs/2)]["zif8_lims"]
	zif71_lims_high_σ = plot_data_storage[1, 1, trunc(Int, num_runs/2)]["zif71_lims"]
	zif8_lims_high_σ = plot_data_storage[1, 1, trunc(Int, num_runs/2)]["zif8_lims"]

#fine tuning boundaries
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)
			if (i > 1)
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif71_lims"] = [zif71_lims_low_σ[1] + bound_tuning_low_variance[1],  zif71_lims_low_σ[2] + bound_tuning_low_variance[2]]
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif8_lims"]  = [zif8_lims_low_σ[1] + bound_tuning_low_variance[3],  zif8_lims_low_σ[2] + bound_tuning_low_variance[4]]
			else
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif71_lims"] = [zif71_lims_high_σ[1] + bound_tuning_high_variance[1], zif71_lims_high_σ[2] + bound_tuning_high_variance[2]]
			plot_data_storage[i, j, trunc(Int, num_runs/2)]["zif8_lims"]  = [zif8_lims_high_σ[1] + bound_tuning_high_variance[3], zif8_lims_high_σ[2] + bound_tuning_high_variance[4]]
			end
		end
	end
end

#Plot the median data, contour, confusion matrix for each water variance and sensor error value
	for (i, σ_H₂O) in enumerate(σ_H₂Os)
		for (j, σ_m) in enumerate(σ_ms)

#identify median data set as the third dim index num_runs/2
			median_data = plot_data_storage[i, j, trunc(Int, num_runs/2)]
			median_data["zif71_lims"] = [truncate(median_data["zif71_lims"][1], 4), truncate(median_data["zif71_lims"][2], 4)]
			median_data["zif8_lims"] = [truncate(median_data["zif8_lims"][1], 4), truncate(median_data["zif8_lims"][2], 4)]

#draw a background box colored according to f1score
			fig[i, j]        = Box(fig)#, color = (ColorSchemes.RdYlGn_4[median_data["f1_score"]], 0.7)) #incl to color by f1
			axes[i, j].title = "F1 score = $(median_data["f1_score"])"
			hidedecorations!(axes[i, j])

#scatter and contour plot position LEFT
			pos = fig[i, j][1, 1] 
			ax  = Axis(fig[i, j][1, 1], 
					  xlabel    = "m, " * mofs[1] * " [g/g]",
					  ylabel    = "m, " * mofs[2] * " [g/g]",
					  xticks    = median_data["zif71_lims"],
					  yticks    = median_data["zif8_lims"],
					  aspect    = DataAspect(),
					  alignmode = Outside(10))

			xlims!(ax, median_data["zif71_lims"])
			ylims!(ax, median_data["zif8_lims"])


			viz_decision_boundary!(ax, 
								  median_data["svm"], 
								  median_data["data"].scaler, 
								  median_data["data"].data_test, 
								  median_data["zif71_lims"], 
								  median_data["zif8_lims"], 
								  incl_legend=false)

#draw a box according to the low variance boundaries to aid visualization of limits
			low_σ_data_lims = (plot_data_storage[2, 1, trunc(Int, num_runs/2)]["zif71_lims"], plot_data_storage[2, 1, trunc(Int, num_runs/2)]["zif8_lims"])
			box_line_width = 3
			viz_limit_box!(ax, low_σ_data_lims[1], low_σ_data_lims[2], box_line_width)

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
					 xlabel="prediction",
					 alignmode = Outside(10))

			viz_cm!(ax, 
					median_data["svm"], 
					median_data["data"].data_test, 
					median_data["data"].scaler, 
					gen_cm_flag=false, 
					cm=cm)
		end
	end

#Save finished 3x3 plot
	if validation_method == "hypersphere"
		save("sensor_error_&_H2O_variance_plot_hypersphere.pdf", fig)
	elseif validation_method == "knee"
		save("sensor_error_&_H2O_variance_plot_knee.pdf", fig)
	end

	if gen_data_flag
		#JLD.save("sensor_error_&_H2O_variance_plot.jld", "plot_data_storage", plot_data_storage)
		@save "sensor_error_&_H2O_variance_plot.jld" plot_data_storage
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
							  res::Int=5, 
							  validation_method="knee",
							  hyperparameter_method="bayesian", 
							  n_avg::Int=10, 
							  λ=0.5,
							  gen_data_flag=true)
	@assert validation_method=="hypersphere" || validation_method=="knee"
	@assert hyperparameter_method=="bayesian" || hyperparameter_method=="grid"

	σ_H₂Os = [σ_H₂O_max * 10.0^(-res + i) for i=1:res]
	σ_ms = reverse([σ_m_max * 10.0^(-res + i) for i=1:res])

	num_normal_test_points = num_normal_train_points = 100
	num_anomaly_train_points = 0
	num_anomaly_test_points = 5
	f1_score_grid = zeros(res, res)
	
	if gen_data_flag
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
							ν_space::Tuple{Float64, Float64}=(1.0e-3, 0.3)
							#γ_space::Tuple{Float64, Float64}=(1.0e-3, 0.99)
							(ν_opt, γ_opt), _ = AnomalyDetection.bayes_validation(data.X_train_scaled, n_iter=50) #, γ_space=γ_space)
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
				
				f1_score_grid[i, res-j+1] = f1_avg/n_avg
				@warn "grid space ($(i), $(j)) finished, σ_H₂O=$(σ_H₂O), σ_m=$(σ_m), f1=$(f1_avg)"
			end
		end
	else
		@load "f1_score_plot.jld2" f1_score_grid
	end

	fig = Figure()
	
	ax = Axis(fig[1, 1],
		  yticks=(1:length(σ_H₂Os), ["$(truncate(σ_H₂Os[i], 5))" for i=1:length(σ_H₂Os)]),
		  xticks=(1:length(σ_ms), ["$(truncate(σ_ms[length(σ_ms)-i+1], 8))" for i=1:length(σ_ms)]),
		  xticklabelrotation=45.0,
		  xlabel="σ, m [g/g]",
		  ylabel="σ, H₂O [relative humidity]"
    )

	hm = heatmap!(1:length(σ_H₂Os), 1:length(σ_ms), f1_score_grid,
			      colormap=ColorSchemes.RdYlGn_4, colorrange=(0.0, 1.0))
	Colorbar(fig[1, 2], hm, label="F1 score")

	if validation_method == "hypersphere"
		save("f1_score_plot_hypersphere.pdf", fig)
	elseif validation_method == "knee"
		save("f1_score_plot_knee.pdf", fig)
	end

	if gen_data_flag
		@save "f1_score_plot.jld2" f1_score_grid
	end

	return fig
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
