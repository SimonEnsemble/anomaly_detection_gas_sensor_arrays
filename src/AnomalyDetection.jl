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

#= 
to find optimal ν, γ in an unsupervised way, via density measure.
=#
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

function opt_ν_γ_by_density_measure_method(X::Array{Float64}, K::Int)
	density_measures = compute_density_measures(X, K)

	elbow_id, density_measure_at_elbow = find_elbow(density_measures)

	nb_data = size(X)[1]
	ν_opt = (nb_data - elbow_id) / nb_data
	γ_opt = 1 / density_measure_at_elbow
	return ν_opt, γ_opt
end

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

function train_anomaly_detector(X_scaled::Matrix, ν::Float64, γ::Float64)
	oc_svm = OneClassSVM(kernel="rbf", nu=ν, gamma=γ)
	return oc_svm.fit(X_scaled)
end

function performance_metric(y_true, y_pred)
    # anomalies (-1) are considered "positives" so need to switch sign.
    return f1_score(-y_true, -y_pred)
end

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
                  position=(i, j), align=(:center, :center), color=:black)
        end
    end
    # Colorbar(fig[1, 2], hm, label="# data points")
    fig
end

#function to generate a grid of anomaly scores based on a trained svm and given resolution.
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

#function to generate and visualize a SVM given a particular nu, gamma and resolution.
function viz_decision_boundary(svm, scaler, data_test::DataFrame, res::Int=100)
    X_test, _ = data_to_Xy(data_test)

	xlims = (minimum(X_test[:, 1]), maximum(X_test[:, 1]))
	ylims = (minimum(X_test[:, 2]), maximum(X_test[:, 2]))

	# generate the grid
	x₁s, x₂s, predictions = generate_response_grid(svm, scaler, xlims, ylims)
	fig = Figure(resolution=(700, 700))
	
	
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

end
