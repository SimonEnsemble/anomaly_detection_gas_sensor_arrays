### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ d090131e-6602-4c03-860c-ad3cb6c7844a
using CSV, DataFrames, ColorSchemes, Distributions, PlutoUI, ScikitLearn, Colors, Random, JLD, JLD2, LinearAlgebra, PyCall, Makie.GeometryBasics, CairoMakie

# ╔═╡ 0a6fe423-c3be-4a75-aa27-dfb84fde7fef
SyntheticDataGen = include("src/SyntheticDataGen.jl")

# ╔═╡ 3e7c36ca-8345-40fb-b199-34fe49dea73e
AnomalyDetection = include("src/AnomalyDetection.jl")

# ╔═╡ 4745788b-d360-4305-b44b-8d0fca2aeb4f
AnomalyDetectionPlots = include("src/AnomalyDetectionPlots.jl")

# ╔═╡ 31f71438-ff2f-49f9-a801-3a6489eaf271
include("plot_theme.jl")

# ╔═╡ 1784c510-5465-11ec-0dd1-13e5a66e4ce6
md"# Anomaly Detection for Gas Sensor Arrays Using Support Vector Data Description (One class SVM) in a Non-Injective System.
"

# ╔═╡ 3ba4e1e5-3187-4811-be09-d990973abc77
TableOfContents()

# ╔═╡ d578dbf8-dc4d-4a80-a68f-e12284a75953
begin
	gases = ["C₂H₄", "CO₂", "H₂O"]
	mofs = ["ZIF-71", "ZIF-8"]
	anomaly_labels = ["CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"]
viable_labels = vcat(["normal"], anomaly_labels)
end

# ╔═╡ 5d920ea0-f04d-475f-b05b-86e7b199d7e0
begin
	@sk_import preprocessing : StandardScaler
	@sk_import metrics : confusion_matrix
	@sk_import metrics : precision_score
	@sk_import metrics : f1_score
	@sk_import metrics : recall_score
	@sk_import svm : OneClassSVM
end

# ╔═╡ 52ac8252-51a2-484c-9dac-bbdafa40de41
md"### Loading previously stored data using JLD
"

# ╔═╡ 21589bf0-e7f0-4cf1-b082-e236cf6b3221
begin
	jld_file_folder = "example"
	jld_file = joinpath(jld_file_folder, "sensor_error_&_H2O_variance_plot.jld")
end

# ╔═╡ 1e30612e-7bcd-47dc-a1fb-1e127aad4a55
plot_data_storage = JLD.load(jld_file, "plot_data_storage")

# ╔═╡ 4348a594-aa99-45dd-af3f-f3b61a4e8142
md"## Generating data and visualizing the effects of sensor error and water variance.
"

# ╔═╡ e5eede17-08bd-4120-846e-36a3058c003e
md"### For new data: set gen\_data_flag to true.
"

# ╔═╡ ebf79f0c-8399-42bf-b790-d4934906ede0
md"!!! example \"\" 
	Generate 100 3x3 plots of SVDD for low, medium, and high measurement error and H₂O composition variance values and return the plot that yields the median F1 score for each measurement error and H₂O variance set. Then use the data for the middle error and variance to perform a more detailed analysis. "

# ╔═╡ 853390f9-6519-4df3-aa24-7b337142dbe4
md"!!! warning \"\" 
	WARNING: generating new data is very computationally expensive. "

# ╔═╡ 075d4a2f-cf63-47b1-b309-14df97672a65
gen_data_flag = false

# ╔═╡ 4b1759a7-eba1-4de5-8d6a-38106f3301c9
begin
	#visualization of the effects of sensor error and water vapor variance
	σ_H₂O_vector = [ 1e-1, 1e-2, 1e-5] #big to small
	σ_m_vector   = [ 1e-8, 1e-5, 1e-4] #small to big

	num_normal_train_points  = 100
	num_anomaly_train_points = 0
	num_normal_test_points   = 100
	num_anomaly_test_points  = 10
	
	mid_data, plot  = AnomalyDetectionPlots.viz_sensorδ_waterσ_grid(σ_H₂O_vector, 
							 σ_m_vector,
							 num_normal_train_points,
							 num_anomaly_train_points,
							 num_normal_test_points,
							 num_anomaly_test_points,
							 validation_method="hypersphere",
							 num_runs=100,
							 gen_data_flag=gen_data_flag,
							 jld_file_location=jld_file_folder,
							 tune_bounds_flag=true,
							 bound_tuning_low_variance=(-0.0004, -0.0006, -0.0002,      -0.0005),
						     bound_tuning_high_variance=(-0.0001, 0.0001, -0.0002,         0.0))
	plot
end

# ╔═╡ a6700f58-c006-4893-8437-8e6c2b3048f7
#SyntheticDataGen.viz_C2H4_CO2_H2O_density_distributions(σ_H₂O)

# ╔═╡ 2083f6c8-429c-40bb-a029-f9d3131886e7
md"""
# Gas composition space visuals
"""

# ╔═╡ 5a4c66d9-0166-4c13-8a30-02ed6481b6fe
md"""
### Training CO₂ and H₂C₄
"""

# ╔═╡ ec8c2bbc-f492-4a22-80a8-f125ea048b34
#SyntheticDataGen.viz_C2H4_CO2_composition(mid_data["data"].data_train)

# ╔═╡ 7a43a79f-de0f-469b-b8d9-a25b4c3f9180
md"""
### Test CO₂ and H₂C₄
"""

# ╔═╡ a6c181fe-e73d-46af-a1e6-5b4740ae89e4
#SyntheticDataGen.viz_C2H4_CO2_composition(mid_data["data"].data_test)

# ╔═╡ 8317251c-69ad-42d8-90df-6e2a5cc94b13
md"""
### Water
"""

# ╔═╡ bfe24d5a-de4d-4634-ad5d-0c093a17135a
SyntheticDataGen.viz_H2O_compositions(mid_data["data"].data_train)

# ╔═╡ 38f64f12-7eb3-4029-ad94-12307a5ee885
"""
visualizes sensor response data by label
"""
function viz_responses!(ax, data::DataFrame; incl_legend::Bool=true, water_only::Bool=false)
	for data_l in groupby(data, "label")
		label = data_l[1, "label"]
		if !water_only
			scatter!(ax, data_l[:, "m $(mofs[1]) [g/g]"], data_l[:, "m $(mofs[2]) [g/g]"],
					strokewidth=1,
					markersize=15,
					marker=label == "normal" ? :circle : :x,
					color=(:white, 0.0),
					strokecolor=SyntheticDataGen.label_to_color[label],
					label=label)
		elseif water_only && label == "low humidity"
			scatter!(ax, data_l[:, "m $(mofs[1]) [g/g]"], data_l[:, "m $(mofs[2]) [g/g]"],
					strokewidth=1,
					markersize=10,
					marker=label == "normal" ? :circle : :x,
					color=(:white, 0.0),
					strokecolor=SyntheticDataGen.label_to_color[label],
					label=label)
		end
    end

	if incl_legend
    	#axislegend(ax, position=:rb, labelsize=19)
	end
end

# ╔═╡ 43945e20-929f-4045-8c44-66eb4a149483
begin
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
								   viz_σ::Bool=true,
								   xlims::Tuple{Float64, Float64}= (0.01, 0.02),
								   ylims::Tuple{Float64, Float64}= (0.01, 0.02),
								   σ_m="1.0e-5",
								   σ_h₂o="0.01", 
								   water_only::Bool=false)
		X_test, _ = AnomalyDetection.data_to_Xy(data_test)
	
		if default_lims
			xlims = (0.98 * minimum(X_test[:, 1]), 1.02 * maximum(X_test[:, 1])+0.0001)
			ylims = (0.98 * minimum(X_test[:, 2]), 1.02 * maximum(X_test[:, 2]))
		end
	
		if water_only
			fig = Figure(resolution=(res, res))
		else
			fig = Figure(resolution=(res, res))
		end
		

		ax = Axis(fig[1:5, 1:5], 
				   xlabel = rich("m", CairoMakie.subscript("ZIF-71"), " [g gas/g ZIF]"),
				   ylabel = rich("m", CairoMakie.subscript("ZIF-8"), " [g gas/g ZIF]"),
				   aspect = DataAspect(),
				   xlabelsize=31,
				   xticklabelsize=25,
				   ylabelsize=31,
				   yticklabelsize=25,)

		water_ax = Axis(fig[3:4, 4:5], 
				   xlabel = rich("m", CairoMakie.subscript("ZIF-71"), " [g gas/g ZIF]"),
				   ylabel = rich("m", CairoMakie.subscript("ZIF-8"), " [g gas/g ZIF]"),
				   aspect = DataAspect(),
			xticks = WilkinsonTicks(5),
				   xlabelsize=31,
				   xticklabelsize=25,
				   ylabelsize=31,
				   yticklabelsize=25)

		sub_plot!(water_ax, svm, scaler, data_test, res=res, xlims=(0.0078, 0.0147), ylims=(0.0108, 0.0197), default_lims=false, water_only=true)

		#ax2 = Axis()
				   
		viz_decision_boundary!(ax, svm, scaler, data_test, xlims, ylims,incl_legend=incl_legend, incl_contour=incl_contour, water_only=water_only)
	
		if !default_lims
			xlims!(ax, xlims)
			ylims!(ax, ylims)
		end
	
		if viz_σ && !water_only
			Label(fig[2, 1], rich("σ", CairoMakie.subscript("m"), " [g/g] = $(σ_m)"), 
					tellwidth=false, 
					tellheight=false, 
					halign=-0.12, 
					valign=0.95,
				  	fontsize=19)
			Label(fig[2, 1], rich("σ", CairoMakie.subscript("H2O"), " [RH] = $(σ_h₂o)"), 
					tellwidth=false, 
					tellheight=false, 
					halign=-0.12, 
					valign=0.65,
					fontsize=19)
		end
		if !water_only
			Legend(fig[2:4, 6], ax)
		else
			hidedecorations!(ax)
			x_vals = [xlims[1], xlims[2], xlims[2], xlims[1], xlims[1]]
			y_vals = [ylims[1], ylims[1], ylims[2], ylims[2], ylims[1]]
			lines!(ax, x_vals, y_vals, color=:gray)
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
									incl_contour::Bool=true,
									res::Int64=300, 
								    water_only::Bool=false)
	
		# generate the grid
		x₁s, x₂s, predictions = generate_response_grid(svm, scaler, xlims, ylims, res=res)
	
		viz_responses!(ax, data_test, incl_legend=incl_legend, water_only=water_only)
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

	function generate_response_grid(svm, scaler, xlims, ylims; res=100)
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
end

# ╔═╡ 5c4dd3f5-f62c-4d76-b03b-2acfd992969e
	function sub_plot!(ax, svm, 
					   scaler, 
					   data_test::DataFrame; 
					   res::Int=700, 
					   incl_legend::Bool=true, 
					   incl_contour::Bool=true,
					   default_lims=true,
					   viz_σ::Bool=true,
					   xlims::Tuple{Float64, Float64}= (0.01, 0.02),
					   ylims::Tuple{Float64, Float64}= (0.01, 0.02),
					   σ_m="1.0e-5",
					   σ_h₂o="0.01", 
					   water_only::Bool=false)
		X_test, _ = AnomalyDetection.data_to_Xy(data_test)

		poly!(ax, Point2f[(xlims[1], ylims[1]), (xlims[2], ylims[1]), (xlims[2], ylims[2]), (xlims[1], ylims[2])], color = ColorSchemes.dense[0.01], strokecolor = ColorSchemes.grays[0.5], strokewidth = 2)
	
		if default_lims
			xlims = (0.98 * minimum(X_test[:, 1]), 1.02 * maximum(X_test[:, 1])+0.0001)
			ylims = (0.98 * minimum(X_test[:, 2]), 1.02 * maximum(X_test[:, 2]))
		end
	
		if water_only
			fig = Figure(resolution=(res, res))
		else
			fig = Figure(resolution=(res, res))
		end
		#=
		if !water_only
		ax = Axis(fig[1, 1], 
				   xlabel = rich("m", CairoMakie.subscript("ZIF-71"), " [g gas/g ZIF]"),
				   ylabel = rich("m", CairoMakie.subscript("ZIF-8"), " [g gas/g ZIF]"),
				   aspect = DataAspect(),
				   xlabelsize=31,
				   xticklabelsize=25,
				   ylabelsize=31,
				   yticklabelsize=25)
		else
		ax = Axis(fig[1, 1], 
				   xlabel = rich("m", CairoMakie.subscript("ZIF-71"), " [g gas/g ZIF]"),
				   ylabel = rich("m", CairoMakie.subscript("ZIF-8"), " [g gas/g ZIF]"),
				   aspect = DataAspect(),
			xticks = WilkinsonTicks(5),
				   xlabelsize=31,
				   xticklabelsize=25,
				   ylabelsize=31,
				   yticklabelsize=25,
					title="low humidity",
					titlesize=30)
		end =#
		#ax2 = Axis()
				   
		viz_decision_boundary!(ax, svm, scaler, data_test, xlims, ylims,incl_legend=incl_legend, incl_contour=incl_contour, water_only=water_only)
	
		if !default_lims
			xlims!(ax, xlims)
			ylims!(ax, ylims)
		end
	
		if viz_σ && !water_only
			Label(fig[1, 1], rich("σ", CairoMakie.subscript("m"), " [g/g] = $(σ_m)"), 
					tellwidth=false, 
					tellheight=false, 
					halign=0.12, 
					valign=0.85,
				  	fontsize=21)
			Label(fig[1, 1], rich("σ", CairoMakie.subscript("H2O"), " [RH] = $(σ_h₂o)"), 
					tellwidth=false, 
					tellheight=false, 
					halign=0.12, 
					valign=0.8,
					fontsize=21)
		end
		if !water_only
			Legend(fig[1, 2], ax)
		else
			hidedecorations!(ax)
			#=
			x_vals = [xlims[1], xlims[2], xlims[2], xlims[1], xlims[1]]
			y_vals = [ylims[1], ylims[1], ylims[2], ylims[2], ylims[1]]
			lines!(ax, x_vals, y_vals, color=:gray)
			=#
		end
	end

# ╔═╡ 77382f3e-98b6-4aef-b946-8375018c3c3e
md"# Step 1) Generate uniform hypersphere of synthetic data around normal training data.
"

# ╔═╡ 6f53b700-6eba-487b-b91b-085d6e4d38b9
mid_data["data"]

# ╔═╡ 3117881e-08e5-435b-b088-be9973bec8aa
AnomalyDetectionPlots.viz_synthetic_anomaly_hypersphere(mid_data["X_sphere"], mid_data["data"].X_train_scaled, mid_data["data"])

# ╔═╡ 7af3b1f6-2c57-40c4-a841-961dd039090a
md"# Step 2) Minimize error function Λ using bayesian optimization.
"

# ╔═╡ 7990ef58-1e45-44d0-8add-ba410a48dc98
#AnomalyDetectionPlots.viz_bayes_values_by_point(mid_data["bayes_plot_data"], length(mid_data["bayes_plot_data"]))

# ╔═╡ 97a7e102-1a87-4364-9835-c7ed370f573c
md"# Step 3) Train one class support vector machine and evaluate performance.
"

# ╔═╡ 86ba61e6-0633-431f-93a1-b53a8de9dd46
begin
	# Finding ideal xlims and ylims for plots to show training and test data
	total_zif71_data = vcat(mid_data["data"].X_test[:, 1], mid_data["data"].X_train[:, 1])
	total_zif8_data = vcat(mid_data["data"].X_test[:, 2], mid_data["data"].X_train[:, 2])
	xlims = (0.98 * minimum(total_zif71_data), 1.02 * maximum(total_zif71_data))
    ylims= (0.98 * minimum(total_zif8_data), 1.02 * maximum(total_zif8_data))

	# Set high variance data for ν and γ effects plot
	high_σm_data = plot_data_storage[2, 3, 50]
end

# ╔═╡ ccbe1d74-df04-4dbf-9ee4-683890963892
md"## Decision boundary and data
"

# ╔═╡ b67f7643-3994-4e04-8a5c-e748c3c54346
AnomalyDetectionPlots.viz_decision_boundary(mid_data["svm"], mid_data["data"].scaler, mid_data["data"].data_train, xlims=(0.0132, 0.0143), ylims=(0.0178, 0.0195), default_lims=false)

# ╔═╡ 13f4acb4-4434-4ac3-97be-900be400d908
begin
	my_test = mid_data["data"].data_test
	low_humidity_data = SyntheticDataGen.gen_data(0, 10, 0.01, 1.0*10^-5, only_water=true)
	append!(my_test, low_humidity_data)
end

# ╔═╡ 6e278c3e-45a3-4aa8-b904-e3dfa73615d5
viz_decision_boundary(mid_data["svm"], mid_data["data"].scaler, my_test, xlims=(0.0132, 0.0148), ylims=(0.0178, 0.0195), default_lims=false)

# ╔═╡ c5e13fe4-3a7e-4aa0-9550-d56c18f673bf
#

# ╔═╡ 95344881-f912-412d-8d9b-42b8bb3452b9
my_test

# ╔═╡ 0b416525-f0b8-496f-98e6-90e6a6f5cbcd
function gen_gas_comps(n_compositions::Int, label::String, σ_H₂O::Float64)
    gas_comp_distn = setup_gas_comp_distn(σ_H₂O, label)

    data = DataFrame("p C₂H₄ [bar]" => zeros(n_compositions), 
                     "p CO₂ [bar]"  => zeros(n_compositions),
                     "p H₂O [bar]"  => zeros(n_compositions),
                     "label" => [label for _ = 1:n_compositions]
                    )

    for i = 1:n_compositions
        data[i, "p C₂H₄ [bar]"] = rand(gas_comp_distn.f_C₂H₄)
        data[i, "p CO₂ [bar]"]  = rand(gas_comp_distn.f_CO₂)
        data[i, "p H₂O [bar]"]  = rand(gas_comp_distn.f_H₂O)
        data[i, "label"]      = label
    end
    return data
end

# ╔═╡ bb9b1c23-db1e-48bb-9b47-1ba239470123
AnomalyDetectionPlots.viz_decision_boundary(mid_data["svm"], mid_data["data"].scaler, mid_data["data"].data_train, xlims=xlims, ylims=ylims)

# ╔═╡ c930cd71-446c-47f5-8bed-15602afa2304
md"## Confusion Matrix
"

# ╔═╡ 2041cc5f-d583-4fca-bf75-6bee5d6d876b


# ╔═╡ 8b9d65e9-2c28-4eca-a44e-1ae051300777
function viz_cm!(ax, 
				 svm, 
				 data_test::DataFrame, 
				 scaler; 
				 gen_cm_flag=true, 
				 cm=zeros(2, length(viable_labels)))

	all_labels = viable_labels
	n_labels = length(all_labels)

	if gen_cm_flag
		cm = AnomalyDetectionPlots.generate_cm(svm, data_test, scaler, all_labels)
	end

	@assert SyntheticDataGen.viable_labels[1] == "normal"
	good_colors = reverse(ColorSchemes.diverging_gwr_55_95_c38_n256[0:0.01:0.5])
	bad_colors = ColorSchemes.diverging_gwr_55_95_c38_n256[0.5:0.01:1.0]

	# anomalies
	heatmap!(1:1, 2:n_labels, reshape(cm[1, 2:end], (1, 5)),
			      colormap=good_colors, colorrange=(0, 5))
	heatmap!(2:2, 2:n_labels, reshape(cm[2, 2:end], (1, 5)),
			      colormap=bad_colors, colorrange=(0, 5))
	# normal data
	heatmap!(1:1, 1:1, [cm[1, 1]],
			      colormap=bad_colors, colorrange=(0, 100))
	heatmap!(2:2, 1:1, [cm[2, 1]],
			      colormap=good_colors, colorrange=(0, 100))
    for i = 1:2
        for j = 1:length(all_labels)
            text!("$(cm[i, j])",
                  position=(i, j), align=(:center, :center), 
                  color=cm[i, j] > sum(cm[:, j]) / 2 ? :white : :black)
        end
    end
end

# ╔═╡ a5be5660-7a97-4730-bb72-938ce12c6b03
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
	#@assert sum(cm) == nrow(data_test)

	return cm
end

# ╔═╡ 9426e500-45c9-4bb3-bfce-85fc6a527d61
function viz_cm(svm, data_test::DataFrame, scaler)
	all_labels = viable_labels
	n_labels = length(all_labels)

	cm = generate_cm(svm, data_test, scaler, all_labels)

	fig = Figure()
	ax = Axis(fig[1, 1],
		  xticks=([1, 2], ["anomaly", "normal"]),
		  yticks=([i for i=1:n_labels], [SyntheticDataGen.reduced_labels[all_labels[i]] for i=1:n_labels]),
		  #xticklabelrotation=25.5,
		  ylabel="truth",
		  xlabel="prediction"
    )

	viz_cm!(ax, svm, data_test, scaler, gen_cm_flag=false, cm=cm)

    fig
end

# ╔═╡ ee8029cf-c6a6-439f-b190-cb297e0ddb70
viz_cm(mid_data["svm"], my_test, mid_data["data"].scaler)

# ╔═╡ 567335d9-8b3f-4bcb-b34c-3e655715b448
md"## Data visuals
"

# ╔═╡ 1aaadc59-deab-4374-969f-cddd1b24a025
md"### train
"

# ╔═╡ 7e45b82b-3c38-4734-9b58-fe0008747e66
#sensor response data train
AnomalyDetectionPlots.viz_decision_boundary(mid_data["svm"], mid_data["data"].scaler, mid_data["data"].data_train, default_lims=false, incl_contour=false,xlims=xlims, ylims=ylims)

# ╔═╡ d20826ad-6775-493a-a124-a2ab146c1381
md"### Test
"

# ╔═╡ dc4eedb5-758d-40f9-ba7b-c7ab71f5ec3b
#sensor response data test
AnomalyDetectionPlots.viz_decision_boundary(mid_data["svm"], mid_data["data"].scaler, mid_data["data"].data_test, default_lims=false, incl_contour=false, xlims=xlims, ylims=ylims)

# ╔═╡ 76e83d0b-da02-4ba3-a51e-3d570d330d3b
md"## Hyperparameter effects
"

# ╔═╡ ee91e0a9-605f-4d8c-8727-d6523e9a72c4
#AnomalyDetectionPlots.viz_ν_γ_effects(high_σm_data, high_σm_data["ν_opt, γ_opt"][1], high_σm_data["ν_opt, γ_opt"][2])

# ╔═╡ 8c426257-f4a5-4015-b39f-eab5e84d91ee
# check the f1 score to compare to other validation method(s)
f1_hypersphere = AnomalyDetection.performance_metric(mid_data["data"].y_test, mid_data["svm"].predict(mid_data["data"].X_test_scaled))

# ╔═╡ a2467d27-0664-43d3-8f22-46b0d2ad4a77
mid_data

# ╔═╡ af557f0c-9cb1-41ba-bcff-c1c95b08c560
md"## f1 score heatmap
"

# ╔═╡ 00d90c63-6f3e-4906-ad35-ba999439e253

begin
	σ_m_max = 1e-4	
	σ_H₂O_max = 1e-1
	
	AnomalyDetectionPlots.viz_f1_score_heatmap(σ_H₂O_max, 
											   σ_m_max, 
											   res=5, 
											   validation_method="hypersphere", 
   											   hyperparameter_method="bayesian", 
											   λ=0.5, 
											   n_avg=100,
											   gen_data_flag=false,
											   jld_file_location=jld_file_folder)
end


# ╔═╡ 6a46c6e8-2dfe-4745-b867-9192265b5d0d
md"## Learning curve
"

# ╔═╡ 627ed8d6-ac50-48e8-aa90-c75232c1bd64
begin
	#number of normal data points for training and test data
	num_normal_train_points_learning_curve = [10, 20, 50, 100, 150, 200, 300, 500]
	#number of each type of anomaly in test data
	# num_normal_test_points   = 100
	# num_anomaly_test_points  = 5

	# grab mid variance values
	σ_H₂O = σ_H₂O_vector[2]
	σ_m = σ_m_vector[2]
end

# ╔═╡ 59d2888f-fd1a-4644-b80f-e6e65ee771bc
#AnomalyDetection.simulate(num_normal_train_points_learning_curve, run_start=18, run_end=20, σ_m=σ_m, σ_H₂O=σ_H₂O)

# ╔═╡ 7ca56cf4-6045-4e1f-bc36-90c0bea8d200
the_data = AnomalyDetection.catenate_data()

# ╔═╡ 26d59d0a-2f1f-4bd6-b1e3-46c41c5db3da
lc = AnomalyDetection.score_stats(the_data)

# ╔═╡ 12df72c2-3228-472b-9e47-4610960ec608
AnomalyDetectionPlots.viz_learning_curve(lc, num_normal_train_points_learning_curve, σ_m=σ_m, σ_H₂O=σ_H₂O)

# ╔═╡ 82ae9099-37cc-4402-9963-62cc064849ad
md"# Alternative method: Knee
"

# ╔═╡ 51b0ebd4-1dec-4b35-bb15-cd3df906aca3
md"!!! example \"\" 
	Unsupervised hyperparameter validation method 2:

	density measure plot and maximum curvature"

# ╔═╡ 6ceab194-4861-4be1-901c-6713db5a4204
# ╠═╡ disabled = true
#=╠═╡
begin
	# according to paper K is optimally 0.05 * number of data points
	K = trunc(Int, 0.05 * num_normal_train_points)
	
	# use a density measure method to find optimal ν and γ
	ν_opt, γ_opt = AnomalyDetection.opt_ν_γ_by_density_measure_method(mid_data["data"].X_train_scaled, K)

	# train the anomaly detector
	svm = AnomalyDetection.train_anomaly_detector(mid_data["data"].X_train_scaled, ν_opt, γ_opt)

	(ν_opt, γ_opt)
end
  ╠═╡ =#

# ╔═╡ 9a9262d4-02ff-4d82-bb7b-8584e8b79022
# ╠═╡ disabled = true
#=╠═╡
AnomalyDetectionPlots.viz_density_measures(mid_data["data"].X_train_scaled, K)
  ╠═╡ =#

# ╔═╡ f0cb9b40-0ed8-450a-8f03-4f16ca65fa77
# ╠═╡ disabled = true
#=╠═╡
AnomalyDetectionPlots.viz_decision_boundary(svm, mid_data["data"].scaler, mid_data["data"].data_test)
  ╠═╡ =#

# ╔═╡ 47d6c332-632c-4880-9708-59e6fa187c6c
# ╠═╡ disabled = true
#=╠═╡
AnomalyDetectionPlots.viz_cm(svm, mid_data["data"].data_test, mid_data["data"].scaler)
  ╠═╡ =#

# ╔═╡ e4723de4-3a82-4c15-9057-c20b331259f7
# ╠═╡ disabled = true
#=╠═╡
AnomalyDetectionPlots.viz_decision_boundary(svm, mid_data["data"].scaler, mid_data["data"].data_train)
  ╠═╡ =#

# ╔═╡ 55640b9c-9a0a-4d0d-8c29-e67a8228edc2
# ╠═╡ disabled = true
#=╠═╡
# check the f1 score to compare to other validation method(s)
f1_density = AnomalyDetection.performance_metric(mid_data["data"].y_test, svm.predict(mid_data["data"].X_test_scaled))
  ╠═╡ =#

# ╔═╡ bbeec9a5-6260-4e8a-a444-a22a59898d22
md"!!! example \"\" 
	 Comparing F1 score between median different validation methods and calculating precision and recall."

# ╔═╡ 11e286be-d3a9-4896-a90c-fdd05fc35073
#=╠═╡
f1_density
  ╠═╡ =#

# ╔═╡ f8dab032-e446-4e6e-8022-39ad3dbb1042
f1_hypersphere

# ╔═╡ bfc27fe6-2f26-41b7-a614-2e0e354267bd
 AnomalyDetectionPlots.viz_cm(mid_data["svm"], mid_data["data"].data_test, mid_data["data"].scaler)

# ╔═╡ cef4a546-18b5-4c4d-a7c6-50caba148d35
begin
	correct_normal = 92 #not taken into account
	false_positives = 8.0 #predicted anomaly, but actually normal
	false_negatives = 0.0 + 5.0 + 1.0 + 0.0 #based on confusion matrix
	true_positives = 5.0 + 0.0 + 4.0 + 5.0 #"  "

	precision_sc = true_positives / (true_positives + false_positives)

	recall_sc = true_positives / (true_positives + false_negatives)
	
end

# ╔═╡ 3e8ca33e-e6e9-4a2f-b235-2221eb994ce3
precision_sc

# ╔═╡ e7cddd77-f1d0-44fd-87bb-e3af5b42558b
recall_sc

# ╔═╡ 3aab547c-8b00-48da-aa8e-3d51e804c5df
md"!!! example \"\" 
	f1 score testing for fun"

# ╔═╡ a1843a87-a8d3-40ab-9959-3e14d520a4d1
function f1(true_pstv, false_pstv, false_ngtv)
	prec = true_pstv / (true_pstv + false_pstv)
	rec = true_pstv / (true_pstv + false_ngtv)

	return 2* (prec*rec) / (prec + rec)
end

# ╔═╡ 923c9837-82ab-4071-b716-faa3565fa327
begin
	# test f1 score
	# the middle plot has 16 true positives, 9 false negatives, 10 false positives
	# it also has f1 0.62, lets test it
	pauls_f1 = f1(16, 10, 9)

	#perfect!
end

# ╔═╡ 211e8b05-6525-448e-80f2-f093e7488beb
f1(2, 15, 18)

# ╔═╡ b62fd403-cf0d-4ab5-94cf-291cefb0bbbc
f1(3, 17, 17)

# ╔═╡ 773793c4-021a-4aa8-9b13-c27f94e694b0
begin
yy_pred = [ 1,  1, 1, -1, -1, 1, -1, -1]
yy_true = [-1, -1, 1,  1,  1, 1,  1, -1]

	# how many predicted as anomalous that are actually anomalous?
	true_pstv = 1.0
	# how many predicted as normal, but actually anomalous?
	false_ngtv = 2.0
	# how many predicted as anomalous, but actually normal?
	false_pstv = 3.0

	println("paul's = $(f1(true_pstv, false_pstv, false_ngtv))")

	println("sklearn = $(f1_score(-yy_true, -yy_pred))")
	
end

# ╔═╡ 3755e438-0850-45d5-992d-e7911ddcb2df
md"# random anomaly detector test
"

# ╔═╡ 02b9e2a3-3b98-46b9-b107-661e2cadd555
function worst_f1(num_normal::Int, num_anomaly::Int, num_sims::Int=10000)
	p_anomaly = num_anomaly / (num_normal + num_anomaly)

	true_labels = vcat([true for i=1:num_anomaly], [false for i=1:num_normal])
	
	f1_sum = 0
	for s = 1:num_sims
		predict_labels = [rand() < p_anomaly for i=1:length(true_labels)]

		f1_sum += f1_score(true_labels, predict_labels)
	end

	return f1_sum / num_sims
end

# ╔═╡ 4c3c93e3-a595-4984-a091-6466a2b54756
worst_f1(100, 20)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
JLD = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"

[compat]
CSV = "~0.10.14"
CairoMakie = "~0.12.2"
ColorSchemes = "~3.25.0"
Colors = "~0.12.11"
DataFrames = "~1.6.1"
Distributions = "~0.25.109"
JLD = "~0.13.5"
JLD2 = "~0.4.48"
Makie = "~0.21.2"
PlutoUI = "~0.7.59"
PyCall = "~1.96.4"
ScikitLearn = "~0.7.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "e8370a0aec21b5f0d67d4a2d74035438025c824c"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "310b77648d38c223d947ff3f50f511d08690b8d5"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.3"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "19b98ee7e3db3b4eff74c5c9c72bf32144e24f10"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.5+0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "6c834533dc1fabd820c1db03c839bf97e45a3fab"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.14"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "9e8eaaff3e5951d8c61b7c9261d935eb27e0304b"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.12.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a2f1c8c668c8e3cb4cca4e57a8efdb09067bb3fd"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+2"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "71acdbf594aab5bbb2cec89b208c41b4c411e49f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.24.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "51cab8e982c5b598eea9c8ceaced4b58d9dd37c9"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelaunayTriangulation]]
deps = ["EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "1755070db557ec2c37df2664c75600298b0c1cfc"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.0.3"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "9c405847cc7ecda2dc921ccf18b47ca150d7317e"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.109"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ab3f7e1819dba9434a3a5126510c8fda3a4e7000"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0653c0a2396a6da5bc4766c43041ef5fd3efbe57"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.11.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "2493cdfd0740015955a8e46de4ef28f49460d8bc"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.3"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "801aef8228f7f04972e596b09d4dba481807c913"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.4"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "7c82e6a6cd34e9d935e9aa4051b66c6ff3af59ba"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.2+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "fc713f007cff99ff9e50accba6373624ddd33588"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.H5Zblosc]]
deps = ["Blosc", "HDF5"]
git-tree-sha1 = "d778420e524bcf56066e8c63c7aa315ae7269da2"
uuid = "c8ec2601-a99c-407f-b158-e79c03c2f5f7"
version = "0.1.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ca0f6bf568b4bfc807e7537f081c81e35ceca114"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.10.0+0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be50fe8df3acbffa0274a744f1a99d29c45a57f4"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "90709228dc114e599a2b62b7d23482a4f50938ee"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.13"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

    [deps.IntervalArithmetic.weakdeps]
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD]]
deps = ["Compat", "FileIO", "H5Zblosc", "HDF5", "Printf"]
git-tree-sha1 = "e42f32690d41f758e126a48ee43459ef91179d1f"
uuid = "4138dd39-2aa7-5051-a626-17a0bb65d9c8"
version = "0.13.5"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Reexport", "Requires", "TranscodingStreams", "UUIDs", "Unicode"]
git-tree-sha1 = "bdbe8222d2f5703ad6a7019277d149ec6d78c301"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.48"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c84a835e1a09b289ffcd2271bf2a337bbdda6637"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.3+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "70c5da094887fd2cae843b8db33920bac4b6f07d"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6c26c5e8a4203d43b5497be3ec5d4e0c3cde240a"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.4+0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "80b2833b56d466b3858d565adcd16a4a05f2089b"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.1.0+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "4099bb6809ac109bfc17d521dad33763bcf026b7"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.1+1"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "8c35d5420193841b2f367e658540e8d9e0601ed0"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.4.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "ec3a60c9de787bc6ef119d13e07d4bfacceebb83"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.2"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "c1c9da1a69f6c635a60581c98da252958c844d70"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.2"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "1865d0b8a2d91477c8b16b49152a32764c7b1f5f"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f12a29c4400ba812841c6ace3f4efbb6dbb3ba01"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "e64b4f5ea6b7389f6f046d13d4896a8f9c1ba71e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e25c1778a98e34219a00455d6e4384e017ea9762"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.6+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cb5a2ab6763464ae0f19c86c56c63d4a2b0f5bda"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.52.2+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "66b20dd35966a748321d3b2537c4584cf40387c7"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "763a8ceb07833dd51bb9e3bbca372de32c0605ad"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.0"

[[deps.PtrArrays]]
git-tree-sha1 = "f011fbb92c4d401059b2212c05c0601b70f8b759"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d483cd324ce5cf5d61b77930f0bbd6cb61927d21"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.2+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "2803cab51702db743f3fda07dd1745aadfbf43bd"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.5.0"

[[deps.ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "3df098033358431591827bb86cada0bed744105a"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.7.0"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "90b4f68892337554d31cdcdbe19e48989f26c7e6"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.3"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "6e00379a24597be4ae1ee6b2d882e15392040132"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.5"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "bc7fd5c91041f44636b2c134041f7e5263ce58ae"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "a947ea21087caba0a798c5e494d0bb78e3a1a3a0"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.9"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "dd260903fdabea27d9b6021689b3cd5401a57748"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.20.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "52ff2af32e591541550bd753c0da8b9bc92bb9d9"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.7+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e678132f07ddb5bfa46857f0d7620fb9be675d3b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46bf7be2917b59b761247be3f317ddf75e50e997"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.2+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─1784c510-5465-11ec-0dd1-13e5a66e4ce6
# ╠═d090131e-6602-4c03-860c-ad3cb6c7844a
# ╠═3ba4e1e5-3187-4811-be09-d990973abc77
# ╠═0a6fe423-c3be-4a75-aa27-dfb84fde7fef
# ╠═3e7c36ca-8345-40fb-b199-34fe49dea73e
# ╠═4745788b-d360-4305-b44b-8d0fca2aeb4f
# ╠═31f71438-ff2f-49f9-a801-3a6489eaf271
# ╠═d578dbf8-dc4d-4a80-a68f-e12284a75953
# ╠═5d920ea0-f04d-475f-b05b-86e7b199d7e0
# ╟─52ac8252-51a2-484c-9dac-bbdafa40de41
# ╠═21589bf0-e7f0-4cf1-b082-e236cf6b3221
# ╠═1e30612e-7bcd-47dc-a1fb-1e127aad4a55
# ╟─4348a594-aa99-45dd-af3f-f3b61a4e8142
# ╟─e5eede17-08bd-4120-846e-36a3058c003e
# ╟─ebf79f0c-8399-42bf-b790-d4934906ede0
# ╟─853390f9-6519-4df3-aa24-7b337142dbe4
# ╠═075d4a2f-cf63-47b1-b309-14df97672a65
# ╠═4b1759a7-eba1-4de5-8d6a-38106f3301c9
# ╠═a6700f58-c006-4893-8437-8e6c2b3048f7
# ╟─2083f6c8-429c-40bb-a029-f9d3131886e7
# ╟─5a4c66d9-0166-4c13-8a30-02ed6481b6fe
# ╠═ec8c2bbc-f492-4a22-80a8-f125ea048b34
# ╟─7a43a79f-de0f-469b-b8d9-a25b4c3f9180
# ╠═a6c181fe-e73d-46af-a1e6-5b4740ae89e4
# ╟─8317251c-69ad-42d8-90df-6e2a5cc94b13
# ╠═bfe24d5a-de4d-4634-ad5d-0c093a17135a
# ╠═5c4dd3f5-f62c-4d76-b03b-2acfd992969e
# ╠═43945e20-929f-4045-8c44-66eb4a149483
# ╠═38f64f12-7eb3-4029-ad94-12307a5ee885
# ╟─77382f3e-98b6-4aef-b946-8375018c3c3e
# ╠═6f53b700-6eba-487b-b91b-085d6e4d38b9
# ╠═3117881e-08e5-435b-b088-be9973bec8aa
# ╟─7af3b1f6-2c57-40c4-a841-961dd039090a
# ╠═7990ef58-1e45-44d0-8add-ba410a48dc98
# ╟─97a7e102-1a87-4364-9835-c7ed370f573c
# ╠═86ba61e6-0633-431f-93a1-b53a8de9dd46
# ╠═ccbe1d74-df04-4dbf-9ee4-683890963892
# ╠═6e278c3e-45a3-4aa8-b904-e3dfa73615d5
# ╠═b67f7643-3994-4e04-8a5c-e748c3c54346
# ╠═13f4acb4-4434-4ac3-97be-900be400d908
# ╠═c5e13fe4-3a7e-4aa0-9550-d56c18f673bf
# ╠═95344881-f912-412d-8d9b-42b8bb3452b9
# ╠═0b416525-f0b8-496f-98e6-90e6a6f5cbcd
# ╠═bb9b1c23-db1e-48bb-9b47-1ba239470123
# ╟─c930cd71-446c-47f5-8bed-15602afa2304
# ╠═2041cc5f-d583-4fca-bf75-6bee5d6d876b
# ╠═9426e500-45c9-4bb3-bfce-85fc6a527d61
# ╠═8b9d65e9-2c28-4eca-a44e-1ae051300777
# ╠═a5be5660-7a97-4730-bb72-938ce12c6b03
# ╠═ee8029cf-c6a6-439f-b190-cb297e0ddb70
# ╟─567335d9-8b3f-4bcb-b34c-3e655715b448
# ╟─1aaadc59-deab-4374-969f-cddd1b24a025
# ╠═7e45b82b-3c38-4734-9b58-fe0008747e66
# ╟─d20826ad-6775-493a-a124-a2ab146c1381
# ╠═dc4eedb5-758d-40f9-ba7b-c7ab71f5ec3b
# ╟─76e83d0b-da02-4ba3-a51e-3d570d330d3b
# ╠═ee91e0a9-605f-4d8c-8727-d6523e9a72c4
# ╠═8c426257-f4a5-4015-b39f-eab5e84d91ee
# ╠═a2467d27-0664-43d3-8f22-46b0d2ad4a77
# ╟─af557f0c-9cb1-41ba-bcff-c1c95b08c560
# ╠═00d90c63-6f3e-4906-ad35-ba999439e253
# ╟─6a46c6e8-2dfe-4745-b867-9192265b5d0d
# ╠═627ed8d6-ac50-48e8-aa90-c75232c1bd64
# ╠═59d2888f-fd1a-4644-b80f-e6e65ee771bc
# ╠═7ca56cf4-6045-4e1f-bc36-90c0bea8d200
# ╠═26d59d0a-2f1f-4bd6-b1e3-46c41c5db3da
# ╠═12df72c2-3228-472b-9e47-4610960ec608
# ╟─82ae9099-37cc-4402-9963-62cc064849ad
# ╟─51b0ebd4-1dec-4b35-bb15-cd3df906aca3
# ╠═6ceab194-4861-4be1-901c-6713db5a4204
# ╠═9a9262d4-02ff-4d82-bb7b-8584e8b79022
# ╠═f0cb9b40-0ed8-450a-8f03-4f16ca65fa77
# ╠═47d6c332-632c-4880-9708-59e6fa187c6c
# ╠═e4723de4-3a82-4c15-9057-c20b331259f7
# ╠═55640b9c-9a0a-4d0d-8c29-e67a8228edc2
# ╟─bbeec9a5-6260-4e8a-a444-a22a59898d22
# ╠═11e286be-d3a9-4896-a90c-fdd05fc35073
# ╠═f8dab032-e446-4e6e-8022-39ad3dbb1042
# ╠═bfc27fe6-2f26-41b7-a614-2e0e354267bd
# ╠═cef4a546-18b5-4c4d-a7c6-50caba148d35
# ╠═3e8ca33e-e6e9-4a2f-b235-2221eb994ce3
# ╠═e7cddd77-f1d0-44fd-87bb-e3af5b42558b
# ╟─3aab547c-8b00-48da-aa8e-3d51e804c5df
# ╠═923c9837-82ab-4071-b716-faa3565fa327
# ╠═a1843a87-a8d3-40ab-9959-3e14d520a4d1
# ╠═211e8b05-6525-448e-80f2-f093e7488beb
# ╠═b62fd403-cf0d-4ab5-94cf-291cefb0bbbc
# ╠═773793c4-021a-4aa8-9b13-c27f94e694b0
# ╟─3755e438-0850-45d5-992d-e7911ddcb2df
# ╠═02b9e2a3-3b98-46b9-b107-661e2cadd555
# ╠═4c3c93e3-a595-4984-a091-6466a2b54756
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
