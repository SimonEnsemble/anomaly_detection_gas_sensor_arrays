module SyntheticDataGen

using Distributions, DataFrames, JLD2, ColorSchemes, CairoMakie

#=
    DEFINE DISTRIBUTIONS OF GAS COMPOSITIONS IN THE FRUIT RIPENING ROOM
    under normal and anomalous circumstances.
=#
# units: bar
mutable struct GasCompDistribution
	f_C₂H₄::Distribution
	f_CO₂::Distribution
	f_H₂O::Distribution
end


# vapor pressure of water
p_H₂O_vapor = 3.1690 * 0.01 # bar
anomaly_labels = ["CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"]
viable_labels = vcat(["normal"], anomaly_labels)
gases = ["C₂H₄", "CO₂", "H₂O"]
mofs = ["ZIF-71", "ZIF-8"]
henry_data = load("henry_coeffs.jld2")["henry_data"]
label_to_color = Dict(zip(
	["normal", "CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"],
	ColorSchemes.Dark2_6)
    )
reduced_labels = Dict("normal" => "normal", 
                    "CO₂ buildup" => "CO₂ ↑", 
                    "C₂H₄ buildup" => "C₂H₄ ↑", 
                    "C₂H₄ off" => "C₂H₄ off", 
                    "CO₂ & C₂H₄ buildup" => "CO₂ & C₂H₄ ↑", 
                    "low humidity" => "H₂O ↓",
                    "anomalous" => "anomaly")

function setup_gas_comp_distn(σ_H₂O::Float64, label::String; only_water::Bool=false)
    if ! (label in viable_labels) && !only_water
        error(label * "not a viable label")
    end
    
    # set up normal gas comp dis'tn. units are bar.
    gas_comp_distn = GasCompDistribution(
        # C₂H₄ at 150 ppm
        truncated(
            Normal(150e-6, 20e-6), 
            0.0, Inf
        ),
        # CO₂. 410 ppm to 5000 ppm
        Uniform(400.0e-6, 5000.0e-6),
        # H₂O: 90 % RH on average
        truncated(
            Normal(0.90 * p_H₂O_vapor, σ_H₂O * p_H₂O_vapor), 
            0.0, Inf
        )
    )
    
    # modify if anomalous
    if label == "C₂H₄ off"
        gas_comp_distn.f_C₂H₄ = Uniform(0.0e-6, 10.0e-6)
    elseif label == "C₂H₄ buildup"
		gas_comp_distn.f_C₂H₄ = Uniform(300e-6, 2000e-6)
    elseif label == "CO₂ buildup"
		gas_comp_distn.f_CO₂ = Uniform(7500e-6, 20000e-6)
    elseif label == "CO₂ & C₂H₄ buildup"
		gas_comp_distn.f_CO₂ = Uniform(7500e-6, 20000e-6)
		gas_comp_distn.f_C₂H₄ = Uniform(300e-6, 2000e-6)
    elseif label == "low humidity"
        gas_comp_distn.f_H₂O = Uniform(0.5 * p_H₂O_vapor, 0.80 * p_H₂O_vapor)
    end

    return gas_comp_distn
end

function gen_gas_comps(n_compositions::Int, label::String, σ_H₂O::Float64; only_water::Bool=false)
    gas_comp_distn = setup_gas_comp_distn(σ_H₂O, label, only_water=only_water)

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

# add columns for sensor response
function sensor_response!(data::DataFrame, noise::Distribution)
    n_compositions = nrow(data)
    for mof in mofs
		data[:, "m $mof [g/g]"] = zeros(n_compositions)
        for g = 1:n_compositions
            # add up contributions from the gases
            for gas in gases
                H_mof_gas = henry_data[mof][gas]["henry coef [g/(g-bar)]"]
                p_gas     = data[g, "p $gas [bar]"]
                data[g, "m $mof [g/g]"] +=  H_mof_gas * p_gas
            end
            
            # add noise
            data[g, "m $mof [g/g]"] += rand(noise)
        end
	end
	return data
end

function gen_data(n_normal::Int, n_anomaly::Int, σ_H₂O::Float64, σ_m::Float64; only_water::Bool=false)
    noise = Normal(0.0, σ_m)
    
	if only_water
		data = gen_gas_comps(n_anomaly, "low humidity", σ_H₂O, only_water=only_water)
		sensor_response!(data, noise)
		return data
	else
		data = gen_gas_comps(n_normal, "normal", σ_H₂O)
		for label in anomaly_labels
			if label != "low humidity"
				append!(data, 
					gen_gas_comps(n_anomaly, label, σ_H₂O)
				)    
			end
		end

		sensor_response!(data, noise)
		return data
	end

end

function viz_C2H4_CO2_composition(data::DataFrame)
    fig = Figure(resolution=(600, 600))
    # create panels
    ax_main  = Axis(fig[2, 1],
                xlabel="p, C₂H₄ [ppm]",
                ylabel="p, CO₂ [ppm]"
    )
    ax_top   = Axis(fig[1, 1], ylabel="density", aspect=AxisAspect(2))
    ax_right = Axis(fig[2, 2], xlabel="density", aspect=AxisAspect(0.5))
    hidedecorations!(ax_top, grid=false, label=false)
    hidedecorations!(ax_right, grid=false, label=false)
    linkyaxes!(ax_main, ax_right)
    linkxaxes!(ax_main, ax_top)
    for c in 1:2
        colsize!(fig.layout, c, Relative(.5))
        rowsize!(fig.layout, c, Relative(.5))
    end
    ylims!(ax_right, 0, nothing)

    for data_g in groupby(data, :label)
        label = data_g[1, "label"]

        scatter!(ax_main,
                 data_g[:, "p C₂H₄ [bar]"] * 1e6,
                 data_g[:, "p CO₂ [bar]"] * 1e6,
                 marker=label == "normal" ? :circle : :x,
                 strokewidth=1,
                 label=label,
                 strokecolor=label_to_color[label],
                 color=(:white, 0.0)
        )

        hist!(ax_top,
             data_g[:, "p C₂H₄ [bar]"] * 1e6,
             label=label,
             probability=true,
             color=(label_to_color[label], 0.5)
        )
        hist!(ax_right,
              data_g[:, "p CO₂ [bar]"] * 1e6,
              direction=:x,
              label=label,
              probability=true,
              color=(label_to_color[label], 0.5)
        )
    end
    # ylims!(ax_top, 0, 0.01)
    # xlims!(ax_right, 0, 0.0005)

    # create legend, save and display
    leg = Legend(fig[1,2], ax_main)
    fig
end

function viz_H2O_compositions(data::DataFrame)
    fig = Figure()
    ax = Axis(fig[1, 1],
              xlabel="p, H₂O [relative humidity]",
              ylabel="# compositions")
			  
	for water_anomaly in [false]
		ids = (data[:, "label"] .== "low humidity") .== water_anomaly
    	hist!(data[ids, "p H₂O [bar]"] / SyntheticDataGen.p_H₂O_vapor)
	end
    ylims!(0.0, nothing)
    fig
end

function viz_C2H4_CO2_H2O_density_distributions(σ_H₂O)
	gasses = SyntheticDataGen.gases
	labels = SyntheticDataGen.viable_labels

    #establish axes and figs for the grid
	fig  = Figure(resolution = (2600, 2000))

	axes = zeros(length(labels), length(gasses))
	axes = convert(Array{Any, 2}, axes)

	for (i, label) in enumerate(labels)
		for (j, gas) in enumerate(gasses)
			gas == "CO₂" ? num_ticks = 4 : num_ticks = 6
			
			if j==1 && i==length(labels)
				axes[i, j] = Axis(fig[i, j], 
								  yticklabelsvisible = false, 
								  ylabel="density", 
								  xlabel="ppm", 
								  xlabelsize=62, 
								  ylabelsize=62,
							      xticklabelsize=40,
								  xticks=WilkinsonTicks(num_ticks))
			elseif j==1
				axes[i, j] = Axis(fig[i, j], 
								  yticklabelsvisible = false, 
								  ylabel="density", 
								  xticklabelsvisible = false, 
								  ylabelsize=62)
			elseif i==length(labels)
				if gas=="H₂O" 
					axes[i, j] = Axis(fig[i, j], 
									  yticklabelsvisible = false, 
									  xlabel="RH", 
								      xlabelsize=62,
									  xticklabelsize=40)
				else
					axes[i, j] = Axis(fig[i, j], 
									  yticklabelsvisible = false, 
									  xlabel="ppm", 
									  xlabelsize=62,
								      xticklabelsize=40,
								      xticks=WilkinsonTicks(num_ticks))
				end
			else
				axes[i, j] = Axis(fig[i, j], 
								  yticklabelsvisible = false, 
								  xticklabelsvisible = false)
			end
		end
	end
	figs = [fig[i, j] for i in 1:length(labels), j in 1:length(gasses)]

    #top gas labels
	for (label, layout) in zip(gasses, figs[1, 1:length(gasses)])
		Label(layout[1, 1, Top()], 
			 label,
			 fontsize = 120,
			 padding = (0, 0, 25, 0),
			 halign = :center)
	end

    #left normal/anomaly type labels
	for (label, layout) in zip(labels, figs[1:length(labels), 1])
		Label(layout[1, 1, Left()], 
			 SyntheticDataGen.reduced_labels[label],
			 fontsize = 80,
			 padding = (0, 120, 0, 0),
			 valign = :center,
			 rotation = 0.0)
	end
		
	for (i, label) in enumerate(labels)
		for (j, gas) in enumerate(gasses)
			#generate a distribution
			gas_distr = SyntheticDataGen.setup_gas_comp_distn(σ_H₂O, label)
			if gas == "H₂O" 
				distr = gas_distr.f_H₂O
			else
				gas == "CO₂" ? distr = gas_distr.f_CO₂ : distr = gas_distr.f_C₂H₄
			end

			#create distribution densities and corresponding pressures
			p_min, p_max = quantile.(distr, [0.001, 0.999])
			ps = range(p_min, p_max, 1000)
			densities = [pdf(distr, ps[i]) for i=1:length(ps)]
			gas=="H₂O" ? ps = ps/SyntheticDataGen.p_H₂O_vapor : ps = ps * 1e6

			#distribution plot using band for shading
			band!(axes[i, j], 
				  ps, 
				  [0 for i=1:length(ps)], 
				  densities, 
				  color=(SyntheticDataGen.label_to_color[label], 0.20))
			lines!(axes[i, j], 
				   ps, 
				   densities,
				   color=SyntheticDataGen.label_to_color[label],				   	 
         		   strokewidth=5,
				   strokecolor=SyntheticDataGen.label_to_color[label])
			
			#line to zero for uniform distr
			lines!(axes[i, j], 
				   [ps[1], ps[1]], 
				   [densities[1], 0],
				   color=SyntheticDataGen.label_to_color[label],				   	 
         		   linewidth=2.5,
				   strokecolor=SyntheticDataGen.label_to_color[label])
			lines!(axes[i, j], 
				   [ps[end], ps[end]], 
				   [densities[end], 0],
				   color=SyntheticDataGen.label_to_color[label],				   
  				   linewidth=2.5,
				   strokecolor=SyntheticDataGen.label_to_color[label])
			#gray axis
			if gas == "H₂O"
				lines!(axes[i, j], 
					   [0.85, 0.85, 0.95], 
					   [1258, 0, 0],
					   color=:grey,				   	 
	         		   linewidth=0.5)
			elseif gas == "CO₂"
				lines!(axes[i, j], 
					   [0, 0, 2.0*10^4], 
					   [220, 0, 0],
					   color=:grey,				   	 
	         		   linewidth=0.5)
			elseif gas == "C₂H₄"
				lines!(axes[i, j], 
					   [0, 0, 2.0*10^3], 
					   [20000, 0, 0],
					   color=:grey,				   	 
	         		   linewidth=0.5)
				#end
			end
		end
	end
	
	colgap!(fig.layout, Relative(0.05))
	linkyaxes!(axes[1, 1], axes[2, 1], axes[3, 1], axes[5, 1])
	
	for j = 1:length(gasses)
		linkxaxes!(axes[1, j], axes[2, j], axes[3, j], axes[4, j], axes[5, j])
		if gasses[j] != "C₂H₄"
			linkyaxes!(axes[1, j], axes[2, j], axes[3, j], axes[4, j], axes[5, j])
		end
	end

	save("data_distributions.pdf", fig)
	return fig
end

end
