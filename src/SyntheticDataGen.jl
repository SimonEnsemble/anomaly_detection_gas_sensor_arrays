module SyntheticDataGen

using Distributions, DataFrames, JLD2, ColorSchemes, CairoMakie, Revise

export GasCompDistribution, setup_gas_comp_distn, gen_gas_comps, sensor_response!, gen_data

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
anomaly_labels = ["CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup"]
viable_labels = vcat(["normal"], anomaly_labels)
gases = ["C₂H₄", "CO₂", "H₂O"]
mofs = ["ZIF-71", "ZIF-8"]
henry_data = load("henry_coeffs.jld2")["henry_data"]
label_to_color = Dict(zip(
	["normal", "CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"],
	ColorSchemes.Dark2_6)
    )

function setup_gas_comp_distn(σ_H₂O::Float64, label::String)
    if ! (label in viable_labels)
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
        # H₂O: 85% RH on average
        truncated(
            Normal(0.85 * p_H₂O_vapor, σ_H₂O * p_H₂O_vapor), 
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
        gas_comp_distn.f_H₂O = Uniform(0.2 * p_H₂O_vapor, 0.75 * p_H₂O_vapor)
    end

    return gas_comp_distn
end

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

function gen_data(n_normal::Int, n_anomaly::Int, σ_H₂O::Float64, σ_m::Float64)
    noise = Normal(0.0, σ_m)
    data = gen_gas_comps(n_normal, "normal", σ_H₂O)
    for label in anomaly_labels
        append!(data, 
            gen_gas_comps(n_anomaly, label, σ_H₂O)
        )    
    end
    sensor_response!(data, noise)
    return data
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
	for water_anomaly in [true, false]
		ids = (data[:, "label"] .== "low humidity") .== water_anomaly
    	hist!(data[ids, "p H₂O [bar]"] / SyntheticDataGen.p_H₂O_vapor)
	end
    ylims!(0.0, nothing)
    fig
end

end
