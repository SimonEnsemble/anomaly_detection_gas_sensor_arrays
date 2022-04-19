module SyntheticDataGen

using Distributions, DataFrames, JLD2

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
anomaly_labels = ["CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"]
viable_labels = vcat(["normal"], anomaly_labels)
gases = ["C₂H₄", "CO₂", "H₂O"]
mofs = ["ZIF-71", "ZIF-8"]
henry_data = load("henry_coeffs.jld2")["henry_data"]

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
		gas_comp_distn.f_C₂H₄ = Uniform(300e-6, 1000e-6)
    elseif label == "low humidity"
        gas_comp_distn.f_H₂O = Uniform(0.7*p_H₂O_vapor, 0.8*p_H₂O_vapor)
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

end
