module FruitRipeningRoom

using Distributions, DataFrames

export GasCompDistribution, setup_gas_comp_distn, gen_synthetic_gas_compositions, generate_sensor_data

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
viable_labels = ["normal", "CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup", "low humidity"]
gases = ["C₂H₄", "CO₂", "H₂O"]
mofs = ["ZIF-71", "ZIF-8"]

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

function gen_synthetic_gas_compositions(label::String, n_compositions::Int, σ_H₂O::Float64)
    gas_comp_distn = setup_gas_comp_distn(σ_H₂O, label)
    data = DataFrame("p C₂H₄ [bar]" => zeros(n_compositions), 
                     "p CO₂ [bar]" => zeros(n_compositions),
                     "p H₂O [bar]" => zeros(n_compositions),
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

#=
function response(composition::Dict{String, Float64})
	response = Dict()
	for mof in mofs
		response["m $(mof) [g/g]"] = 0.0
		for gas in gases
			H_mof_gas = henry_data[mof][gas]["henry coef [g/(g-bar)]"]
			p_gas     = composition["p $gas [bar]"]
			response["m $(mof) [g/g]"] += H_mof_gas * p_gas
		end
	end
	return response
end



function generate_normal_sensor_data(n_gas_compositions::Int, δ::Normal{Float64}, σ_H₂O::Float64, henry_data::Dict)
	sensor_data = gen_synthetic_gas_compositions("normal", n_gas_compositions, σ_H₂O)
    for mof in mofs
		sensor_data[!, "m $mof [g/g]"] = zeros(n_gas_compositions)

        for g = 1:n_gas_compositions
            for gas in gases
                H_mof_gas = henry_data[mof][gas]["henry coef [g/(g-bar)]"]
                p_gas = sensor_data[g, "p $gas [bar]"]
                sensor_data[g, "m $mof [g/g]"] +=  H_mof_gas * p_gas
            end

            sensor_data[g, "m $mof [g/g]"] += rand(δ)
        end
	end

	return sensor_data
end
=#

function generate_sensor_data(n_gas_compositions::Int, label::String, δ::Normal{Float64}, σ_H₂O::Float64, henry_data::Dict)
	sensor_data = gen_synthetic_gas_compositions(label, n_gas_compositions, σ_H₂O)
    for mof in mofs
		sensor_data[!, "m $mof [g/g]"] = zeros(n_gas_compositions)

        for g = 1:n_gas_compositions
            for gas in gases
                H_mof_gas = henry_data[mof][gas]["henry coef [g/(g-bar)]"]
                p_gas = sensor_data[g, "p $gas [bar]"]
                sensor_data[g, "m $mof [g/g]"] +=  H_mof_gas * p_gas
            end

            sensor_data[g, "m $mof [g/g]"] += rand(δ)
        end
	end

	return sensor_data
end

end
