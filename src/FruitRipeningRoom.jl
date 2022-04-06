module FruitRipeningRoom

using Distributions, DataFrames

export GasCompDistribution, setup_gas_comp_distn, gen_synthetic_gas_compositions

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
viable_labels = ["normal", "CO₂ buildup", "C₂H₄ buildup", "C₂H₄ off", "CO₂ & C₂H₄ buildup"]
gases = ["C₂H₄", "CO₂", "H₂O"]

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
    end

    return gas_comp_distn
end

function gen_synthetic_gas_compositions(label::String, n_compositions::Int, σ_H₂O::Float64)
    gas_comp_distn = setup_gas_comp_distn(σ_H₂O, label)
    data = DataFrame("C₂H₄ [bar]" => zeros(n_compositions), 
                     "CO₂ [bar]" => zeros(n_compositions),
                     "H₂O [bar]" => zeros(n_compositions),
                     "label" => [label for _ = 1:n_compositions]
                    )
    for i = 1:n_compositions
        data[i, "C₂H₄ [bar]"] = rand(gas_comp_distn.f_C₂H₄)
        data[i, "CO₂ [bar]"]  = rand(gas_comp_distn.f_CO₂)
        data[i, "H₂O [bar]"]  = rand(gas_comp_distn.f_H₂O)
        data[i, "label"]      = label
    end
    return data
end

end
