module FruitRipeningRoom

using Distributions

export GasCompDistribution, setup_gas_comp_distn


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

function setup_gas_comp_distn(σ_water::Float64, label::String)
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
            Normal(0.85 * p_H₂O_vapor, σ_water * p_H₂O_vapor), 
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

end
