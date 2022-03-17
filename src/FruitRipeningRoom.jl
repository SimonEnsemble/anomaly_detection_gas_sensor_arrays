module FruitRipeningRoom

using Distributions

export setup_normal_gas_comp_distn


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

function setup_normal_gas_comp_distn(σ_water::Float64)
    return GasComp(
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
end

function setup_C₂H₄_off_gas_comp_distn(σ_water::Float64)
    gcd = setup_normal_gas_comp_distn(σ_water)
    gcd.f_C₂H₄ = Uniform(0.0e-6, 10.0e-6)
    return gcd
end

end
