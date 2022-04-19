push!(LOAD_PATH, joinpath(pwd(), "src"))
using FruitRipeningRoom, Distributions

noise = Normal(0, 0.01)

gas_comp = setup_gas_comp_distn(0.1, "normal")
gas_comp = setup_gas_comp_distn(0.1, "C₂H₄ off")
println(gas_comp)

data = gen_gas_comps(100, "normal", 0.1)
sensor_response!(data, noise)
data = gen_data(20, 5, 0.1, 0.01)
println(data)
