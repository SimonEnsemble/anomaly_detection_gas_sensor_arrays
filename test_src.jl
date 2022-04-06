push!(LOAD_PATH, joinpath(pwd(), "src"))
using FruitRipeningRoom

gas_comp = setup_gas_comp_distn(0.1, "normal")
gas_comp = setup_gas_comp_distn(0.1, "C₂H₄ off")
println(gas_comp)

data = gen_synthetic_gas_compositions("normal", 100, 0.1)
println(data)
